"""Full fine-tuning loop with warmup + cosine schedule and early stopping on validation loss."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrainingOutcome:
	best_epoch: int
	best_validation_loss: float
	epochs_completed: int
	stopped_early: bool
	training_seconds: float
	history: List[Dict[str, float]] = field(default_factory=list)


class FineTuningTrainer:
	def __init__(
		self,
		model,
		training_settings: Dict[str, Any],
		device=None,
	) -> None:
		import torch

		self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = model.to(self.device)
		self.max_epochs = training_settings["max_epochs"]
		self.early_stopping_patience = training_settings["early_stopping_patience"]
		self.early_stopping_min_delta = training_settings.get("early_stopping_min_delta", 0.0)
		self.warmup_epochs = training_settings.get("warmup_epochs", 0)
		self.gradient_clip_norm = training_settings.get("gradient_clip_norm")
		self.learning_rate = training_settings["learning_rate"]
		self.weight_decay = training_settings["weight_decay"]
		self.loss_function = torch.nn.CrossEntropyLoss()
		self.autocast_dtype = self._pick_autocast_dtype(training_settings.get("mixed_precision", True))
		# Loss scaling is only needed for float16; bfloat16 has enough range on its own
		self.gradient_scaler = torch.amp.GradScaler(
			device=self.device.type,
			enabled=self.autocast_dtype == torch.float16,
		)

	def _pick_autocast_dtype(self, mixed_precision_enabled: bool):
		import torch

		if not mixed_precision_enabled or self.device.type != "cuda":
			return None
		return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

	def _autocast(self):
		import torch

		return torch.autocast(
			device_type=self.device.type,
			dtype=self.autocast_dtype or torch.float32,
			enabled=self.autocast_dtype is not None,
		)

	def _build_optimizer_and_scheduler(self, steps_per_epoch: int):
		import torch

		# Biases and norm weights are usually left out of weight decay for ViTs
		decayed_parameters, undecayed_parameters = [], []
		for parameter in self.model.parameters():
			if not parameter.requires_grad:
				continue
			if parameter.ndim <= 1:
				undecayed_parameters.append(parameter)
			else:
				decayed_parameters.append(parameter)
		optimizer = torch.optim.AdamW(
			[
				{"params": decayed_parameters, "weight_decay": self.weight_decay},
				{"params": undecayed_parameters, "weight_decay": 0.0},
			],
			lr=self.learning_rate,
		)

		total_steps = max(1, self.max_epochs * steps_per_epoch)
		warmup_steps = self.warmup_epochs * steps_per_epoch

		def learning_rate_multiplier(current_step: int) -> float:
			if current_step < warmup_steps:
				return (current_step + 1) / warmup_steps
			progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
			return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
		return optimizer, scheduler

	def fit(self, train_loader, validation_loader) -> TrainingOutcome:
		optimizer, scheduler = self._build_optimizer_and_scheduler(len(train_loader))

		best_validation_loss = math.inf
		best_epoch = 0
		best_model_state = None
		epochs_without_improvement = 0
		history: List[Dict[str, float]] = []
		stopped_early = False
		training_start_time = time.time()

		for epoch_number in range(1, self.max_epochs + 1):
			epoch_start_time = time.time()
			train_loss, train_accuracy = self._run_training_epoch(train_loader, optimizer, scheduler)
			validation_loss, validation_accuracy = self._run_validation_epoch(validation_loader)

			history.append({
				"epoch": epoch_number,
				"train_loss": round(train_loss, 5),
				"train_accuracy": round(train_accuracy, 4),
				"validation_loss": round(validation_loss, 5),
				"validation_accuracy": round(validation_accuracy, 4),
				"learning_rate": optimizer.param_groups[0]["lr"],
				"epoch_seconds": round(time.time() - epoch_start_time, 1),
			})
			print(
				f"    epoch {epoch_number:>3}/{self.max_epochs}  "
				f"train_loss={train_loss:.4f}  val_loss={validation_loss:.4f}  "
				f"val_acc={validation_accuracy:.4f}",
				flush=True,
			)

			if validation_loss < best_validation_loss - self.early_stopping_min_delta:
				best_validation_loss = validation_loss
				best_epoch = epoch_number
				best_model_state = {
					name: tensor.detach().cpu().clone()
					for name, tensor in self.model.state_dict().items()
				}
				epochs_without_improvement = 0
			else:
				epochs_without_improvement += 1
				if epochs_without_improvement >= self.early_stopping_patience:
					stopped_early = True
					print(f"    early stop: no val_loss improvement for {self.early_stopping_patience} epochs", flush=True)
					break

		if best_model_state is not None:
			self.model.load_state_dict(best_model_state)

		return TrainingOutcome(
			best_epoch=best_epoch,
			best_validation_loss=round(best_validation_loss, 5),
			epochs_completed=len(history),
			stopped_early=stopped_early,
			training_seconds=round(time.time() - training_start_time, 1),
			history=history,
		)

	def _run_training_epoch(self, train_loader, optimizer, scheduler) -> Tuple[float, float]:
		import torch

		self.model.train()
		summed_loss, correct_predictions, seen_samples = 0.0, 0, 0
		for images, labels in train_loader:
			images = images.to(self.device, non_blocking=True)
			labels = labels.to(self.device, non_blocking=True)

			optimizer.zero_grad(set_to_none=True)
			with self._autocast():
				logits = self.model(images)
				loss = self.loss_function(logits, labels)

			self.gradient_scaler.scale(loss).backward()
			if self.gradient_clip_norm:
				self.gradient_scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
			self.gradient_scaler.step(optimizer)
			self.gradient_scaler.update()
			scheduler.step()

			batch_size = labels.size(0)
			summed_loss += loss.item() * batch_size
			correct_predictions += (logits.argmax(dim=1) == labels).sum().item()
			seen_samples += batch_size
		return summed_loss / seen_samples, correct_predictions / seen_samples

	def _run_validation_epoch(self, validation_loader) -> Tuple[float, float]:
		true_labels, predicted_labels, summed_loss = self._predict(validation_loader, track_loss=True)
		return summed_loss / len(true_labels), float(np.mean(true_labels == predicted_labels))

	def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
		true_labels, predicted_labels, _ = self._predict(data_loader, track_loss=False)
		return true_labels, predicted_labels

	def _predict(self, data_loader, track_loss: bool) -> Tuple[np.ndarray, np.ndarray, float]:
		import torch

		self.model.eval()
		collected_true_labels, collected_predictions = [], []
		summed_loss = 0.0
		with torch.no_grad():
			for images, labels in data_loader:
				images = images.to(self.device, non_blocking=True)
				labels = labels.to(self.device, non_blocking=True)
				with self._autocast():
					logits = self.model(images)
				if track_loss:
					summed_loss += self.loss_function(logits.float(), labels).item() * labels.size(0)
				collected_true_labels.append(labels.cpu().numpy())
				collected_predictions.append(logits.argmax(dim=1).cpu().numpy())
		return np.concatenate(collected_true_labels), np.concatenate(collected_predictions), summed_loss
