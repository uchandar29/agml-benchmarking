"""
Generative zero-shot classifier (Gemma 4, Qwen 3.5) run through vLLM.

Decoding is constrained to the list of class names, so every answer is exactly
one valid class and nothing has to be parsed out of free text.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

INVALID_PREDICTION = -1


def image_to_data_url(image) -> str:
	# PNG keeps the pixels lossless, so the model sees the same image the other models see
	buffer = io.BytesIO()
	image.save(buffer, format="PNG")
	return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def shrink_image(image, max_image_side: int):
	if image.mode != "RGB":
		image = image.convert("RGB")
	if max(image.size) <= max_image_side:
		return image
	resized_image = image.copy()
	resized_image.thumbnail((max_image_side, max_image_side))
	return resized_image


class VisionLanguageZeroShotClassifier:
	def __init__(self, model_settings: Dict[str, Any], prompt_template: str, random_seed: int, language_model=None) -> None:
		self.model_settings = model_settings
		self.prompt_template = prompt_template
		self.batch_size: int = model_settings["batch_size"]
		self.max_image_side: int = model_settings["max_image_side"]
		# Thinking is switched off: we want one direct answer per image, and the chat
		# templates of both Gemma 4 and Qwen 3.5 read this flag
		self.chat_template_kwargs = {"enable_thinking": False}
		if language_model is None:
			from vllm import LLM

			language_model = LLM(
				model=model_settings["model_id"],
				max_model_len=model_settings["max_model_len"],
				gpu_memory_utilization=model_settings["gpu_memory_utilization"],
				limit_mm_per_prompt={"image": 1},
				trust_remote_code=model_settings.get("trust_remote_code", False),
				seed=random_seed,
			)
		self.language_model = language_model
		self.display_class_names: List[str] = []
		self.class_index_by_name: Dict[str, int] = {}
		self.prompt_text = ""
		self.sampling_params = None

	def set_class_names(self, display_class_names: List[str]) -> None:
		self.display_class_names = list(display_class_names)
		self.class_index_by_name = {name: index for index, name in enumerate(self.display_class_names)}
		class_list_text = "\n".join(f"- {name}" for name in self.display_class_names)
		self.prompt_text = self.prompt_template.format(class_list=class_list_text)

		tokenizer = self.language_model.get_tokenizer()
		longest_answer_tokens = max(len(tokenizer.encode(name, add_special_tokens=False)) for name in self.display_class_names)
		self.sampling_params = self._build_sampling_params(max_tokens=longest_answer_tokens + 8)

	def _build_sampling_params(self, max_tokens: int):
		from vllm import SamplingParams
		from vllm.sampling_params import StructuredOutputsParams

		# temperature 0 means greedy decoding, so reruns give the same answers
		return SamplingParams(
			temperature=0.0,
			max_tokens=max_tokens,
			structured_outputs=StructuredOutputsParams(choice=self.display_class_names),
		)

	def _build_conversation(self, image) -> List[Dict[str, Any]]:
		return [{
			"role": "user",
			"content": [
				{"type": "image_url", "image_url": {"url": image_to_data_url(shrink_image(image, self.max_image_side))}},
				{"type": "text", "text": self.prompt_text},
			],
		}]

	def predict(self, images: List[Any]) -> List[int]:
		if self.sampling_params is None:
			raise RuntimeError("set_class_names() must be called before predict()")
		predicted_labels: List[int] = []
		for batch_start in range(0, len(images), self.batch_size):
			image_batch = images[batch_start:batch_start + self.batch_size]
			conversations = [self._build_conversation(image) for image in image_batch]
			outputs = self.language_model.chat(
				conversations,
				self.sampling_params,
				use_tqdm=False,
				chat_template_kwargs=self.chat_template_kwargs,
			)
			for output in outputs:
				answer_text = output.outputs[0].text.strip()
				predicted_labels.append(self.class_index_by_name.get(answer_text, INVALID_PREDICTION))
		return predicted_labels

	def describe(self) -> Dict[str, Any]:
		return {
			"prompt_template": self.prompt_template,
			"max_image_side": self.max_image_side,
			"decoding": "greedy, constrained to class names",
			"chat_template_kwargs": self.chat_template_kwargs,
		}
