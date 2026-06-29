"""CLI entry point: orchestrates the two-pass pipeline over one or more datasets."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import yaml
from PIL import Image

from ag_vlm import analytics as Analytics
from ag_vlm import io as Io
from ag_vlm import prompts as Prompts
from ag_vlm.data import ImageSample, LoadDataset
from ag_vlm.engine import VlmEngine


def ParseArgs() -> argparse.Namespace:
    """Parse the minimal CLI (3 required, 3 optional) described in the spec."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Ag-VLM reasoning pipeline")
    parser.add_argument("--datasets", required=True, help="Comma-separated AgML dataset names")
    parser.add_argument("--model", required=True, help="HF VLM model name for vLLM")
    parser.add_argument("--output-dir", required=True, help="Where JSONL + analytics go")
    parser.add_argument("--limit", type=int, default=None, help="Cap samples per dataset")
    parser.add_argument("--no-predict", action="store_true", help="Skip Stage A (no analytics)")
    parser.add_argument("--contrastive", action="store_true", help="Enable contrastive Stage B")
    return parser.parse_args()


def LoadConfig() -> Dict:
    """Load config.yaml that sits next to this module."""
    configPath: str = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(configPath, "r", encoding="utf-8") as handle:
        config: Dict = yaml.safe_load(handle)
    return config


def RunPredictPass(
    engine: VlmEngine,
    samples: List[ImageSample],
    classList: List[str],
    config: Dict,
) -> tuple[List[Optional[Dict]], int]:
    """Run Stage A over a batch; return parsed predictions and a malformed count."""
    promptText: str = Prompts.BuildPredictPrompt(classList)
    schema: Dict = Prompts.BuildPredictSchema(classList)
    requests: List[Dict] = [engine.BuildRequest(promptText, sample.image) for sample in samples]
    parsed: List[Optional[Dict]] = engine.Generate(requests, schema, config["sampling"])
    malformed: int = sum(1 for item in parsed if item is None)
    return parsed, malformed


def RunRationalizePass(
    engine: VlmEngine,
    samples: List[ImageSample],
    config: Dict,
    contrastive: bool,
    confusableLookup: Dict[str, Optional[str]],
) -> List[Optional[Dict]]:
    """Run Stage B over a batch; each request uses that sample's true label."""
    schema: Dict = Prompts.BuildRationalizeSchema(config["evidence_fields"])
    requests: List[Dict] = []
    for sample in samples:
        confusable: Optional[str] = confusableLookup.get(sample.trueLabel) if contrastive else None
        promptText: str = Prompts.BuildRationalizePrompt(sample.trueLabel, contrastive, confusable)
        requests.append(engine.BuildRequest(promptText, sample.image))
    return engine.Generate(requests, schema, config["sampling"])


def BuildRecord(
    sample: ImageSample,
    datasetName: str,
    prediction: Optional[Dict],
    rationale: Optional[Dict],
    runPredict: bool,
) -> Dict:
    """Assemble one output record combining both passes for a single image."""
    predictedClass: Optional[str] = prediction.get("predicted_class") if prediction else None
    confidence: Optional[float] = prediction.get("confidence") if prediction else None
    predictReasoning: Optional[List[str]] = prediction.get("visual_evidence") if prediction else None
    correct: Optional[bool] = None
    if runPredict and predictedClass is not None:
        correct = predictedClass == sample.trueLabel
    return {
        "image_id": sample.imageId,
        "dataset": datasetName,
        "true_label": sample.trueLabel,
        "predicted_class": predictedClass,
        "confidence": confidence,
        "correct": correct,
        "predict_reasoning": predictReasoning,
        "rationale": rationale,
    }


def ProcessDataset(engine: VlmEngine, datasetName: str, args: argparse.Namespace, config: Dict) -> None:
    """Run the full two-pass pipeline for one dataset and write all outputs."""
    datasetDir: str = os.path.join(args.output_dir, datasetName)
    Io.EnsureDir(datasetDir)
    recordsPath: str = os.path.join(datasetDir, "records.jsonl")

    dataDir: Optional[str] = os.environ.get("AGML_DATA_DIR")
    maxPixels: int = config["engine"]["image_max_pixels"]
    classList: List[str]
    classList, sampleIterator = LoadDataset(datasetName, dataDir, maxPixels, args.limit)

    doneIds: set = Io.LoadDoneIds(recordsPath)
    runPredict: bool = not args.no_predict
    batchSize: int = config["batching"]["batch_size"]

    # First sweep gives us a confusion-based confusable lookup for contrastive Stage B.
    confusableLookup: Dict[str, Optional[str]] = {}

    pendingBatch: List[ImageSample] = []
    totalMalformed: int = 0

    def FlushBatch(batch: List[ImageSample]) -> int:
        """Run both passes on a batch, write the records, return malformed count."""
        if not batch:
            return 0
        predictions: List[Optional[Dict]] = [None] * len(batch)
        malformed: int = 0
        if runPredict:
            predictions, malformed = RunPredictPass(engine, batch, classList, config)
        rationales: List[Optional[Dict]] = RunRationalizePass(
            engine, batch, config, args.contrastive, confusableLookup
        )
        records: List[Dict] = [
            BuildRecord(batch[i], datasetName, predictions[i], rationales[i], runPredict)
            for i in range(len(batch))
        ]
        Io.AppendRecords(recordsPath, records)
        return malformed

    for sample in sampleIterator:
        if sample.imageId in doneIds:
            continue
        pendingBatch.append(sample)
        if len(pendingBatch) >= batchSize:
            totalMalformed += FlushBatch(pendingBatch)
            pendingBatch = []
    totalMalformed += FlushBatch(pendingBatch)

    # Analytics are computed from the predict-pass only.
    if runPredict:
        allRecords: List[Dict] = Io.ReadAllRecords(recordsPath)
        analytics: Dict = Analytics.BuildAnalytics(allRecords, classList, totalMalformed)
        Io.WriteJson(os.path.join(datasetDir, "analytics.json"), analytics)
        markdown: str = Analytics.RenderAnalyticsMarkdown(analytics, datasetName, classList)
        Io.WriteText(os.path.join(datasetDir, "analytics.md"), markdown)


def Main() -> None:
    """Top-level driver: load config, init engine once, loop over datasets."""
    args: argparse.Namespace = ParseArgs()
    config: Dict = LoadConfig()
    Io.EnsureDir(args.output_dir)

    runMeta: Dict = {
        "datasets": args.datasets,
        "model": args.model,
        "limit": args.limit,
        "no_predict": args.no_predict,
        "contrastive": args.contrastive,
    }
    Io.WriteRunMeta(os.path.join(args.output_dir, "run_meta.json"), runMeta)

    engine: VlmEngine = VlmEngine(args.model, config["engine"])
    datasetNames: List[str] = [name.strip() for name in args.datasets.split(",") if name.strip()]
    for datasetName in datasetNames:
        ProcessDataset(engine, datasetName, args, config)


if __name__ == "__main__":
    Main()
