import json
from pathlib import Path

import pytest
import yaml

from benchmark import run as run_module
from benchmark.config import DetectionConfig, PipelineConfig
from benchmark.output.writer import ReportWriter
from benchmark.pipelines.image_classification import AgMLBenchmarkPipeline
from benchmark.pipelines.object_detection import DetectionBenchmarkPipeline
from core.data import DetectionDatasetAdapter
from tests.detection.synthetic import TinyEncoder, make_detection_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_merged_yaml_resolves():
    defaults, entries = run_module.load_yaml(REPO_ROOT / "benchmark" / "slurm" / "datasets.yaml")
    resolved = [run_module.resolve_entry(defaults, entry) for entry in entries]
    assert all(cfg["task"] in run_module.TASK_PIPELINES for cfg in resolved)
    names = [cfg["name"] for cfg in resolved]
    assert len(names) == len(set(names))
    # An entry with no task falls back to image classification and its defaults
    classification = run_module.resolve_entry(defaults, {"name": "x"})
    assert classification["task"] == "image_classification"
    assert classification["label_col"] == "label" and classification["compound_label_cols"] is None
    detection = [cfg for cfg in resolved if cfg["task"] == "object_detection"]
    assert detection and all(cfg["objects_col"] == "objects" and cfg["phases"] == [1, 2, 3] for cfg in detection)


def test_alias_and_entry_override():
    defaults = {**run_module.BUILTIN_DEFAULTS}
    cfg = run_module.resolve_entry(defaults, {"name": "x", "task": "classification", "label_col": "kind"})
    assert cfg["task"] == "image_classification" and cfg["label_col"] == "kind"


def test_config_sections(tmp_path):
    cfg = PipelineConfig.from_json(str(REPO_ROOT / "benchmark" / "slurm" / "config.json"))
    assert cfg.split_seed == 42 and cfg.cv_folds == 5
    assert cfg.image_classification.backbone == "resnet18" and cfg.image_classification.cartography_epochs == 30
    assert cfg.object_detection.n_epochs == 12 and cfg.object_detection.crop_max_total == 20000
    cfg.to_json(str(tmp_path / "used.json"))
    written = json.loads((tmp_path / "used.json").read_text())
    assert set(written) == {"common", "image_classification", "object_detection"}
    assert PipelineConfig.from_json(str(tmp_path / "used.json")).to_dict() == cfg.to_dict()


def test_image_classification_writer_path(tmp_path):
    pipeline = AgMLBenchmarkPipeline("Project-AgML/rice_leaf_disease_classification", cfg=PipelineConfig(output_dir=str(tmp_path)))
    path = Path(pipeline.report_path())
    assert path.parent.parent.name == "rice_leaf_disease_classification"
    assert path.parent.parent.parent == tmp_path / "image_classification"
    assert json.loads(path.read_text())["task"] == "image_classification"


def test_writer_keeps_inat_prefix(tmp_path):
    writer = ReportWriter("iNatAg/acacia_auriculiformis", str(tmp_path), "image_classification")
    assert Path(writer.run_dir).parent == tmp_path / "image_classification" / "iNatAg" / "acacia_auriculiformis"


def write_config_and_yaml(tmp_path, entries):
    config = PipelineConfig(
        output_dir=str(tmp_path / "results"), embed_batch_size=8, cv_folds=3,
        object_detection=DetectionConfig(n_epochs=1, batch_size=4, image_min_size=96, pretrained=False, crop_max_per_class=100),
    )
    config.to_json(str(tmp_path / "config.json"))
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(yaml.safe_dump({"datasets": entries}))
    return yaml_path


class StubEncoderPipeline(DetectionBenchmarkPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, encoder=TinyEncoder(), **kwargs)


def test_dispatch_runs_object_detection(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    dataset = make_detection_dataset(num_images=30)
    monkeypatch.setattr(DetectionDatasetAdapter, "_load_hf", lambda self: dataset)
    monkeypatch.setitem(
        run_module.TASK_PIPELINES, "object_detection",
        run_module.TaskPipeline(StubEncoderPipeline, run_module.TASK_PIPELINES["object_detection"].arguments),
    )
    yaml_path = write_config_and_yaml(tmp_path, [{"name": "fake/synthetic_detection", "task": "object_detection"}])
    monkeypatch.setenv("AGML_CONFIG", str(tmp_path / "config.json"))

    exit_code = run_module.main(["--yaml", str(yaml_path), "--dataset", "fake/synthetic_detection"])

    assert exit_code == 0
    reports = list((tmp_path / "results" / "object_detection" / "synthetic_detection").glob("*/report.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["scores"]["overall"] is not None


def test_failed_run_removes_dataset_dir_but_not_task_dir(tmp_path, monkeypatch):
    def broken_load(self):
        raise RuntimeError("download failed")

    monkeypatch.setattr(DetectionDatasetAdapter, "_load_hf", broken_load)
    yaml_path = write_config_and_yaml(tmp_path, [{"name": "fake/broken", "task": "object_detection"}])
    monkeypatch.setenv("AGML_CONFIG", str(tmp_path / "config.json"))

    assert run_module.main(["--yaml", str(yaml_path)]) == 1
    assert (tmp_path / "results" / "object_detection").is_dir()
    assert not (tmp_path / "results" / "object_detection" / "broken").exists()


def test_prefetch_only_calls_prefetch_not_run(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(AgMLBenchmarkPipeline, "prefetch", lambda self: calls.append(("ic", self.dataset_name)))
    monkeypatch.setattr(DetectionBenchmarkPipeline, "prefetch", lambda self: calls.append(("od", self.dataset_name)))

    def forbidden(self, *args, **kwargs):
        raise AssertionError("run() must not be called with --prefetch-only")

    monkeypatch.setattr(AgMLBenchmarkPipeline, "run", forbidden)
    monkeypatch.setattr(DetectionBenchmarkPipeline, "run", forbidden)
    yaml_path = write_config_and_yaml(tmp_path, [
        {"name": "fake/leaves"},
        {"name": "fake/pods", "task": "object_detection"},
    ])
    monkeypatch.setenv("AGML_CONFIG", str(tmp_path / "config.json"))

    assert run_module.main(["--yaml", str(yaml_path), "--prefetch-only"]) == 0
    assert calls == [("ic", "fake/leaves"), ("od", "fake/pods")]

    calls.clear()
    assert run_module.main(["--yaml", str(yaml_path), "--prefetch-only", "--task", "object_detection"]) == 0
    assert calls == [("od", "fake/pods")]
    assert not (tmp_path / "results").exists()


def test_unknown_dataset_is_an_error(tmp_path, monkeypatch):
    yaml_path = write_config_and_yaml(tmp_path, [{"name": "fake/pods", "task": "object_detection"}])
    monkeypatch.setenv("AGML_CONFIG", str(tmp_path / "config.json"))
    with pytest.raises(SystemExit):
        run_module.main(["--yaml", str(yaml_path), "--dataset", "fake/missing"])
