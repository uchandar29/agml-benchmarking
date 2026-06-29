"""Incremental JSONL writing, resume support and run metadata."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List, Set


def EnsureDir(path: str) -> None:
    """Create a directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def LoadDoneIds(recordsPath: str) -> Set[str]:
    """Read an existing JSONL file and collect image_ids already processed (for resume)."""
    doneIds: Set[str] = set()
    if not os.path.exists(recordsPath):
        return doneIds
    with open(recordsPath, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record: Dict = json.loads(line)
            except json.JSONDecodeError:
                continue
            imageId: str = record.get("image_id", "")
            if imageId:
                doneIds.add(imageId)
    return doneIds


def AppendRecords(recordsPath: str, records: List[Dict]) -> None:
    """Append a batch of records to the JSONL file, flushing immediately."""
    with open(recordsPath, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.flush()


def WriteJson(path: str, data: Dict) -> None:
    """Write a dict to a JSON file (pretty-printed)."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def WriteText(path: str, text: str) -> None:
    """Write plain text to a file."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def ReadAllRecords(recordsPath: str) -> List[Dict]:
    """Load every record from a JSONL file (used to recompute analytics after a run)."""
    records: List[Dict] = []
    if not os.path.exists(recordsPath):
        return records
    with open(recordsPath, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def GitCommitHash() -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    try:
        result: str = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return result
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def WriteRunMeta(path: str, runConfig: Dict) -> None:
    """Save the run configuration plus git commit for reproducibility."""
    runConfig = dict(runConfig)
    runConfig["git_commit"] = GitCommitHash()
    WriteJson(path, runConfig)
