from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(output_path: Path, payload: Any) -> None:
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w") as output_file:
		json.dump(payload, output_file, indent=4)
