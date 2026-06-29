"""In-process vLLM engine wrapper with batched, guided-JSON generation."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


class VlmEngine:
    """Wraps a single in-process vLLM LLM and runs batched multimodal requests."""

    def __init__(self, modelName: str, engineConfig: Dict) -> None:
        """Load the model once; both passes reuse this engine."""
        self.modelName: str = modelName
        self.engineConfig: Dict = engineConfig
        self.llm: LLM = LLM(
            model=modelName,
            trust_remote_code=True,
            tensor_parallel_size=engineConfig["tensor_parallel_size"],
            max_model_len=engineConfig["max_model_len"],
            dtype=engineConfig["dtype"],
            gpu_memory_utilization=engineConfig["gpu_memory_utilization"],
            limit_mm_per_prompt={"image": engineConfig["max_images_per_prompt"]},
        )

    def BuildRequest(self, promptText: str, image: Image.Image) -> Dict:
        """Build one vLLM multimodal request from a text prompt and an image."""
        chatPrompt: str = (
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{promptText}<|im_end|>\n<|im_start|>assistant\n"
        )
        return {"prompt": chatPrompt, "multi_modal_data": {"image": image}}

    def Generate(
        self,
        requests: List[Dict],
        schema: Dict,
        samplingConfig: Dict,
    ) -> List[Optional[Dict]]:
        """Run a batch of requests with guided JSON; return parsed dicts (None if bad)."""
        guidedParams: GuidedDecodingParams = GuidedDecodingParams(json=schema)
        samplingParams: SamplingParams = SamplingParams(
            temperature=samplingConfig["temperature"],
            top_p=samplingConfig["top_p"],
            max_tokens=samplingConfig["max_tokens"],
            guided_decoding=guidedParams,
        )
        rawOutputs: List = self.llm.generate(requests, samplingParams)
        parsedResults: List[Optional[Dict]] = []
        for output in rawOutputs:
            generatedText: str = output.outputs[0].text
            parsedResults.append(self.SafeParse(generatedText))
        return parsedResults

    def SafeParse(self, text: str) -> Optional[Dict]:
        """Parse model text as JSON; return None on failure (counted as malformed)."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
