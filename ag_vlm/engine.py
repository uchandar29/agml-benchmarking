"""In-process vLLM engine wrapper with batched, guided-JSON generation."""

from __future__ import annotations

import base64
import io as StdIo
import json
from typing import Dict, List, Optional

from PIL import Image
from vllm import LLM, SamplingParams


# vLLM renamed its structured-output knob across releases. Detect what this
# install provides so the pipeline works on old and new vLLM alike.
# - ~0.6.x:  GuidedDecodingParams + SamplingParams(guided_decoding=...)
# - ~0.8+ :  StructuredOutputsParams + SamplingParams(structured_outputs=...)
GuidedParamsCls: object = None
GuidedKwargName: str = ""
try:
    from vllm.sampling_params import GuidedDecodingParams as _GuidedCls
    GuidedParamsCls = _GuidedCls
    GuidedKwargName = "guided_decoding"
except ImportError:
    try:
        from vllm.sampling_params import StructuredOutputsParams as _StructCls
        GuidedParamsCls = _StructCls
        GuidedKwargName = "structured_outputs"
    except ImportError as importError:
        raise ImportError(
            "Could not find a guided/structured decoding params class in this "
            "vLLM build. Run the probe in the README to identify the right API."
        ) from importError


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

    def ImageToDataUrl(self, image: Image.Image) -> str:
        """Encode a PIL image as a base64 JPEG data URL for the chat API."""
        buffer: StdIo.BytesIO = StdIo.BytesIO()
        image.save(buffer, format="JPEG")
        encoded: str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def BuildRequest(self, promptText: str, image: Image.Image) -> List[Dict]:
        """Build one chat-format request (a message list) with text + image content.

        Using the chat message format lets vLLM apply each model's own chat
        template, so the engine is not tied to any one model family's tokens.
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self.ImageToDataUrl(image)}},
                    {"type": "text", "text": promptText},
                ],
            }
        ]

    def Generate(
        self,
        requests: List[List[Dict]],
        schema: Dict,
        samplingConfig: Dict,
    ) -> List[Optional[Dict]]:
        """Run a batch of chat requests with guided JSON; return parsed dicts (None if bad)."""
        guidedParams: object = GuidedParamsCls(json=schema)
        samplingKwargs: Dict = {
            "temperature": samplingConfig["temperature"],
            "top_p": samplingConfig["top_p"],
            "max_tokens": samplingConfig["max_tokens"],
            GuidedKwargName: guidedParams,
        }
        samplingParams: SamplingParams = SamplingParams(**samplingKwargs)
        rawOutputs: List = self.llm.chat(requests, samplingParams)
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
