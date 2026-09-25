"""Higgs Audio v3 STT backend adapter (bosonai/higgs-audio-v3-stt).

Combines Whisper-Large-v3 audio encoder with Qwen3 LLM decoder for high-accuracy
single-pass speech-to-text.
"""

import importlib.util
import logging
from pathlib import Path
import re
import sys
from typing import Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class HiggsSTTModel:
    """Wrapper for bosonai/higgs-audio-v3-stt model using official repo transcribe helper."""

    def __init__(
        self,
        model_id: str = "bosonai/higgs-audio-v3-stt",
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.model: Any = None
        self.tokenizer: Any = None
        self._transcribe_fn: Any = None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            from transformers.utils import cached_file
        except ImportError as exc:
            raise ImportError(
                "Higgs STT requires PyTorch and Transformers. Please run:\n"
                "  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n"
                "  uv pip install transformers>=4.51.0"
            ) from exc

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        selected_dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)

        actual_device = self.device
        if "cuda" in str(actual_device):
            if not torch.cuda.is_available():
                logger.warning("[INFO] No NVIDIA CUDA device available. Falling back to CPU.")
                actual_device = "cpu"
                selected_dtype = torch.float32

        logger.info("[INFO] Initializing Higgs Audio v3 STT model '%s' on %s (downloading weights if first run)...", self.model_id, actual_device)
        try:
            self.model = AutoModel.from_pretrained(
                self.model_id,
                torch_dtype=selected_dtype,
                trust_remote_code=True,
                attn_implementation="eager",
                device_map=actual_device,
            )
        except Exception as err:
            if "cuda" in str(actual_device):
                logger.warning("[WARN] CUDA model load failed (%s). Falling back to CPU.", err)
                actual_device = "cpu"
                selected_dtype = torch.float32
                self.model = AutoModel.from_pretrained(
                    self.model_id,
                    torch_dtype=selected_dtype,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    device_map="cpu",
                )
            else:
                raise

        self.device = actual_device
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # Load transcribe helper from cached repo files
        try:
            transcribe_path = cached_file(self.model_id, "transcribe.py")
            if transcribe_path:
                model_dir = str(Path(transcribe_path).parent.resolve())
                if model_dir not in sys.path:
                    sys.path.insert(0, model_dir)
                spec = importlib.util.spec_from_file_location("higgs_transcribe", transcribe_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    self._transcribe_fn = getattr(mod, "transcribe", None)
        except Exception as e:
            logger.warning("Could not load bundled transcribe.py from model repo: %s", e)

    def transcribe(
        self,
        audio_np: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Transcribe 16kHz mono audio chunk into finalized text in a single pass."""
        if len(audio_np) == 0:
            return []

        try:
            if self._transcribe_fn is not None:
                text = self._transcribe_fn(self.model, self.tokenizer, audio_np, sample_rate=sample_rate)
                # Clean any residue thinking or special tags
                clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                clean = re.sub(r"<\|.*?\|>", "", clean).strip()
                return [clean] if clean else []
        except Exception as exc:
            logger.error("Higgs inference error: %s", exc)

        return []


def get_higgs_model(
    model_id: str = "bosonai/higgs-audio-v3-stt",
    device: str = "cuda",
    torch_dtype: str = "float16",
) -> HiggsSTTModel:
    """Factory helper to instantiate HiggsSTTModel."""
    return HiggsSTTModel(model_id=model_id, device=device, torch_dtype=torch_dtype)
