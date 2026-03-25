"""High-level inference API for AnimeScore."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
import torch

from animescore._audio import build_padded_batch, load_audio
from animescore.ranknet_model import RankNetMos
from animescore.ssl_encoder import DEFAULT_SSL_NAMES, SSLSpec, build_ssl, freeze_all


DEFAULT_HF_REPO_ID = "tsukumijima/animescore-without-coconut-hubert"
DEFAULT_CHECKPOINT_FILENAME = "animescore_without_coconut_hubert_best.pt"
logger = logging.getLogger(__name__)


class AnimeScorePredictor:
    """
    Load an AnimeScore checkpoint and score audio files.

    By default, the predictor resolves the checkpoint from the public Hugging Face
    model repository maintained for this fork.

    Args:
        checkpoint_path (str | Path | None): Local checkpoint path. When omitted,
            the predictor attempts to download the checkpoint from Hugging Face.
        hf_repo_id (str | None): Hugging Face repository ID. If omitted, the
            predictor falls back to the `ANIMESCORE_MODEL_ID` environment variable,
            then to the default public repository for this fork.
        checkpoint_filename (str): Filename inside the Hugging Face repository.
        ssl_type (str): SSL backbone family.
        ssl_name (str | None): Hugging Face SSL model ID. When omitted, the
            default for `ssl_type` is used.
        device (str | torch.device | None): Inference device. Defaults to CUDA
            when available, otherwise CPU.
        hf_cache_dir (str | Path | None): Optional Hugging Face cache directory.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        hf_repo_id: str | None = None,
        checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
        ssl_type: str = "hubert",
        ssl_name: str | None = None,
        device: str | torch.device | None = None,
        hf_cache_dir: str | Path | None = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self.ssl_type = ssl_type
        self.ssl_name = ssl_name or DEFAULT_SSL_NAMES[ssl_type]
        self.checkpoint_path = self._resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
            hf_repo_id=hf_repo_id,
            checkpoint_filename=checkpoint_filename,
            hf_cache_dir=hf_cache_dir,
        )

        self.model = self._load_model()
        target_sample_rate = getattr(self.model.ssl, "target_sample_rate", None)
        if target_sample_rate is None:
            target_sample_rate = getattr(self.model.ssl, "target_sr", None)
        if target_sample_rate is None:
            raise ValueError(
                "The loaded SSL model does not expose target_sample_rate or target_sr.",
            )
        self.target_sample_rate = int(target_sample_rate)

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        """
        Resolve the inference device.

        Args:
            device (str | torch.device | None): Requested device.

        Returns:
            torch.device: Resolved device object.
        """

        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available() is True:
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _resolve_checkpoint_path(
        checkpoint_path: str | Path | None,
        hf_repo_id: str | None,
        checkpoint_filename: str,
        hf_cache_dir: str | Path | None,
    ) -> Path:
        """
        Resolve the checkpoint path from either a local file or Hugging Face.

        Args:
            checkpoint_path (str | Path | None): Local checkpoint path override.
            hf_repo_id (str | None): Hugging Face repository ID.
            checkpoint_filename (str): Checkpoint filename in the repository.
            hf_cache_dir (str | Path | None): Optional Hugging Face cache directory.

        Returns:
            Path: Resolved local checkpoint path.

        Raises:
            ValueError: Raised when no checkpoint source can be resolved.
        """

        if checkpoint_path is not None:
            return Path(checkpoint_path).expanduser().resolve()

        resolved_repo_id = hf_repo_id or os.environ.get("ANIMESCORE_MODEL_ID") or DEFAULT_HF_REPO_ID

        downloaded_path = hf_hub_download(
            repo_id=resolved_repo_id,
            filename=checkpoint_filename,
            cache_dir=None if hf_cache_dir is None else str(hf_cache_dir),
        )
        return Path(downloaded_path).resolve()

    def _load_model(self) -> RankNetMos:
        """
        Load the model checkpoint and move it to the target device.

        Returns:
            RankNetMos: Initialized model in evaluation mode.
        """

        ssl_model = build_ssl(
            SSLSpec(
                ssl_type=self.ssl_type,
                name_or_path=self.ssl_name,
                target_sr=16000,
                feat_dim=0,
            ),
        )
        freeze_all(ssl_model)
        model = RankNetMos(ssl=ssl_model).to(self.device)

        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        normalized_state_dict = {
            key[len("module."):] if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }
        result = model.load_state_dict(normalized_state_dict, strict=False)
        model_keys = set(model.state_dict().keys())
        matched_keys = len(set(normalized_state_dict.keys()) & model_keys)
        if len(result.missing_keys) > 0:
            logger.warning(
                f"Checkpoint is missing model keys. first_20: {result.missing_keys[:20]}",
            )
        if len(result.unexpected_keys) > 0:
            logger.warning(
                f"Checkpoint has unexpected keys. first_20: {result.unexpected_keys[:20]}",
            )
        if matched_keys < int(0.9 * len(model_keys)):
            raise RuntimeError(
                "Too few checkpoint keys matched the model. "
                "Check ssl_type, ssl_name, and checkpoint_path.",
            )
        model.eval()
        return model

    def score_files(
        self,
        audio_paths: list[str | Path],
        batch_size: int = 8,
    ) -> list[float]:
        """
        Score multiple audio files.

        Args:
            audio_paths (list[str | Path]): Audio file paths to score.
            batch_size (int): Number of utterances scored together.

        Returns:
            list[float]: Predicted AnimeScore values in the same order.
        """

        if len(audio_paths) == 0:
            return []

        score_values: list[float] = []
        for batch_start in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[batch_start:batch_start + batch_size]
            waveforms = [
                load_audio(audio_path, self.target_sample_rate)
                for audio_path in batch_paths
            ]
            waveform_batch = build_padded_batch(waveforms).to(self.device)
            with torch.no_grad():
                batch_scores = self.model.score(waveform_batch).detach().cpu().tolist()
            score_values.extend(float(batch_score) for batch_score in batch_scores)

        return score_values

    def score_file(self, audio_path: str | Path) -> float:
        """
        Score a single audio file.

        Args:
            audio_path (str | Path): Audio file path to score.

        Returns:
            float: Predicted AnimeScore value.
        """

        return self.score_files([audio_path], batch_size=1)[0]

    def compare_files(self, audio_path_a: str | Path, audio_path_b: str | Path) -> dict[str, float]:
        """
        Compare two files with the trained ranker.

        Args:
            audio_path_a (str | Path): First audio file path.
            audio_path_b (str | Path): Second audio file path.

        Returns:
            dict[str, float]: Scores for each file, the margin, and the pairwise
            preference probability that file A is preferred over file B.
        """

        score_a, score_b = self.score_files([audio_path_a, audio_path_b], batch_size=2)
        margin = score_a - score_b
        if margin >= 0.0:
            probability_a_wins = 1.0 / (1.0 + math.exp(-margin))
        else:
            exp_margin = math.exp(margin)
            probability_a_wins = exp_margin / (1.0 + exp_margin)
        return {
            "score_a": score_a,
            "score_b": score_b,
            "margin": margin,
            "probability_a_wins": probability_a_wins,
        }
