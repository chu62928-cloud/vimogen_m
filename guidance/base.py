"""Stable top-level contracts for the M1--M7 comparison.

The methods in this study operate at different levels: some alter sampling
states, one optimises source noise, one performs forward shooting, and M7 is
a terminal geometric edit.  This module therefore standardises complete
runs instead of forcing every method into a per-step callback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, runtime_checkable

import torch

from motion_rep.phase1 import MOTION_LAYOUT


PROTOCOL_NAME = "vimogen_pelvis_m1_m7_scale_v1"
REQUIRED_DIAGNOSTICS = (
    "angle_residual_mae_deg",
    "angle_residual_p95_deg",
    "state_update_norm",
    "max_update_norm",
    "model_nfe",
    "full_rollout_count",
    "smplx_forward_count",
    "jvp_count",
    "vjp_count",
    "backward_count",
    "solver_iterations",
    "rejected_steps",
    "nonfinite_count",
    "fallback_used",
    "failure_reason",
    "wall_time_sec",
    "peak_gpu_mem_gb",
)


class ConstraintPack(str, Enum):
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"


def _validate_motion(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 3 or value.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError(f"{name} must have shape [B,T,276]")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _validate_prefix_mask(mask: torch.Tensor, shape: tuple[int, int]) -> None:
    if mask.dtype is not torch.bool or tuple(mask.shape) != shape:
        raise ValueError(f"valid_mask must be bool{shape}")
    if torch.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError("valid_mask must be a contiguous valid prefix")
    if torch.any(mask.sum(dim=-1) < 1):
        raise ValueError("every sequence needs at least one valid frame")


@dataclass(frozen=True)
class SharedEvidence:
    """Frozen evidence shared by every method for one paired run."""

    valid_mask: torch.Tensor
    target_angle_curve_deg: torch.Tensor
    contact_masks: Mapping[str, torch.Tensor] = field(default_factory=dict)
    ground_height_m: float | torch.Tensor | None = None
    contact_evidence_version: str = "not_available"
    ground_version: str = "not_available"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, shape: tuple[int, int]) -> None:
        _validate_prefix_mask(self.valid_mask, shape)
        if tuple(self.target_angle_curve_deg.shape) != shape:
            raise ValueError("target_angle_curve_deg must match [B,T]")
        if not torch.isfinite(self.target_angle_curve_deg).all():
            raise ValueError("target_angle_curve_deg contains non-finite values")
        for name, mask in self.contact_masks.items():
            if mask.dtype is not torch.bool or tuple(mask.shape) != shape:
                raise ValueError(f"contact mask {name!r} must be bool{shape}")


@dataclass(frozen=True)
class GuidanceRequest:
    prompt_id: str
    seed: int
    target_dose_deg: float
    constraint_pack: ConstraintPack | str
    base_noise: torch.Tensor
    baseline_motion: torch.Tensor
    shared_evidence: SharedEvidence
    run_id: str = ""
    baseline_motion_id: str = ""

    def __post_init__(self) -> None:
        _validate_motion("baseline_motion", self.baseline_motion)
        if not isinstance(self.base_noise, torch.Tensor):
            raise TypeError("base_noise must be a torch.Tensor")
        if tuple(self.base_noise.shape) != tuple(self.baseline_motion.shape):
            raise ValueError("base_noise must match baseline_motion [B,T,276]")
        if not torch.is_floating_point(self.base_noise):
            raise TypeError("base_noise must be floating point")
        if not torch.isfinite(self.base_noise).all():
            raise ValueError("base_noise contains non-finite values")
        if not math.isfinite(float(self.target_dose_deg)):
            raise ValueError("target_dose_deg must be finite")
        pack = ConstraintPack(self.constraint_pack)
        object.__setattr__(self, "constraint_pack", pack)
        self.shared_evidence.validate(tuple(self.baseline_motion.shape[:2]))


def slice_request(request: GuidanceRequest, index: int) -> GuidanceRequest:
    """Create an independent single-sample request for batch-invariant runs."""

    batch = request.baseline_motion.shape[0]
    if not 0 <= index < batch:
        raise IndexError(f"sample index {index} is outside batch size {batch}")

    def one(value: torch.Tensor) -> torch.Tensor:
        return value[index : index + 1]

    ground = request.shared_evidence.ground_height_m
    if isinstance(ground, torch.Tensor) and ground.ndim and ground.shape[0] == batch:
        ground = one(ground)
    evidence = SharedEvidence(
        valid_mask=one(request.shared_evidence.valid_mask),
        target_angle_curve_deg=one(request.shared_evidence.target_angle_curve_deg),
        contact_masks={
            name: one(mask) for name, mask in request.shared_evidence.contact_masks.items()
        },
        ground_height_m=ground,
        contact_evidence_version=request.shared_evidence.contact_evidence_version,
        ground_version=request.shared_evidence.ground_version,
        metadata=request.shared_evidence.metadata,
    )
    return GuidanceRequest(
        prompt_id=request.prompt_id,
        seed=request.seed,
        target_dose_deg=request.target_dose_deg,
        constraint_pack=request.constraint_pack,
        base_noise=one(request.base_noise),
        baseline_motion=one(request.baseline_motion),
        shared_evidence=evidence,
        run_id=request.run_id,
        baseline_motion_id=request.baseline_motion_id,
    )


def slice_batch_stat(value: torch.Tensor, index: int, batch: int) -> torch.Tensor:
    """Slice a batch-specific normalisation statistic while preserving globals."""

    if value.ndim >= 2 and value.shape[0] == batch:
        return value[index : index + 1]
    return value


@dataclass
class GuidanceDiagnostics:
    angle_residual_mae_deg: float | None = None
    angle_residual_p95_deg: float | None = None
    state_update_norm: float = 0.0
    max_update_norm: float = 0.0
    model_nfe: int = 0
    full_rollout_count: int = 0
    smplx_forward_count: int = 0
    jvp_count: int = 0
    vjp_count: int = 0
    backward_count: int = 0
    solver_iterations: int = 0
    rejected_steps: int = 0
    nonfinite_count: int = 0
    fallback_used: bool = False
    failure_reason: str = ""
    wall_time_sec: float = 0.0
    peak_gpu_mem_gb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        extra = result.pop("extra")
        result.update(extra)
        missing = [name for name in REQUIRED_DIAGNOSTICS if name not in result]
        if missing:
            raise RuntimeError(f"missing required diagnostics: {missing}")
        return result


@dataclass
class GuidedSample:
    motion: torch.Tensor
    diagnostics: GuidanceDiagnostics | Mapping[str, Any]
    status: str

    def __post_init__(self) -> None:
        _validate_motion("motion", self.motion)
        if not self.status:
            raise ValueError("status must be non-empty")

    def diagnostics_dict(self) -> dict[str, Any]:
        result = (
            self.diagnostics.to_dict()
            if isinstance(self.diagnostics, GuidanceDiagnostics)
            else dict(self.diagnostics)
        )
        missing = [name for name in REQUIRED_DIAGNOSTICS if name not in result]
        if missing:
            raise ValueError(f"guided sample diagnostics missing: {missing}")
        return result


@runtime_checkable
class GuidanceMethod(Protocol):
    name: str

    def run(
        self,
        vimogen: Any,
        request: GuidanceRequest,
        cfg: Mapping[str, Any] | None = None,
    ) -> GuidedSample:
        ...


def tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    header = json.dumps(
        {"shape": list(cpu.shape), "dtype": str(cpu.dtype)},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(header + cpu.numpy().tobytes()).hexdigest()


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("strict JSON does not allow NaN or Infinity")
    return value


def write_run_record(path: Path, record: Mapping[str, Any]) -> None:
    """Atomically write a strict, single-sample audit record."""

    required = {
        "run_id",
        "code_commit",
        "vimogen_checkpoint_hash",
        "evaluator_version",
        "method_name",
        "method_config_hash",
        "prompt_id",
        "seed",
        "target_dose_deg",
        "constraint_pack",
        "baseline_motion_id",
        "contact_evidence_version",
        "ground_version",
        "status",
        "failure_reason",
        "all_metrics",
        "all_diagnostics",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"run record is missing required fields: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_strict_json_value(record), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class RunTimer:
    def __enter__(self) -> "RunTimer":
        self.started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *_: object) -> None:
        self.wall_time_sec = time.perf_counter() - self.started
        self.peak_gpu_mem_gb = (
            torch.cuda.max_memory_reserved() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
