"""Evaluate the immutable four-sample M1-v2 engineering pilot.

This evaluator is intentionally read-only with respect to generated tensors:
it checks the frozen sample-level M0 protocol, audits redundant velocity
channels, and reuses the existing representation-only angle/FK/foot metrics.
It is not a development-set gate and must not be used for tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe
from scripts.evaluate_m1_pilot import (
    angle_stats,
    build_b0,
    fk_consistency,
    foot_metrics,
    load_norm,
    one_condition,
    tensor_sha256,
)


def _rotation_error_degrees(expected: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    relative = expected @ actual.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
    return torch.acos(cosine.clamp(-1.0, 1.0)) * (180.0 / torch.pi)


def channel_consistency(motion: torch.Tensor) -> dict[str, object]:
    """Compare explicit pose channels against their stored velocity channels."""

    if motion.ndim != 2 or motion.shape[-1] != 276:
        raise ValueError(f"expected [T,276], got {tuple(motion.shape)}")
    frames = motion.shape[0]
    joints = motion[:, MOTION_LAYOUT.joints].reshape(frames, 22, 3)
    joint_velocity = motion[:, MOTION_LAYOUT.joints_velocity].reshape(frames, 22, 3)
    translation = motion[:, MOTION_LAYOUT.root_translation]
    translation_velocity = motion[:, MOTION_LAYOUT.root_translation_velocity]
    root = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation])
    root_velocity = decode_rot6d_safe(motion[:, MOTION_LAYOUT.root_rotation_velocity])
    if frames > 1:
        joint_error = joint_velocity[:-1] - (joints[1:] - joints[:-1])
        translation_error = translation_velocity[:-1] - (translation[1:] - translation[:-1])
        root_error = _rotation_error_degrees(
            root[1:] @ root[:-1].transpose(-1, -2), root_velocity[:-1]
        )
        joint_max = float(torch.linalg.vector_norm(joint_error, dim=-1).max().item())
        translation_max = float(torch.linalg.vector_norm(translation_error, dim=-1).max().item())
        root_max = float(root_error.max().item())
        joint_median = float(torch.linalg.vector_norm(joint_error, dim=-1).median().item())
        translation_median = float(torch.linalg.vector_norm(translation_error, dim=-1).median().item())
        root_median = float(root_error.median().item())
    else:
        joint_max = translation_max = root_max = 0.0
        joint_median = translation_median = root_median = 0.0
    return {
        "joint_position_velocity_max_m": joint_max,
        "joint_position_velocity_median_m": joint_median,
        "root_translation_velocity_max_m": translation_max,
        "root_translation_velocity_median_m": translation_median,
        "root_rotation_velocity_max_degrees": root_max,
        "root_rotation_velocity_median_degrees": root_median,
        "last_row_finite": bool(torch.isfinite(motion).all().item()),
        "checked_internal_rows": max(frames - 1, 0),
        "status": "PASS" if max(joint_max, translation_max, root_max) <= 1e-5 else "FAIL",
    }


def _artifact_manifest(run: Path) -> dict[str, object]:
    m0_dir = run / "m0_artifacts" / "batch_000"
    m1_dir = run / "m1_artifacts" / "batch_000"
    trainer = run / "trainer" / "test_visualization"
    files = [
        m0_dir / "z0_replayed.pt",
        m0_dir / "m0_raw_norm_batch.pt",
        m0_dir / "m0_official_norm_batch.pt",
        m0_dir / "sample_noise_manifest.json",
        m1_dir / "m1_raw_norm_batch.pt",
        m1_dir / "m1_official_norm_batch.pt",
        m1_dir / "m1_config.json",
    ]
    videos = sorted(trainer.rglob("*.mp4")) if trainer.exists() else []
    result: dict[str, object] = {"run": str(run), "files": [], "videos": []}
    for path in files:
        item = {"path": str(path), "exists": path.exists()}
        if path.exists() and path.suffix == ".pt":
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            item.update({"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": tensor_sha256(tensor)})
        result["files"].append(item)
    for path in videos:
        result["videos"].append({"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result["required_files_present"] = all(item["exists"] for item in result["files"])
    result["video_count"] = len(videos)
    return result


def evaluate(root: Path, *, output: Path | None = None, markdown: Path | None = None) -> dict[str, object]:
    mean = torch.from_numpy(__import__("numpy").load(ROOT / "data/meta_info/mean.npy")).float()
    std = torch.from_numpy(__import__("numpy").load(ROOT / "data/meta_info/std.npy")).float()
    frozen = ROOT / "results/phase0/m0_sample_noise_v1_batch4_invariant/artifacts/batch_000"
    frozen_z0 = torch.load(frozen / "z0_replayed.pt", map_location="cpu", weights_only=True)
    frozen_raw = torch.load(frozen / "m0_raw_norm_batch.pt", map_location="cpu", weights_only=True)
    frozen_official = torch.load(frozen / "m0_official_norm_batch.pt", map_location="cpu", weights_only=True)
    model = __import__("smplx").SMPLX(
        model_path=str(ROOT / "data/body_models/smplx"), gender="neutral",
        use_pca=False, num_betas=10, batch_size=100,
    ).eval()
    report: dict[str, object] = {
        "status": "ENGINEERING_PILOT_ONLY",
        "interpretation": "旧四样本工程验证；不替代reviewer1冻结集，不宣称M1入口通过，不用于调参。",
        "protocol": {
            "samples": 4, "target_deltas_degrees": [5.0, -5.0], "seed": 42,
            "sample_noise_protocol": "vimogen-sample-noise-v1", "batch_invariant": True,
            "validation_steps": 50, "shift": 5.0, "denoising_strength": 0.7,
            "dtype": "bfloat16", "attention": "PyTorch SDPA fallback",
            "m1_sigma_window": [0.25, 0.65], "m1_rms_cap": 0.05,
            "heading_mode": "canonical_y", "consistency_mode": "velocity_authoritative_v2",
        },
        "frozen_reference": {
            "z0_sha256": tensor_sha256(frozen_z0),
            "m0_raw_sha256": tensor_sha256(frozen_raw),
            "m0_official_sha256": tensor_sha256(frozen_official),
            "path": str(frozen),
        },
        "runs": {},
    }
    for label, delta in (("plus5_realpilot01", 5.0), ("minus5_realpilot01", -5.0)):
        run = root / label
        m0_dir, m1_dir = run / "m0_artifacts" / "batch_000", run / "m1_artifacts" / "batch_000"
        z0 = torch.load(m0_dir / "z0_replayed.pt", map_location="cpu", weights_only=True)
        m0_raw_norm = torch.load(m0_dir / "m0_raw_norm_batch.pt", map_location="cpu", weights_only=True)
        m0_official_norm = torch.load(m0_dir / "m0_official_norm_batch.pt", map_location="cpu", weights_only=True)
        m1_raw = load_norm(m1_dir / "m1_raw_norm_batch.pt", mean, std)
        m1_official = load_norm(m1_dir / "m1_official_norm_batch.pt", mean, std)
        m0_raw = load_norm(m0_dir / "m0_raw_norm_batch.pt", mean, std)
        baseline = torch.stack([build_b0(sample).motion for sample in m0_raw], dim=0)
        m0_equal = torch.equal(m0_official_norm, frozen_official)
        m0_raw_equal = torch.equal(m0_raw_norm, frozen_raw)
        z0_equal = torch.equal(z0, frozen_z0)
        run_report = {
            "target_delta_degrees": delta,
            "z0_bitwise_equal_frozen": z0_equal,
            "m0_raw_bitwise_equal_frozen": m0_raw_equal,
            "m0_official_bitwise_equal_frozen": m0_equal,
            "z0_sha256": tensor_sha256(z0),
            "m0_raw_sha256": tensor_sha256(m0_raw_norm),
            "m0_official_sha256": tensor_sha256(m0_official_norm),
            "m1_raw_channel_consistency": [channel_consistency(item) for item in m1_raw],
            "m1_official_channel_consistency": [channel_consistency(item) for item in m1_official],
            "m1_raw_metrics": one_condition(baseline, m1_raw, delta, model, heading_mode="canonical_y"),
            "m1_official_metrics": one_condition(baseline, m1_official, delta, model, heading_mode="canonical_y"),
            "artifacts": _artifact_manifest(run),
        }
        report["runs"][label] = run_report
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if markdown is not None:
        lines = [
            "# M1-v2 真实四样本 Pilot 诊断报告", "",
            "状态：**ENGINEERING_PILOT_ONLY**。本报告只验证实现和固定噪声复现，不替代 reviewer1 冻结集，也不宣称 M1 入口通过。", "",
            "## 固定协议", "",
            "50 步，shift=5.0，denoising_strength=0.7，seed=42，sample_v1 样本级 z0，batch_invariant=true，BF16，SDPA 回退；M1-v2 使用 sigma `[0.25, 0.65]`、RMS cap=0.05、canonical_y。", "",
            "## 结果", "",
            "| 条件 | z0/M0 bitwise | M1 原始通道 | M1 official 通道 | MP4 数量 |", "|---|---|---|---|---|",
        ]
        for label, item in report["runs"].items():
            raw_status = all(x["status"] == "PASS" for x in item["m1_raw_channel_consistency"])
            official_status = all(x["status"] == "PASS" for x in item["m1_official_channel_consistency"])
            bitwise = item["z0_bitwise_equal_frozen"] and item["m0_raw_bitwise_equal_frozen"] and item["m0_official_bitwise_equal_frozen"]
            lines.append(f"| {label} | {'PASS' if bitwise else 'FAIL'} | {'PASS' if raw_status else 'FAIL'} | {'PASS' if official_status else 'FAIL'} | {item['artifacts']['video_count']} |")
        lines += ["", "## 数值审计", "", "官方输出的逐样本模型空间代理角结果（四条样本，单位：度）：", "", "| 条件 | 样本 | 实际中位移量 | 目标误差绝对值中位数 |", "|---|---:|---:|---:|"]
        for label, item in report["runs"].items():
            for index, angle in enumerate(item["m1_official_metrics"]["angle"]):
                lines.append(f"| {label} | {index} | {angle['median_shift_degrees']:.4f} | {angle['median_absolute_target_error_degrees']:.4f} |")
            raw = item["m1_raw_channel_consistency"]
            official = item["m1_official_channel_consistency"]
            raw_medians = [float(torch.tensor([x[key] for x in raw]).median().item()) for key in ("joint_position_velocity_median_m", "root_translation_velocity_median_m", "root_rotation_velocity_median_degrees")]
            official_medians = [float(torch.tensor([x[key] for x in official]).median().item()) for key in ("joint_position_velocity_median_m", "root_translation_velocity_median_m", "root_rotation_velocity_median_degrees")]
            lines += ["", f"**{label} 三类通道一致性（四样本中位数）**", "", f"- 原始 M1：关节位置速度 {raw_medians[0]:.6f} m，根平移速度 {raw_medians[1]:.6f} m，根旋转速度 {raw_medians[2]:.6f}°。", f"- official M1：关节位置速度 {official_medians[0]:.6f} m，根平移速度 {official_medians[1]:.6f} m，根旋转速度 {official_medians[2]:.6f}°。", f"- z0/M0 bitwise：z0={item['z0_sha256']}；M0 raw={item['m0_raw_sha256']}；M0 official={item['m0_official_sha256']}。"]
        lines += ["", "## 解释边界", "", "角度指标是模型空间代理角；FK、足部和波形指标是辅助诊断，不是官方 MBench 物理评分。下一步若工程门通过，才进入新留出集；不启动 M1-K/M2。", ""]
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/phase3/m1_v2_pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostics/phase3/m1_v2/real_pilot01/m1_v2_real_pilot01.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "diagnostics/phase3/m1_v2/real_pilot01/M1_V2_REAL_PILOT01.md")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.root, output=args.output, markdown=args.markdown), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
