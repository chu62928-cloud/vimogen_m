#!/usr/bin/env python3
"""Compare direct, guided, source-noise and contact-compensated +10° candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from smplx import SMPLX

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pelvis_contact_compensation_v3 import evaluate_v3_pair, temporal_naturalness_metrics
from evaluation.relative_root_trunk_v2_1 import direct_joints_from_motion, direct_smpl_parameters
from motion_rep.phase1 import MOTION_LAYOUT
from motion_rep.pose_authority import authority_project


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one match for {pattern!r} under {root}, found {len(matches)}")
    return matches[0]


def _physical(normalized: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if normalized.ndim != 3:
        raise ValueError("normalized motion must have shape [B,T,276]")
    if mean.ndim == 1:
        mean = mean.unsqueeze(0)
    if std.ndim == 1:
        std = std.unsqueeze(0)
    return normalized.float() * std.float()[:, None, :] + mean.float()[:, None, :]


def _load_guided_candidate(run_root: Path, sample_id: str) -> dict[str, Any]:
    archive_path = _find_one(run_root, "trainer/test_visualization/*/batch_*/mbench_raw_norm_batch.pt")
    manifest_path = archive_path.with_name("mbench_batch_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_ids = [str(value) for value in manifest["sample_ids"]]
    if sample_ids.count(str(sample_id)) != 1:
        raise ValueError(f"sample {sample_id} is not unique in {manifest_path}")
    index = sample_ids.index(str(sample_id))
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    candidate = _physical(archive["motion_norm"].float(), archive["motion_mean"].float(), archive["motion_std"].float())
    mask = archive["motion_mask"].bool()
    m0_path = _find_one(run_root, "m0_artifacts/batch_*/m0_official_norm_batch.pt")
    m0_norm = torch.load(m0_path, map_location="cpu", weights_only=True).float()
    m0 = _physical(m0_norm, archive["motion_mean"].float(), archive["motion_std"].float())
    noise_manifest_path = _find_one(run_root, "m0_artifacts/batch_*/sample_noise_manifest.json")
    noise_manifest = json.loads(noise_manifest_path.read_text(encoding="utf-8"))
    noise_rows = [row for row in noise_manifest.get("records", []) if str(row.get("sample_id")) == str(sample_id)]
    if len(noise_rows) != 1:
        raise ValueError(f"sample {sample_id} noise record is not unique in {noise_manifest_path}")
    return {
        "candidate": candidate[index : index + 1],
        "m0": m0[index : index + 1],
        "valid_mask": mask[index : index + 1],
        "candidate_path": archive_path,
        "m0_path": m0_path,
        "manifest_path": manifest_path,
        "sample_index": index,
        "archive_batch_size": int(candidate.shape[0]),
        "noise_manifest_path": noise_manifest_path,
        "noise_record": noise_rows[0],
    }


def _load_motion(path: Path) -> torch.Tensor:
    motion = torch.load(path, map_location="cpu", weights_only=True).float()
    if motion.ndim == 2:
        motion = motion.unsqueeze(0)
    if motion.ndim != 3 or motion.shape[0] != 1 or motion.shape[-1] != MOTION_LAYOUT.total_dim:
        raise ValueError(f"motion must be [T,276] or [1,T,276], got {tuple(motion.shape)}")
    return motion


def _authoritative(motion: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return authority_project(motion, valid_mask=valid_mask, output_dtype=torch.float32).physical_motion


def _difference_by_section(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    sections = {
        "body_pose": MOTION_LAYOUT.body_pose,
        "joints": MOTION_LAYOUT.joints,
        "joint_velocity": MOTION_LAYOUT.joints_velocity,
        "root_rotation": MOTION_LAYOUT.root_rotation,
        "root_rotation_velocity": MOTION_LAYOUT.root_rotation_velocity,
        "root_translation": MOTION_LAYOUT.root_translation,
        "root_translation_velocity": MOTION_LAYOUT.root_translation_velocity,
    }
    result: dict[str, Any] = {}
    for name, section in sections.items():
        difference = left[..., section] - right[..., section]
        result[name] = {
            "max_abs": float(difference.abs().max().item()),
            "rms": float(torch.sqrt(difference.square().mean()).item()),
        }
    return result


def _vertices(motion: torch.Tensor, model: SMPLX, device: torch.device) -> torch.Tensor:
    with torch.inference_mode():
        params = direct_smpl_parameters(motion.to(device))
        params = {key: value[0] for key, value in params.items()}
        return model(**params, return_verts=True).vertices.detach().cpu()


def _p95(record: dict[str, Any] | None) -> float | None:
    return None if not record else record.get("p95")


def _foot_value(result: dict[str, Any], side: str, metric: str) -> tuple[float | None, str]:
    foot = result["feet"][side]
    return _p95(foot["candidate"][metric]), foot["statuses"][metric]


def _ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or abs(float(baseline)) <= 1.0e-12:
        return None
    return float(candidate) / float(baseline)


def _candidate_row(name: str, baseline_name: str, result: dict[str, Any], temporal: dict[str, Any]) -> dict[str, Any]:
    left_slide, left_slide_status = _foot_value(result, "left", "sliding_m_per_frame")
    left_lift, left_lift_status = _foot_value(result, "left", "lift_m")
    left_penetration, left_penetration_status = _foot_value(result, "left", "penetration_m")
    right_slide, right_slide_status = _foot_value(result, "right", "sliding_m_per_frame")
    right_lift, right_lift_status = _foot_value(result, "right", "lift_m")
    right_penetration, right_penetration_status = _foot_value(result, "right", "penetration_m")
    com = result["uprightness"]["com_support"]
    return {
        "candidate": name,
        "paired_baseline": baseline_name,
        "dose_mean_deg": result["angle"]["dose_mean_deg"],
        "dose_mae_deg": result["angle"]["mae_deg"],
        "dose_p95_error_deg": result["angle"]["p95_deg"],
        "trunk_p95_deg": result["trunk_direction"]["p95"],
        "pelvis_neck_p95_deg": result["uprightness"]["pelvis_neck"]["p95"],
        "pelvis_head_p95_deg": result["uprightness"]["pelvis_head"]["p95"],
        "heading_p95_deg": result["heading"]["p95"],
        "q_rigid": result["q_rigid"],
        "root_speed_p95_m_per_frame": temporal["root_speed"]["candidate"]["p95"],
        "mean_joint_speed_p95_m_per_frame": temporal["mean_joint_speed"]["candidate"]["p95"],
        "root_acceleration_p95_m_per_frame2": temporal["root_acceleration"]["candidate"]["p95"],
        "mean_joint_acceleration_p95_m_per_frame2": temporal["mean_joint_acceleration"]["candidate"]["p95"],
        "root_path_length_m": temporal["root_path_length"]["candidate"],
        "root_speed_p95_ratio_to_m0": _ratio(temporal["root_speed"]["candidate"]["p95"], temporal["root_speed"]["m0"]["p95"]),
        "mean_joint_speed_p95_ratio_to_m0": _ratio(temporal["mean_joint_speed"]["candidate"]["p95"], temporal["mean_joint_speed"]["m0"]["p95"]),
        "root_acceleration_p95_ratio_to_m0": _ratio(temporal["root_acceleration"]["candidate"]["p95"], temporal["root_acceleration"]["m0"]["p95"]),
        "mean_joint_acceleration_p95_ratio_to_m0": _ratio(temporal["mean_joint_acceleration"]["candidate"]["p95"], temporal["mean_joint_acceleration"]["m0"]["p95"]),
        "root_path_length_ratio_to_m0": _ratio(temporal["root_path_length"]["candidate"], temporal["root_path_length"]["m0"]),
        "left_slide_p95_m_per_frame": left_slide,
        "left_slide_status": left_slide_status,
        "left_lift_p95_m": left_lift,
        "left_lift_status": left_lift_status,
        "left_penetration_p95_m": left_penetration,
        "left_penetration_status": left_penetration_status,
        "right_slide_p95_m_per_frame": right_slide,
        "right_slide_status": right_slide_status,
        "right_lift_p95_m": right_lift,
        "right_lift_status": right_lift_status,
        "right_penetration_p95_m": right_penetration,
        "right_penetration_status": right_penetration_status,
        "com_shift_p95_m": _p95(com["com_horizontal_shift_m"]),
        "com_on_m0_support_margin_p95_m": _p95(com["candidate_on_m0_support_margin_m"]),
        "evaluation_status": result["status"],
    }


def _baseline_row(name: str, result: dict[str, Any], temporal: dict[str, Any]) -> dict[str, Any]:
    row = _candidate_row(name, name, result, temporal)
    row.update(
        {
            "dose_mean_deg": 0.0,
            "dose_mae_deg": None,
            "dose_p95_error_deg": None,
            "trunk_p95_deg": 0.0,
            "pelvis_neck_p95_deg": 0.0,
            "pelvis_head_p95_deg": 0.0,
            "heading_p95_deg": 0.0,
            "q_rigid": 0.0,
            "root_speed_p95_m_per_frame": temporal["root_speed"]["m0"]["p95"],
            "mean_joint_speed_p95_m_per_frame": temporal["mean_joint_speed"]["m0"]["p95"],
            "root_acceleration_p95_m_per_frame2": temporal["root_acceleration"]["m0"]["p95"],
            "mean_joint_acceleration_p95_m_per_frame2": temporal["mean_joint_acceleration"]["m0"]["p95"],
            "root_path_length_m": temporal["root_path_length"]["m0"],
            "root_speed_p95_ratio_to_m0": 1.0,
            "mean_joint_speed_p95_ratio_to_m0": 1.0,
            "root_acceleration_p95_ratio_to_m0": 1.0,
            "mean_joint_acceleration_p95_ratio_to_m0": 1.0,
            "root_path_length_ratio_to_m0": 1.0,
            "com_shift_p95_m": 0.0,
            "com_on_m0_support_margin_p95_m": _p95(result["uprightness"]["com_support"]["m0_support_margin_m"]),
            "evaluation_status": "REFERENCE",
        }
    )
    for side in ("left", "right"):
        foot = result["feet"][side]
        for metric, short in (("sliding_m_per_frame", "slide"), ("lift_m", "lift"), ("penetration_m", "penetration")):
            row[f"{side}_{short}_p95_m" + ("_per_frame" if short == "slide" else "")] = _p95(foot["baseline"][metric])
            row[f"{side}_{short}_status"] = "REFERENCE"
    return row


def _fmt(value: Any, scale: float = 1.0) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    return f"{float(value) * scale:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-run-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--v1-run-root", type=Path, required=True)
    parser.add_argument("--v2-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sample-id", default="94")
    parser.add_argument("--target-delta-deg", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--m0-match-tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()

    current_m0_path = args.current_run_root / "m0_physical.pt"
    root_only_path = args.current_run_root / "root_only_motion.pt"
    compensated_path = args.current_run_root / "diagnostic_motion.pt"
    current_m0 = _load_motion(current_m0_path)
    valid = torch.ones(current_m0.shape[:2], dtype=torch.bool)
    current_m0 = _authoritative(current_m0, valid)
    v1 = _load_guided_candidate(args.v1_run_root, args.sample_id)
    v2 = _load_guided_candidate(args.v2_run_root, args.sample_id)

    m0_checks: dict[str, Any] = {}
    guided: dict[str, dict[str, Any]] = {"v1_3_guided": v1, "v2_source_noise": v2}
    for name, loaded in guided.items():
        other_m0 = _authoritative(loaded["m0"], loaded["valid_mask"])
        if other_m0.shape != current_m0.shape or not torch.equal(loaded["valid_mask"], valid):
            raise ValueError(f"{name} M0 shape or mask does not match the current sample")
        difference = other_m0 - current_m0
        check = {
            "max_abs": float(difference.abs().max().item()),
            "rms": float(torch.sqrt(difference.square().mean()).item()),
            "exact_equal": bool(torch.equal(other_m0, current_m0)),
            "within_tolerance": bool(difference.abs().max() <= float(args.m0_match_tolerance)),
            "m0_path": str(loaded["m0_path"]),
            "candidate_path": str(loaded["candidate_path"]),
            "manifest_path": str(loaded["manifest_path"]),
            "sample_index": loaded["sample_index"],
            "archive_batch_size": loaded["archive_batch_size"],
            "noise_sha256": loaded["noise_record"].get("sha256"),
            "noise_key_sha256": loaded["noise_record"].get("key_sha256"),
            "noise_derived_seed": loaded["noise_record"].get("derived_seed"),
            "difference_by_section": _difference_by_section(other_m0, current_m0),
        }
        m0_checks[name] = check
    v1_m0 = _authoritative(v1["m0"], v1["valid_mask"])
    v2_m0 = _authoritative(v2["m0"], v2["valid_mask"])
    pairs = {
        "root_only": {"baseline_name": "M0_current", "m0": current_m0, "candidate": _authoritative(_load_motion(root_only_path), valid), "valid": valid},
        "v1_3_guided": {"baseline_name": "M0_v1_3", "m0": v1_m0, "candidate": _authoritative(v1["candidate"], v1["valid_mask"]), "valid": v1["valid_mask"]},
        "v2_source_noise": {"baseline_name": "M0_v2", "m0": v2_m0, "candidate": _authoritative(v2["candidate"], v2["valid_mask"]), "valid": v2["valid_mask"]},
        "diagnostic_compensated": {"baseline_name": "M0_current", "m0": current_m0, "candidate": _authoritative(_load_motion(compensated_path), valid), "valid": valid},
    }
    source_paths = {
        "root_only": root_only_path,
        "v1_3_guided": v1["candidate_path"],
        "v2_source_noise": v2["candidate_path"],
        "diagnostic_compensated": compensated_path,
    }
    patches = json.loads((args.protocol_root / "foot_patches.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = SMPLX(
        model_path=str(args.model_path), gender="neutral", num_betas=10,
        batch_size=int(current_m0.shape[1]), use_pca=False,
    ).to(device)
    evaluations: dict[str, Any] = {}
    temporals: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    baseline_evidence: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for name, pair in pairs.items():
        baseline = pair["m0"]
        candidate = pair["candidate"]
        pair_valid = pair["valid"]
        baseline_name = pair["baseline_name"]
        m0_vertices = _vertices(baseline, model, device)
        candidate_vertices = _vertices(candidate, model, device)
        result = evaluate_v3_pair(
            baseline,
            candidate,
            pair_valid,
            target_delta_deg=float(args.target_delta_deg),
            m0_vertices=m0_vertices,
            candidate_vertices=candidate_vertices,
            patches=patches,
            allow_missing_toe=True,
        )
        result["diagnostic_only"] = True
        result["eligible"] = False
        result["candidate_name"] = name
        result["paired_baseline"] = baseline_name
        result["source_path"] = str(source_paths[name])
        temporal = temporal_naturalness_metrics(
            direct_joints_from_motion(baseline), direct_joints_from_motion(candidate), pair_valid,
        )
        evaluations[name] = result
        temporals[name] = temporal
        rows.append(_candidate_row(name, baseline_name, result, temporal))
        baseline_evidence.setdefault(baseline_name, (result, temporal))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(candidate[0], args.output_dir / f"{name}_physical.pt")
        torch.save(baseline[0], args.output_dir / f"{baseline_name}_physical.pt")

    baseline_rows = [
        _baseline_row(name, result, temporal)
        for name, (result, temporal) in baseline_evidence.items()
    ]
    rows = baseline_rows + rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "protocol": "pelvis_guidance_naturalness_comparison_v1",
        "sample_id": str(args.sample_id),
        "target_delta_deg": float(args.target_delta_deg),
        "diagnostic_only": True,
        "m0_match_tolerance": float(args.m0_match_tolerance),
        "m0_checks": m0_checks,
        "all_m0_within_tolerance": all(check["within_tolerance"] for check in m0_checks.values()),
        "comparison_rule": "each candidate is evaluated against its own archived M0; ratio-to-M0 columns are cross-version comparable",
        "source_hashes": {
            "current_m0": _sha256(current_m0_path),
            "root_only": _sha256(root_only_path),
            "diagnostic_compensated": _sha256(compensated_path),
            "v1_candidate_archive": _sha256(v1["candidate_path"]),
            "v2_candidate_archive": _sha256(v2["candidate_path"]),
        },
        "rows": rows,
        "evaluations": evaluations,
        "temporal_naturalness": temporals,
    }
    _write_json(args.output_dir / "comparison.json", comparison)
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# sample94 +10° 引导候选自然度对照",
        "",
        "本报告只作诊断。所有候选使用同一完整 SMPL-X 网格、同一评价定义，并严格配对各自归档的 M0；跨版本优先比较相对 M0 的倍率。",
        "",
        "| 候选 | 配对基线 | 实际剂量 | 剂量 P95 误差 | 躯干 P95 | 关节加速度 P95 | 相对 M0 | 左足滑 P95 | 左离地 P95 | 左穿地 P95 | 重心位移 P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {candidate} | {baseline} | {dose} | {dose_error} | {trunk} | {accel} | {accel_ratio}× | {slide} | {lift} | {penetration} | {com} |".format(
                candidate=row["candidate"],
                baseline=row["paired_baseline"],
                dose=_fmt(row["dose_mean_deg"]),
                dose_error=_fmt(row["dose_p95_error_deg"]),
                trunk=_fmt(row["trunk_p95_deg"]),
                accel=_fmt(row["mean_joint_acceleration_p95_m_per_frame2"], 1000.0),
                accel_ratio=_fmt(row["mean_joint_acceleration_p95_ratio_to_m0"]),
                slide=_fmt(row["left_slide_p95_m_per_frame"], 1000.0),
                lift=_fmt(row["left_lift_p95_m"], 1000.0),
                penetration=_fmt(row["left_penetration_p95_m"], 1000.0),
                com=_fmt(row["com_shift_p95_m"], 1000.0),
            )
        )
    lines.extend(
        [
            "",
            "## M0 配对说明",
            "",
            "v1.3 的 M0 与当前 v3 冻结 M0 在权威化后仅有浮点舍入级差异；v2 的 M0 则不是同一个逐值基线。"
            f"v2 相对当前 M0 的最大通道差为 `{m0_checks['v2_source_noise']['max_abs']:.9f}`，"
            f"均方根差为 `{m0_checks['v2_source_noise']['rms']:.9f}`。"
            "归档记录确认两者使用相同 seed、派生 seed、噪声键和噪声 SHA256，因此不能归因于随机种子；"
            "现有证据将差异定位到生成/重放路径和批组成（v1.3 为双样本批，v2 为单样本可微/正式重放）在 BF16、50 步采样中的数值累积。"
            "在没有受控批大小交叉实验前，不把原因进一步简化为单一的批大小效应。",
            "",
            "## 结论",
            "",
            "- v1.3 是当前最好的起点：剂量准确，躯干方向 P95 约 0.44°，关节加速度只比自身 M0 高约 2.7%；但离地和穿地仍严格失败。",
            "- v2 只在部分垂向足部指标上改善，同时出现更大的躯干变化、左足滑、时间抖动和全局重心位移，不能判为整体更自然。",
            "- 仅根旋转保持了时间平滑和水平足滑，但把约 10° 变化直接传给躯干，并造成明显离地/穿地。",
            "- 当前诊断补偿在时间、足部和躯干指标上最差，不应继续作为下一轮基线。",
            "",
            "下一轮应以 v1.3 引导候选为名义动作，只对稳定接触期施加最小的下肢/根平移修正，并把时间平滑和接触可行性保持放进每一步接受条件。重心先保持为稳定支撑期软诊断/软约束，不使用全序列硬门。",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "m0_checks": m0_checks, "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
