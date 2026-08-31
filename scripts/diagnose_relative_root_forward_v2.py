#!/usr/bin/env python3
"""One entry point for the isolated v2 evaluation and reachability audits.

The ``audit`` command is the only command that can create a v2 result table.
``kinematic`` and ``subspace`` write diagnostic-oracle artifacts with explicit
``counts_as_v2_success=false`` metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.relative_root_forward_v1 import compute_relative_root_forward_metrics
from evaluation.relative_root_forward_v2 import (
    FAIL,
    NOT_EVALUABLE,
    PASS,
    build_v2_gates,
    causal_audit,
    combine_gate_statuses,
    tensor_sha256,
    write_json_strict,
)
from motion_rep.phase1 import MOTION_LAYOUT, decode_rot6d_safe, encode_rot6d
from motion_rep.pose_authority import _root_forward, authority_project


PROTOCOL = "vimogen_relative_root_forward_v2_minimal_source_noise"
DEFAULT_RUN = ROOT / "results/phase7/relative_root_forward_v2/minimal_source_noise/seed_000/delta_+10deg/attempt_09"
DEFAULT_OUTPUT = ROOT / "results/phase7/relative_root_forward_v2/diagnostics/relative_root_forward_v2_unified"


def _find_one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern} below {root}, found {len(matches)}")
    return matches[0]


def _load_run(run_root: Path) -> dict[str, Any]:
    m0_raw_path = _find_one(run_root, "m0_artifacts/batch_*/m0_raw_norm_batch.pt")
    m0_official_path = _find_one(run_root, "m0_artifacts/batch_*/m0_official_norm_batch.pt")
    archive_path = _find_one(run_root, "trainer/test_visualization/*/batch_*/mbench_raw_norm_batch.pt")
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    m0_raw = torch.load(m0_raw_path, map_location="cpu", weights_only=True).float()
    m0_official = torch.load(m0_official_path, map_location="cpu", weights_only=True).float()
    candidate = archive["motion_norm"].float()
    mean = archive["motion_mean"].float()
    std = archive["motion_std"].float()
    if mean.ndim == 1:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    valid_mask = archive["motion_mask"].bool()
    if m0_raw.shape != candidate.shape or m0_official.shape != candidate.shape:
        raise ValueError("M0 and candidate archives have different shapes")
    physical = lambda value: value * std[:, None, :] + mean[:, None, :]
    return {
        "m0_raw_norm": m0_raw,
        "m0_official_norm": m0_official,
        "candidate_norm": candidate,
        "m0_raw": physical(m0_raw),
        "m0_official": physical(m0_official),
        "candidate": physical(candidate),
        "mean": mean,
        "std": std,
        "valid_mask": valid_mask,
        "archive_path": archive_path,
        "m0_raw_path": m0_raw_path,
        "m0_official_path": m0_official_path,
        "sample_ids": archive.get("sample_ids", []),
    }


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    rows = metrics.get("per_sample", [])
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", *fields])
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({"sample_index": index, **row})


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def run_audit(run_root: Path, output_root: Path, target_delta_deg: float) -> dict[str, Any]:
    data = _load_run(run_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "motions").mkdir(exist_ok=True)
    (output_root / "noise").mkdir(exist_ok=True)
    baseline_projection = authority_project(
        data["m0_official"], valid_mask=data["valid_mask"], output_dtype=torch.float32
    )
    candidate_projection = authority_project(
        data["candidate"], valid_mask=data["valid_mask"], output_dtype=torch.float32
    )
    consistent_m0 = baseline_projection.physical_motion
    final_output = candidate_projection.physical_motion
    metrics = compute_relative_root_forward_metrics(
        consistent_m0,
        final_output,
        data["valid_mask"],
        target_delta_deg,
        protocol_name=PROTOCOL,
    )
    audit = causal_audit(
        official_raw=data["m0_raw"],
        official_pre_cast=data["m0_official"],
        consistent_m0=consistent_m0,
        candidate_uncanonical=data["candidate"],
        final_output=final_output,
        valid_mask=data["valid_mask"],
    )
    gates = build_v2_gates(metrics)
    naturalness = _load_optional_json(run_root / "source_noise_naturalness_evaluation.json")
    if naturalness is None:
        naturalness_gate = {"name": "naturalness", "status": NOT_EVALUABLE, "reason": "external naturalness audit is absent"}
    else:
        naturalness_gate = {
            "name": "naturalness",
            "status": PASS if naturalness.get("naturalness_gate", {}).get("passed") else FAIL,
            "reason": "external non-optimizing audit",
            "source": str(run_root / "source_noise_naturalness_evaluation.json"),
        }
    gates.append(naturalness_gate)
    gate_dicts = [gate.as_dict() if hasattr(gate, "as_dict") else gate for gate in gates]
    total_status = combine_gate_statuses(gate_dicts)

    tensors = {
        "official_raw_m0": data["m0_raw"],
        "official_pre_cast_m0": data["m0_official"],
        "consistent_m0": consistent_m0,
        "candidate_uncanonical": data["candidate"],
        "final_output": final_output,
    }
    for name, value in tensors.items():
        torch.save(value.detach().cpu(), output_root / "motions" / f"{name}.pt")
    delta_path = run_root / "source_noise_artifacts/batch_000/text/source_delta.pt"
    if delta_path.is_file():
        delta = torch.load(delta_path, map_location="cpu", weights_only=True)
        torch.save(delta, output_root / "noise" / "source_delta.pt")

    metrics["unified_gate_status"] = total_status
    metrics["input_run"] = str(run_root)
    metrics["sample_ids"] = [str(value) for value in data["sample_ids"]]
    baseline_root = decode_rot6d_safe(consistent_m0[..., MOTION_LAYOUT.root_rotation])
    candidate_root = decode_rot6d_safe(final_output[..., MOTION_LAYOUT.root_rotation])
    baseline_pitch = _root_forward(baseline_root)[3].detach().cpu().numpy()
    candidate_pitch = _root_forward(candidate_root)[3].detach().cpu().numpy()
    np.savez_compressed(
        output_root / "curves.npz",
        baseline_pitch_deg=baseline_pitch,
        candidate_pitch_deg=candidate_pitch,
        target_pitch_deg=baseline_pitch - float(target_delta_deg),
        valid_mask=data["valid_mask"].detach().cpu().numpy(),
    )
    if naturalness is not None:
        write_json_strict(output_root / "naturalness.json", naturalness)
    (output_root / "resolved_config.yaml").write_text(
        "\n".join([
            f"protocol: {PROTOCOL}",
            "role: pure_v2_evaluation",
            "counts_as_v2_success: false",
            f"input_run: {run_root}",
            f"target_delta_deg: {float(target_delta_deg)}",
            "evaluation:",
            "  root_pitch_mae_deg: 1.0",
            "  root_forward_p95_deg: 2.0",
            "  tail_extra_so3_jump_deg: 2.0",
            "  tail_extra_pitch_step_deg: 2.0",
            "  trunk_direction_p95_deg: 2.0",
            "  q_rigid: 0.2",
            "contact:",
            "  contact_height_m: 0.025",
            "  contact_speed_m_per_frame: 0.030",
            "  flat_gap_m: 0.020",
            "  minimum_height_frames: 3",
            "  minimum_sliding_pairs: 3",
            "diagnostic_separation: source_noise_and_kinematic_results_are_not_v2_success",
        ]) + "\n",
        encoding="utf-8",
    )
    (output_root / "EXECUTION_SPEC.md").write_text(
        f"""# v2 统一诊断执行规格

固定对象为 sample94、seed0、+10°；本次纯 v2 输入运行是 `{run_root}`。动态测试、采样、诊断和渲染均通过 `connect_server.py` 在服务器执行。纯 v2 只使用源噪声目标；运动学和源噪声子空间对照均标记为 `diagnostic_oracle`，不参与候选选择。

评价使用一致化 M0、直接姿态权威和统一 FK；接触证据不足返回 `NOT_EVALUABLE`。停止门、自然性、躯干随动和尾部约束分别记录，失败优先于无法评价。
""",
        encoding="utf-8",
    )
    (output_root / "RUNBOOK.md").write_text(
        """# v2 诊断复现运行手册

服务器环境：`/root/miniconda3/envs/mdm5090/bin/python`，项目目录：`/root/autodl-tmp/vimogen_clean`。

1. 通过 `connect_server.py` 运行停止门。
2. 运行 `scripts/diagnose_relative_root_forward_v2.py audit`、`gradient`、`kinematic` 和 `subspace`。
3. 运行 `scripts/diagnose_relative_root_forward_v2.py report`。
4. 用统一化 M0、纯 v2 实际输出和 `kinematic_oracle_motion.pt` 运行网格渲染器。

不要删除失败 attempt；不要把诊断对照复制回纯 v2 结果目录或用于调参。
""",
        encoding="utf-8",
    )
    write_json_strict(output_root / "gates.json", {
        "protocol": PROTOCOL,
        "role": "pure_v2_evaluation",
        "counts_as_v2_success": total_status == PASS,
        "status": total_status,
        "phase_gate_status": {
            "representation_and_control": total_status,
            "naturalness": naturalness_gate["status"],
        },
        "gates": gate_dicts,
    })
    write_json_strict(output_root / "causal_audit.json", audit)
    write_json_strict(output_root / "metrics.json", metrics)
    _write_metrics_csv(output_root / "metrics.csv", metrics)
    manifest = {
        "protocol": PROTOCOL,
        "role": "pure_v2_evaluation",
        "counts_as_v2_success": False,
        "input_run": str(run_root),
        "target_delta_deg": float(target_delta_deg),
        "sample_ids": [str(value) for value in data["sample_ids"]],
        "valid_frame_counts": data["valid_mask"].sum(dim=1).tolist(),
        "inputs": {
            name: {"path": str(path), "sha256": tensor_sha256(data[key])}
            for name, path, key in (
                ("m0_raw", data["m0_raw_path"], "m0_raw_norm"),
                ("m0_official", data["m0_official_path"], "m0_official_norm"),
                ("candidate_archive", data["archive_path"], "candidate_norm"),
            )
        },
        "outputs": {str(path.relative_to(output_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in output_root.rglob("*") if path.is_file()},
    }
    write_json_strict(output_root / "manifest.json", manifest)
    run_record = {
        "protocol": PROTOCOL,
        "execution_status": "COMPLETED",
        "optimization_status": "INHERITED_FROM_INPUT_RUN",
        "output_source": "actual_candidate_after_unified_authority_audit",
        "input_run": str(run_root),
        "status": total_status,
    }
    write_json_strict(output_root / "run_record.json", run_record)
    return {"output_root": str(output_root), "status": total_status, "gate_count": len(gate_dicts)}


def run_gradient(run_root: Path, output_root: Path) -> dict[str, Any]:
    """Archive the server stop-gate evidence as the gradient audit."""

    candidates = list(run_root.glob("**/differentiable_50step_gate.json"))
    if not candidates:
        candidates = list((ROOT / "results/phase7/relative_root_forward_v2/gates").glob("**/differentiable_50step_gate.json"))
    if not candidates:
        result = {"execution_status": "COMPLETED", "status": NOT_EVALUABLE, "reason": "server stop-gate artifact is absent"}
    else:
        gate = json.loads(candidates[-1].read_text(encoding="utf-8"))
        result = {
            "execution_status": "COMPLETED",
            "status": PASS if gate.get("passed") else FAIL,
            "source": str(candidates[-1]),
            "bitwise_reproduction": gate.get("bitwise_reproduction"),
            "gradient": gate.get("gradient"),
            "memory_mib": gate.get("memory_mib"),
            "timing_seconds": gate.get("timing_seconds"),
            "checkpoint_policy": "gradient_checkpointing=True; no unchecked full-model comparison claimed",
            "counts_as_v2_success": False,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_strict(output_root / "gradient_checks.json", result)
    return result


def _load_motion_for_diagnostic(run_root: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = _load_run(run_root)
    return data["m0_official"], data["valid_mask"]


def run_kinematic(run_root: Path, output_root: Path, target_delta_deg: float) -> dict[str, Any]:
    """Solve an independent per-frame kinematic oracle with SciPy SLSQP."""

    from scipy.optimize import minimize
    from motion_rep.consistency_v2 import default_smplx_neutral_22_skeleton, differentiable_forward_kinematics
    from motion_rep.rotation_transform import axis_angle_to_mat3x3, mat3x3_to_axis_angle

    m0, mask = _load_motion_for_diagnostic(run_root)
    baseline = authority_project(m0.float(), valid_mask=mask, output_dtype=torch.float32).physical_motion
    skeleton = default_smplx_neutral_22_skeleton()
    root_index = {"left_hip": 1, "right_hip": 2, "spine1": 3}
    foot_indices = (10, 11)
    spine1_index, neck_index = 3, 9
    target_roots = []
    target_frames = []
    solver_rows = []
    tolerance_rad = float(np.deg2rad(1.0))
    cos_tolerance = float(np.cos(tolerance_rad))
    foot_tolerance_m = 0.00025

    def make_frame(base_frame: torch.Tensor, x: torch.Tensor, target_root: torch.Tensor):
        body0 = decode_rot6d_safe(base_frame[MOTION_LAYOUT.body_pose].reshape(21, 6))
        deltas = axis_angle_to_mat3x3(x[:63].reshape(21, 3))
        body = deltas @ body0
        translation = x[63:66]
        fk = differentiable_forward_kinematics(
            body.unsqueeze(0), target_root.unsqueeze(0), translation.unsqueeze(0), skeleton=skeleton
        )
        return body, fk.joints[0]

    for frame in range(int(mask[0].sum().item())):
        base = baseline[0, frame]
        base_root = decode_rot6d_safe(base[MOTION_LAYOUT.root_rotation])
        _, _, right_axis, _ = _root_forward(base_root)
        target_root = axis_angle_to_mat3x3(
            right_axis * (-float(target_delta_deg) * np.pi / 180.0)
        ) @ base_root
        target_roots.append(target_root)
        base_body = decode_rot6d_safe(base[MOTION_LAYOUT.body_pose].reshape(21, 6))
        target_relative = target_root.transpose(-1, -2) @ base_root
        x0 = torch.zeros(66, dtype=torch.float64)
        for joint in root_index.values():
            x0[(joint - 1) * 3:joint * 3] = mat3x3_to_axis_angle(target_relative.double())
        base_fk = differentiable_forward_kinematics(
            base_body.double().unsqueeze(0), base_root.double().unsqueeze(0), base[MOTION_LAYOUT.root_translation].double().unsqueeze(0), skeleton=skeleton
        ).joints[0]
        init_body = axis_angle_to_mat3x3(x0[:63].reshape(21, 3)).float() @ base_body
        init_fk = differentiable_forward_kinematics(
            init_body.double().unsqueeze(0), target_root.double().unsqueeze(0), base[MOTION_LAYOUT.root_translation].double().unsqueeze(0), skeleton=skeleton
        ).joints[0]
        base_hip_mid = 0.5 * (base_fk[1] + base_fk[2])
        init_hip_mid = 0.5 * (init_fk[1] + init_fk[2])
        x0[63:66] = base[MOTION_LAYOUT.root_translation].double() + base_hip_mid - init_hip_mid
        base_trunk = base_fk[neck_index] - base_fk[spine1_index]
        base_trunk = base_trunk / torch.linalg.vector_norm(base_trunk).clamp_min(1e-8)
        base_feet = base_fk[list(foot_indices)]
        base_translation = base[MOTION_LAYOUT.root_translation].double()

        def evaluate(x_array: np.ndarray, *, need_jac: bool = False):
            x = torch.tensor(x_array, dtype=torch.float64, requires_grad=need_jac)
            _, joints = make_frame(base.double(), x, target_root.double())
            trunk = joints[neck_index] - joints[spine1_index]
            trunk = trunk / torch.linalg.vector_norm(trunk).clamp_min(1e-8)
            foot_delta = joints[list(foot_indices)] - base_feet
            constraints = torch.cat((
                (trunk * base_trunk).sum().reshape(1) - cos_tolerance,
                (foot_tolerance_m ** 2 - foot_delta.square().sum(dim=-1)),
            ))
            rotation_cost = (x[:63] / np.deg2rad(1.0)).square().sum()
            translation_cost = ((x[63:66] - base_translation) / 0.001).square().sum()
            objective = rotation_cost + translation_cost
            if not need_jac:
                return float(objective.detach()), constraints.detach().numpy()
            jac_obj = torch.autograd.grad(objective, x, retain_graph=True)[0]
            jac_constraints = []
            for value in constraints:
                jac_constraints.append(torch.autograd.grad(value, x, retain_graph=True)[0])
            return (
                float(objective.detach()),
                jac_obj.detach().numpy(),
                constraints.detach().numpy(),
                torch.stack(jac_constraints).detach().numpy(),
            )

        def objective(x_array):
            return evaluate(x_array, need_jac=True)[0]

        def objective_jac(x_array):
            return evaluate(x_array, need_jac=True)[1]

        def constraint_fun(x_array):
            return evaluate(x_array, need_jac=False)[1]

        def constraint_jac(x_array):
            return evaluate(x_array, need_jac=True)[3]

        bounds = [(-np.deg2rad(30.0), np.deg2rad(30.0))] * 63 + [
            (float(value) - 0.05, float(value) + 0.05) for value in base_translation
        ]
        solved = minimize(
            objective,
            x0.detach().numpy(),
            jac=objective_jac,
            bounds=bounds,
            constraints={"type": "ineq", "fun": constraint_fun, "jac": constraint_jac},
            method="SLSQP",
            options={"maxiter": 100, "ftol": 1e-12, "disp": False},
        )
        x_final = torch.tensor(solved.x, dtype=torch.float64)
        body_final, joints_final = make_frame(base.double(), x_final, target_root.double())
        frame_motion = base.clone()
        frame_motion[MOTION_LAYOUT.body_pose] = encode_rot6d(body_final.float()).reshape(-1)
        frame_motion[MOTION_LAYOUT.root_rotation] = encode_rot6d(target_root.float())
        frame_motion[MOTION_LAYOUT.root_translation] = x_final[63:66].float()
        target_frames.append(frame_motion)
        final_constraints = constraint_fun(solved.x)
        solver_rows.append({
            "frame": frame,
            "success": bool(solved.success),
            "message": str(solved.message),
            "iterations": int(getattr(solved, "nit", 0)),
            "objective": float(solved.fun),
            "trunk_constraint_margin": float(final_constraints[0]),
            "foot_constraint_margins_m2": [float(value) for value in final_constraints[1:]],
            "within_trust_region": bool(np.max(np.abs(solved.x[:63])) <= np.deg2rad(30.0) + 1e-8 and np.max(np.abs(solved.x[63:66] - base_translation.numpy())) <= 0.05 + 1e-8),
        })

    oracle = torch.stack(target_frames, dim=0).unsqueeze(0)
    oracle = authority_project(oracle, valid_mask=mask, output_dtype=torch.float32).physical_motion
    residual = compute_relative_root_forward_metrics(
        baseline, oracle, mask, target_delta_deg, protocol_name="diagnostic_oracle"
    )
    result = {
        "protocol": "diagnostic_oracle_kinematic_slsqp_fixed_contact_joint_proxy",
        "role": "diagnostic_oracle",
        "counts_as_v2_success": False,
        "status": "CONSTRAINTS_FEASIBLE" if all(
            all(value >= -1e-6 for value in row["foot_constraint_margins_m2"])
            and row["trunk_constraint_margin"] >= -1e-6
            for row in solver_rows
        ) else "INCOMPLETE",
        "solver": "scipy.optimize.SLSQP",
        "solver_constraints": {
            "trunk_direction_deg": 1.0,
            "foot_joint_position_m": foot_tolerance_m,
            "local_rotation_bound_deg": 30.0,
            "root_translation_bound_m": 0.05,
            "maxiter": 100,
        },
        "target_delta_deg": float(target_delta_deg),
        "valid_frames": int(mask[0].sum()),
        "solver_success_count": sum(row["success"] for row in solver_rows),
        "solver_failure_count": sum(not row["success"] for row in solver_rows),
        "constraint_satisfied_count": sum(
            row["trunk_constraint_margin"] >= -1e-6
            and all(value >= -1e-6 for value in row["foot_constraint_margins_m2"])
            for row in solver_rows
        ),
        "solver_rows": solver_rows,
        "root_control": residual["per_sample"][0],
        "whole_body": residual["whole_body"],
        "interpretation": "kinematic reachability evidence only; foot constraints use fixed SMPL-X foot-joint proxies and make no dynamics claim",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    torch.save(oracle.cpu(), output_root / "kinematic_oracle_motion.pt")
    write_json_strict(output_root / "reachability.json", result)
    return result


def run_subspace(run_root: Path, output_root: Path, target_delta_deg: float, seed: int = 314159) -> dict[str, Any]:
    """Collect the paired server probe and its independent coefficient solve."""

    probe_paths = list(run_root.glob("subspace_probe_artifacts/**/subspace_probe.json"))
    if not probe_paths:
        status = NOT_EVALUABLE
        probe = {}
        directions = []
        actual_rms = []
        actual_validation = []
        response_copied = False
    else:
        probe_path = probe_paths[-1]
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        directions = [record.get("sha256") for record in probe.get("direction_records", [])]
        actual_rms = [record.get("rms") for record in probe.get("direction_records", [])]
        actual_validation = probe.get("actual_validation", [])
        solver = probe.get("linear_solver", {})
        status = "DIAGNOSTIC_COMPLETE" if probe.get("response_matrices") is not None or solver else NOT_EVALUABLE
        response_path = probe_path.with_name("subspace.npz")
        if response_path.is_file():
            values = np.load(response_path)
            output_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_root / "subspace.npz", **{key: values[key] for key in values.files})
            response_copied = True
        else:
            response_copied = False
    result = {
        "protocol": "diagnostic_oracle_source_noise_subspace",
        "role": "diagnostic_oracle",
        "counts_as_v2_success": False,
        "status": status,
        "reason": "paired 50-step outputs and actual scale replays are stored separately from the pure v2 result",
        "target_delta_deg": float(target_delta_deg),
        "direction_seed": seed,
        "direction_count": len(directions),
        "direction_sha256": directions,
        "actual_direction_rms": actual_rms,
        "requested_rms": [0.005, 0.01],
        "response_matrix": "subspace.npz" if probe else None,
        "linear_solver": probe.get("linear_solver"),
        "actual_validation": actual_validation,
        "input_run": str(run_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_strict(output_root / "subspace.json", result)
    if not response_copied:
        np.savez_compressed(output_root / "subspace.npz", direction_rms=np.asarray(actual_rms, dtype=np.float32))
    return result


def run_report(output_root: Path) -> dict[str, Any]:
    gates = _load_optional_json(output_root / "gates.json") or {"status": NOT_EVALUABLE, "gates": []}
    gradient = _load_optional_json(output_root / "gradient_checks.json") or {"status": NOT_EVALUABLE}
    reachability = (
        _load_optional_json(output_root / "reachability.json")
        or _load_optional_json(output_root / "kinematic_slsqp/reachability.json")
        or _load_optional_json(output_root / "kinematic/reachability.json")
        or {"status": NOT_EVALUABLE}
    )
    subspace = (
        _load_optional_json(output_root / "subspace.json")
        or _load_optional_json(output_root / "subspace/subspace.json")
        or {"status": NOT_EVALUABLE}
    )
    metrics = _load_optional_json(output_root / "metrics.json") or {}
    row = (metrics.get("per_sample") or [{}])[0]
    whole = metrics.get("whole_body") or {}
    natural = _load_optional_json(output_root / "naturalness.json") or {}
    natural_gate = natural.get("naturalness_gate") or {}
    actual = subspace.get("actual_validation") or []
    best_actual = min(actual, key=lambda item: item.get("pitch_p95_deg", float("inf"))) if actual else {}
    report = f"""# 根前向引导 v2 统一诊断报告

协议：`{PROTOCOL}`
纯 v2 统一评价总门：`{gates.get('status', NOT_EVALUABLE)}`
服务器梯度停止门：`{gradient.get('status', NOT_EVALUABLE)}`

纯 v2 实际输出

- 根俯仰平均绝对误差：{row.get('mean_absolute_error_deg')}°；完整前向 P95：{row.get('forward_vector_error_p95_deg')}°；剂量符号：{row.get('dose_sign_correct')}。
- 躯干方向 P95：{(whole.get('trunk_change_deg') or {}).get('p95')}°；刚性随动比 `q_rigid`：{whole.get('q_rigid')}。
- 自然性外审状态：`{natural_gate.get('status', natural_gate.get('passed'))}`；具体滑动、离地、穿地和脚尖回归见 `naturalness.json`。

隔离诊断

- 运动学对照：`{reachability.get('status', NOT_EVALUABLE)}`；约束满足帧数 `{reachability.get('constraint_satisfied_count')}/{reachability.get('valid_frames')}`，SLSQP 成功返回帧数 `{reachability.get('solver_success_count')}`。数值线搜索警告已与约束残差分开记录。
- 源噪声子空间：方向数 `{subspace.get('direction_count')}`，状态 `{subspace.get('status', NOT_EVALUABLE)}`；线性求解器成功：`{(subspace.get('linear_solver') or {}).get('success')}`；四个真实比例中最佳根俯仰 P95：`{best_actual.get('pitch_p95_deg')}`°。

本报告把纯 v2 输出、运动学诊断对照和源噪声子空间诊断分开。对照结果的 `counts_as_v2_success` 固定为 `false`。

所有门槛、有效样本数和原始产物位置见 `gates.json`、`metrics.csv`、`causal_audit.json` 及 `manifest.json`。
"""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "USER_REPORT.md").write_text(report, encoding="utf-8")
    write_json_strict(output_root / "report.json", {
        "protocol": PROTOCOL,
        "status": gates.get("status", NOT_EVALUABLE),
        "pure_v2": gates,
        "gradient": gradient,
        "kinematic": reachability,
        "subspace": subspace,
    })
    return {"output_root": str(output_root), "status": gates.get("status", NOT_EVALUABLE)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "gradient", "kinematic", "subspace"):
        command = sub.add_parser(name)
        command.add_argument("--input-run", type=Path, default=DEFAULT_RUN)
        command.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
        command.add_argument("--target-delta-deg", type=float, default=10.0)
    sub.add_parser("report").add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    sub.choices["subspace"].add_argument("--direction-seed", type=int, default=314159)
    args = parser.parse_args()
    if args.command == "audit":
        result = run_audit(args.input_run, args.output_root, args.target_delta_deg)
    elif args.command == "gradient":
        result = run_gradient(args.input_run, args.output_root)
    elif args.command == "kinematic":
        result = run_kinematic(args.input_run, args.output_root, args.target_delta_deg)
    elif args.command == "subspace":
        result = run_subspace(args.input_run, args.output_root, args.target_delta_deg, args.direction_seed)
    else:
        result = run_report(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == FAIL and args.command in {"audit", "gradient"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
