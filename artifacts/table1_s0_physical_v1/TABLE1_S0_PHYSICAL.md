# Table 1. S0 骨盆控制与物理闭环（调参前）

| 方法 | 角度 MAE（°）↓ | 角度 P95（°）↓ | 斜率→1 | 角度通过 | MPJPE（mm）↓ | 物理通过 | 穿透 P95（mm）↓ | 接触切速 P95（mm/帧）↓ | 支撑高度误差 P95（mm）↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 能量引导 | 0.273 [0.148, 0.487] | 0.557 [0.346, 0.940] | 0.788 | 12/12 | 8.4 [2.7, 9.7] | 0/12 | 0.0 [0.0, 0.0] | 28.1 [25.9, 29.0] | 2.9 [1.3, 3.9] |
| M2 源噪声优化 | 0.757 [0.044, 0.984] | 1.308 [0.100, 2.187] | 0.996 | 8/12 | 33.8 [2.9, 72.4] | 2/12 | 0.0 [0.0, 0.0] | 26.5 [22.7, 28.2] | 6.2 [1.0, 10.9] |
| M3 局部投影 | 0.181 [0.173, 0.200] | 0.425 [0.384, 0.505] | 0.950 | 12/12 | 34.5 [30.6, 63.6] | 0/12 | 0.0 [0.0, 7.7] | 34.4 [29.6, 39.0] | 12.1 [9.1, 19.0] |
| M4 前向射击 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 | 12/12 | 51.9 [45.8, 59.5] | 0/12 | 0.0 [0.0, 4.3] | 31.1 [27.4, 33.2] | 16.5 [10.5, 20.6] |
| M5 原始—对偶流 | 1.487 [0.054, 1.560] | 1.573 [0.196, 1.769] | 0.230 | 4/12 | 3.6 [2.5, 3.8] | 2/12 | 0.0 [0.0, 0.0] | 26.4 [25.2, 27.8] | 1.3 [1.1, 1.7] |
| M6 李雅普诺夫引导 | 0.467 [0.086, 0.501] | 0.600 [0.292, 0.979] | 0.730 | 12/12 | 8.0 [2.4, 9.5] | 2/12 | 0.0 [0.0, 0.0] | 27.2 [24.0, 29.1] | 2.5 [1.0, 3.9] |
| M7 生成后几何编辑† | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 1.000 | 12/12 | 13.4 [0.0, 13.5] | 1/12 | 0.0 [0.0, 0.0] | 26.9 [25.3, 28.7] | 4.4 [1.1, 6.9] |

物理门使用冻结协议 `vimogen_m1_m7_physical_thresholds_v1`。数值为每方法 12 条序列的中位数 [Q1, Q3]；通过数按逐序列全部物理阈值联合判定。

本表已完成 S0 物理闭环，但仍是调参前诊断表。M2/M3/M4 的结构修正、M5 有限预算调参及 S1 唯一配置冻结完成前，不得作为最终论文主表，也不得启动 S2。M7 始终仅为生成后几何编辑参考。

## 物理失败原因计数

- M1：contact_height_max_mm_fail=3；contact_height_p95_mm_fail=2；contact_tangent_speed_max_mm_per_frame_fail=1；contact_tangent_speed_p95_mm_per_frame_fail=1；floating_frame_rate_fail=6；max_segment_slide_distance_mm_fail=1；penetration_frame_rate_fail=1；penetration_max_mm_fail=2；support_height_error_p95_mm_fail=10
- M2：contact_height_max_mm_fail=6；contact_height_p95_mm_fail=4；contact_tangent_speed_max_mm_per_frame_fail=5；contact_tangent_speed_p95_mm_per_frame_fail=2；floating_frame_rate_fail=7；max_segment_slide_distance_mm_fail=1；penetration_frame_rate_fail=3；penetration_max_mm_fail=1；penetration_p95_mm_fail=1；support_height_error_p95_mm_fail=8
- M3：contact_height_max_mm_fail=8；contact_height_p95_mm_fail=7；contact_tangent_speed_max_mm_per_frame_fail=11；contact_tangent_speed_p95_mm_per_frame_fail=9；floating_frame_rate_fail=8；max_segment_slide_distance_mm_fail=6；penetration_frame_rate_fail=7；penetration_max_mm_fail=6；penetration_p95_mm_fail=3；support_height_error_p95_mm_fail=12；total_slide_distance_mm_fail=6
- M4：contact_height_max_mm_fail=9；contact_height_p95_mm_fail=8；contact_tangent_speed_max_mm_per_frame_fail=10；contact_tangent_speed_p95_mm_per_frame_fail=6；floating_frame_rate_fail=9；max_segment_slide_distance_mm_fail=4；penetration_frame_rate_fail=12；penetration_max_mm_fail=6；penetration_p95_mm_fail=3；support_height_error_p95_mm_fail=12；total_slide_distance_mm_fail=6
- M5：contact_tangent_speed_max_mm_per_frame_fail=1；contact_tangent_speed_p95_mm_per_frame_fail=1；floating_frame_rate_fail=3；penetration_max_mm_fail=1；support_height_error_p95_mm_fail=10
- M6：contact_height_max_mm_fail=3；contact_height_p95_mm_fail=1；floating_frame_rate_fail=6；penetration_max_mm_fail=1；support_height_error_p95_mm_fail=8
- M7：contact_height_max_mm_fail=6；contact_height_p95_mm_fail=2；contact_tangent_speed_max_mm_per_frame_fail=5；contact_tangent_speed_p95_mm_per_frame_fail=3；floating_frame_rate_fail=7；penetration_frame_rate_fail=1；penetration_max_mm_fail=1；support_height_error_p95_mm_fail=10
