# ViMoGen M1–M7 骨盆控制规模实验

本分支实现并验证 ViMoGen 骨盆姿态控制的 M1–M7 统一比较框架。当前主线是冻结协议下的 **S0 物理闭环与 S1 有界筛选**：7 种方法、2 个动作、2 个随机种子和 3 个 S0 剂量，共 84 条受控序列。

当前结论不是最终论文结论。S0 的严格物理门 v1 与独立校准的物理门 v2 均已完成；v1 保留历史审计结论，v2 用于 S1 配置筛选。S1 的有界筛选、压力测试和唯一配置冻结均已完成；S2 按方案未启动。

面向论文写作的累计结果、结论边界和完整产物索引统一维护在 [论文结果账本](result.md)；后续每次新增、重跑或冻结实验均同步更新该文件。

## 当前状态

- 当前分支：`codex/m1-m7-s0-physical-closure`
- 冻结协议：`vimogen_pelvis_m1_m7_scale_v1`
- 约束包：C0，仅控制局部矢状面骨盆角
- S0 动作：sample94、sample34122
- 随机种子：0、42
- 剂量：−2°、0°、+2°
- 每种方法：12 条序列
- 总序列数：84 条，其中 M1–M6 为 72 条，M7 为 12 条
- 服务器专项回归：`43 passed`
- S0 状态：84 条物理数值已完成 v1/v2 双轨评价；v1 `7/84`，v2 `22/84`
- S1 状态：已完成有界筛选；M2 为 `S1_FAILED_INVARIANT`，M1/M3/M5/M6 为 `S1_FAILED_ANGLE`，M4 为 `S1_FAILED_PHYSICAL`；无方法获得 `S1_PASS`
- S2 状态：未启动；S1 未产生满足全部门的生成方法，继续启动 S2 不符合冻结方案

## 研究目标

实验使用统一的动作权威边界、同一骨盆角定义、同一配对 M0、同一剂量和同一评价契约，比较七类控制机制：

| 方法 | 机制 | 当前角色 |
|---|---|---|
| M1 | 采样状态能量引导 | 生成期控制候选 |
| M2 | 源噪声优化 | 完整轨迹可微优化候选 |
| M3 | 局部投影流 | 采样期投影候选 |
| M4 | 前向射击与终端高斯—牛顿校正 | 规划式控制候选 |
| M5 | 原始—对偶流 | 约束优化候选 |
| M6 | 李雅普诺夫伪投影 | 稳定性引导候选 |
| M7 | 生成后几何编辑 | 角度命中参考，不代表生成器自然响应 |

动作表示以身体姿态、根旋转和根平移为直接权威通道；关节、关节速度、根旋转速度和根平移速度均由权威动作重新计算。候选不得从自身重新定义接触帧、地面或目标曲线。

## 本次实施过程

### 1. 冻结公共协议与评价契约

首先建立 `guidance/` 下的统一方法接口，以及 `configs/m1_m7/` 下的公共协议和七种方法配置。协议冻结了：

- 直接与派生动作通道；
- 局部矢状面骨盆角评价器；
- C0–C3 约束包；
- 七档完整剂量与 S0 的 −2°/0°/+2° 子集；
- S0 动作、随机种子和每方法最多 8 组等价调参预算；
- 角度、数值稳定性、物理副作用、内容保持和计算成本的选择顺序。

核心框架提交为 `57a689d`。

### 2. 接入 M1–M7 方法

七种机制分别位于：

- `guidance/m1_loss_guidance.py`
- `guidance/m2_dflow_source.py`
- `guidance/m3_projflow_local.py`
- `guidance/m4_pcfm.py`
- `guidance/m5_ldf.py`
- `guidance/m6_lyaguide.py`
- `guidance/m7_paht_edit.py`

M1、M3、M5、M6 通过统一采样钩子运行；M2 使用可微完整轨迹反传并只优化源噪声；M4 在预定噪声尺度执行剩余轨迹射击和终端求解；M7 读取配对 M0 缓存执行生成后几何编辑。

### 3. 真实服务器冒烟与边界修正

真实 ViMoGen 冒烟中发现并保留了以下问题和修正记录：

- M6 首次运行方向错误，但梯度有限。原因是噪声尺度递减时李雅普诺夫导数符号使用错误；在 `91722a2` 修正后，独立运行通过当前角度门。
- M2 首次运行把标准化动作当作物理动作计算角度，导致方向和全身内容严重错误；在 `951d410` 增加标准化到物理权威动作的边界转换。修正后方向正确，但仍存在过冲和个体不稳定。
- M4 首次运行暴露批上下文、批统计广播和嵌套配置解析问题；分别在 `4d9b292`、`83e9713`、`89ee3ce` 修正。修正后角度可精确命中，但内容副作用仍然存在。

失败尝试均保留在独立 attempt 目录，没有覆盖或选择性删除。

### 4. 可恢复 S0 矩阵运行

`experiments/run_s0_matrix.py` 顺序执行 M1–M6 的 36 个唯一批次，每批包含两个动作。运行器使用方法、随机种子和剂量作为唯一键，保存每次命令、配置、运行状态和评价状态；已完成项可跳过，失败评价可单独恢复。

可恢复运行器和进度修正对应提交：

- `4bb22b6`：增加 S0 矩阵运行器；
- `85048a3`：增加评价恢复能力；
- `2aaa02b`：修正唯一键完成状态统计。

M7 由 `experiments/run_m7_smoke_from_cache.py` 使用两组经过验证的配对 M0 缓存独立生成 12 条序列。最终 S0 共 84 条序列。

### 5. 统一评价与 Table 1

每条序列先经过权威动作重建，再计算：

- 骨盆角平均绝对误差和 P95；
- 非零剂量符号；
- 剂量响应斜率；
- 零剂量漂移；
- 相对 M0 的 22 关节平均位置误差；
- 根平移 P95 偏差；
- 物理评价状态。

`experiments/build_s0_table1.py` 对严格 JSON 结果做完整性检查并生成 Markdown、LaTeX、PNG 和机器可读 JSON。生成器不会把缺失的脚部标记解释为物理门通过。

## S0 预备结果

### 角度控制

| 方法 | 角度 MAE（°）↓ | 角度 P95（°）↓ | 符号正确率↑ | 剂量响应斜率→1 | 零剂量漂移（°）↓ | 序列角度通过率↑ |
|---|---:|---:|---:|---:|---:|---:|
| M1 能量引导 | 0.273 [0.148, 0.487] | 0.557 [0.346, 0.940] | 100.0% | 0.788 | 0.113 [0.100, 0.129] | 12/12 |
| M2 源噪声优化 | 0.757 [0.044, 0.984] | 1.308 [0.100, 2.187] | 100.0% | 0.996 | 0.038 [0.037, 0.040] | 8/12 |
| M3 局部投影 | 0.181 [0.173, 0.200] | 0.425 [0.384, 0.505] | 100.0% | 0.950 | 0.173 [0.172, 0.180] | 12/12 |
| M4 前向射击 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 100.0% | 1.000 | 0.000 [0.000, 0.000] | 12/12 |
| M5 原始—对偶流 | 1.487 [0.054, 1.560] | 1.573 [0.196, 1.769] | 100.0% | 0.230 | 0.045 [0.039, 0.052] | 4/12 |
| M6 李雅普诺夫引导 | 0.467 [0.086, 0.501] | 0.600 [0.292, 0.979] | 100.0% | 0.730 | 0.082 [0.069, 0.085] | 12/12 |
| M7 生成后几何编辑† | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 100.0% | 1.000 | 0.000 [0.000, 0.000] | 12/12 |

数值为每种方法 12 条 S0 序列的中位数 `[Q1, Q3]`。当前序列角度门为 MAE≤1°、P95≤2°，且非零剂量符号正确。†M7 是生成后编辑参考。

### 内容保持诊断

| 方法 | 22 关节 MPJPE vs M0（mm）↓ | 根平移 P95 偏差（mm）↓ | 最大 MPJPE（mm） | 物理 v1 | 物理 v2 |
|---|---:|---:|---:|---:|---:|
| M1 能量引导 | 8.4 [2.7, 9.7] | 4.5 [4.1, 5.0] | 11.0 | 0/12 | 4/12 |
| M2 源噪声优化 | 33.8 [2.9, 72.4] | 39.3 [5.2, 105.8] | 552.7 | 2/12 | 2/12 |
| M3 局部投影 | 34.5 [30.6, 63.6] | 50.9 [45.5, 93.1] | 148.8 | 0/12 | 0/12 |
| M4 前向射击 | 51.9 [45.8, 59.5] | 88.5 [74.5, 95.8] | 75.7 | 0/12 | 0/12 |
| M5 原始—对偶流 | 3.6 [2.5, 3.8] | 4.4 [3.9, 4.7] | 4.0 | 2/12 | 8/12 |
| M6 李雅普诺夫引导 | 8.0 [2.4, 9.5] | 4.6 [4.1, 4.9] | 10.7 | 2/12 | 5/12 |
| M7 生成后几何编辑† | 13.4 [0.0, 13.5] | 0.0 [0.0, 0.0] | 13.7 | 1/12 | 3/12 |

![S0 预备 Table 1](artifacts/table1_s0_preliminary/table1_s0_preliminary.png)

完整表格与机器可读数据（该段为 S0 早期角度/内容预备表，物理数值以“双轨 Table 1”小节为准）：

- [Markdown 表格](artifacts/table1_s0_preliminary/TABLE1_S0_PRELIMINARY.md)
- [LaTeX 表格](artifacts/table1_s0_preliminary/table1_s0_preliminary.tex)
- [汇总 JSON](artifacts/table1_s0_preliminary/table1_s0_summary.json)
- [PNG 图片](artifacts/table1_s0_preliminary/table1_s0_preliminary.png)

### S0 物理闭环更新

共享 M0 的逐标记点脚跟/脚尖接触、地面和有效帧证据已经物化。v1 阈值只由 4 条配对 M0 自评和程序化扰动冻结；v2 在不读取候选或 S2 结果的前提下，使用预注册的支撑误差/悬空阶梯校准，选定支撑误差 `≤3 mm`、悬空率 `≤2.5%`，并取消接触高度作为 v2 硬门但继续报告其数值。

84 条序列的物理通过数为 v1 `7/84`、v2 `22/84`；方法级 v1/v2 分别为 M1 `0/12`/`4/12`、M2 `2/12`/`2/12`、M3 `0/12`/`0/12`、M4 `0/12`/`0/12`、M5 `2/12`/`8/12`、M6 `2/12`/`5/12`、M7 `1/12`/`3/12`。两套门均保留原始 11 项数值、阈值、失败条数和左右脚跟/脚尖明细。

服务器产物目录：`/root/autodl-tmp/vimogen_m1_m7_scale/results/phase9/pelvis_m1_m7/`。

- [双轨 Table 1 Markdown（服务器）](/root/autodl-tmp/vimogen_m1_m7_scale/results/phase9/pelvis_m1_m7/table1_s0_physical_v2/attempt_01/TABLE1_S0_PHYSICAL.md)
- [双轨 Table 1 汇总 JSON（服务器）](/root/autodl-tmp/vimogen_m1_m7_scale/results/phase9/pelvis_m1_m7/table1_s0_physical_v2/attempt_01/table1_s0_physical_summary.json)
- [84 行物理附表（服务器）](/root/autodl-tmp/vimogen_m1_m7_scale/results/phase9/pelvis_m1_m7/table1_s0_physical_v2/attempt_01/TABLE1_S0_PHYSICAL_PER_SEQUENCE.md)
- [物理 v2 校准证据（服务器）](/root/autodl-tmp/vimogen_m1_m7_scale/results/phase9/pelvis_m1_m7/physical_thresholds_v2/attempt_02/calibration_evidence.json)

这张表是 S1 配置筛选的物理输入，不等于方法入围；v1 的 `7/84` 历史审计结论不可被 v2 覆盖。

### S1 有界筛选结果

S1 已按每方法最多 8 组配置完成屏选、前两名确认和 `±10°` 压力测试。确认阶段按角度通过数→数值稳定→物理 v2→物理严重度→内容→成本排序，每种方法冻结一个唯一配置；M2 因不变量失败跳过调参。

| 方法 | 冻结状态 | 配置 | confirm 角度 | confirm 物理 v2 | stress 角度 |
|---|---|---:|---:|---:|---:|
| M1 | `S1_FAILED_ANGLE` | 03 | 17/20 | 5/20 | 4/8 |
| M2 | `S1_FAILED_INVARIANT` | — | — | — | — |
| M3 | `S1_FAILED_ANGLE` | 05 | 4/20 | 19/20 | 0/8 |
| M4 | `S1_FAILED_PHYSICAL` | 06 | 20/20 | 4/20 | 8/8 |
| M5 | `S1_FAILED_ANGLE` | 05 | 16/20 | 5/20 | 0/8 |
| M6 | `S1_FAILED_ANGLE` | 02 | 12/20 | 7/20 | 0/8 |

服务器 S1 产物目录：`/root/autodl-tmp/vimogen_m1_m7_scale/results/phase9/pelvis_m1_m7/s1/`。

- [S1 主表（服务器）](/root/autodl-tmp/vimogen_m1_m7_scale/results/phase9/pelvis_m1_m7/s1/reports/S1_MAIN_TABLE.md)
- [S1 失败案例表（服务器）](/root/autodl-tmp/vimogen_m1_m7_scale/results/phase9/pelvis_m1_m7/s1/reports/S1_FAILURE_CASES.md)
- [S1 压力测试表（服务器）](/root/autodl-tmp/vimogen_m1_m7_scale/results/phase9/pelvis_m1_m7/s1/reports/S1_STRESS_TABLE.md)
- [S1 可复现性清单（服务器）](/root/autodl-tmp/vimogen_m1_m7_scale/results/phase9/pelvis_m1_m7/s1/reports/S1_REPRODUCIBILITY_CHECKLIST.md)
- [S1 冻结 JSON（服务器）](/root/autodl-tmp/vimogen_m1_m7_scale/results/phase9/pelvis_m1_m7/s1/S1_FROZEN.json)

## 结果诊断

### M1 与 M6

两者在 ±2° S0 上角度和内容表现较好，但 S1 确认显示 M1 仍为 `17/20`、M6 为 `12/20` 角度通过，均未达到完整角度门；压力测试只作稳定性记录，不改变失败结论。

### M2

M2 的重复批轨迹复现通过，但批量—单样本一致性失败，sample94 与 sample34122 均出现明显动作、源噪声、角度和 MPJPE 差异。因此按不变量优先规则冻结为 `S1_FAILED_INVARIANT`，未追加无界调参。

### M3

M3 的内容和物理 v2 在确认阶段相对较好（配置 05 为物理 `19/20`），但角度仅 `4/20`，因此冻结为 `S1_FAILED_ANGLE`；零剂量旁路审计本身通过。

### M4

M4 在确认阶段角度 `20/20`，但物理 v2 仅 `4/20`；压力测试角度 `8/8` 不能抵消物理失败，因此冻结为 `S1_FAILED_PHYSICAL`。射击时刻最多触发一次的修复已通过专项测试。

### M5

M5 数值稳定性在确认阶段为 `20/20`，高增益配置改善了角度响应，但最佳配置仍为角度 `16/20`、物理 `5/20`，冻结为 `S1_FAILED_ANGLE`。

### M7

M7 的精确角度是生成后几何编辑的预期结果，只能作为角度命中的参考上界。它不能证明 ViMoGen 在条件控制下自然地产生目标动作，也必须接受与其他方法相同的物理和内容评价。

## 为什么暂不进入 S2

S1 已完成，但没有方法满足进入 S2 的全部条件：

1. M2 因批量—单样本一致性失败，冻结为 `S1_FAILED_INVARIANT`；
2. M1、M3、M5、M6 未通过确认角度门；
3. M4 虽通过确认角度门，但物理 v2 仅 `4/20`；因此所有方法均未获得 `S1_PASS`。

不能用角度命中抵消内容或物理失败，也不能在看到 S0 结果后静默修改冻结的 v1 协议。任何结构性修正均应登记为新版本，并保留当前 S0-v1 作为诊断证据。

## 下一步

1. 保留 S1 冻结配置、失败案例、压力测试和复现清单，不追加配置。
2. 按方案停止在 S1，不启动 S2；任何 S2 申请都必须先形成新的冻结方案和新的门控理由。

## 运行入口

服务器需要完整 ViMoGen 模型、配置、检查点、数据和配对噪声缓存。本地仓库负责版本管理与推送，GPU 生成在服务器执行。

运行或恢复 M1–M6 的 S0 矩阵：

```bash
python experiments/run_s0_matrix.py \
  --code-commit <当前提交> \
  --manifest <S0清单> \
  --noise-cache <配对噪声缓存> \
  --output results/phase9/pelvis_m1_m7/s0_sampling
```

运行单个方法、剂量和随机种子：

```bash
python experiments/run_sampling_guidance_smoke.py \
  --method M1 \
  --dose 2 \
  --seed 0 \
  --code-commit <当前提交> \
  --manifest <S0清单> \
  --noise-cache <配对噪声缓存>
```

从已有评价结果生成 Table 1：

```bash
python experiments/build_s0_table1.py \
  --sampling-root results/phase9/pelvis_m1_m7/s0_sampling \
  --m7-root results/phase9/pelvis_m1_m7/s0_m7/attempt_01 \
  --output results/phase9/pelvis_m1_m7/table1_s0_preliminary
```

冻结并执行 S1 的屏选/确认/压力测试：

```bash
python experiments/run_s1_bounded_tuning.py \
  --stage all \
  --output results/phase9/pelvis_m1_m7/s1 \
  --code-commit <当前提交> \
  --m2-invariant <M2不变量报告> \
  --runtime-root <服务器运行时根目录> \
  --base-config <tm2m_infer.yaml> \
  --protocol <冻结协议> \
  --manifest <S1清单> \
  --noise-cache <配对噪声缓存> \
  --reference <physical_reference.pt> \
  --thresholds <v1 thresholds.json> \
  --thresholds-v2 <v2 thresholds.json>
```

## 重要文件

- `result.md`：面向论文写作的权威结果账本、结论边界、证据索引和更新记录；
- `PROJECT_MEMORY.md`：跨会话的完整实验状态、已验证事实和继续步骤；
- `configs/m1_m7/`：冻结公共协议和方法配置；
- `guidance/`：M1–M7 方法实现；
- `motion_rep/pose_authority.py`：直接动作权威边界和派生量重建；
- `experiments/run_s0_matrix.py`：可恢复 S0 矩阵运行器；
- `experiments/evaluate_sampling_guidance_smoke.py`：逐运行严格评价；
- `experiments/build_s0_table1.py`：S0 Table 1 生成器；
- `artifacts/table1_s0_preliminary/`：已归档的预备表格与机器可读结果。
- `evaluation/physical_reference.py`、`evaluation/physical_metrics.py`：共享 M0 物理证据与统一评价器；
- `scripts/calibrate_physical_thresholds.py`：仅基于 M0 与合成扰动冻结物理阈值；
- `scripts/calibrate_physical_thresholds_v2.py`：候选无关的物理门 v2 校准；
- `experiments/audit_m2_v2_reproducibility.py`、`experiments/audit_m2_v2_single_batch.py`：M2 不变量审计；
- `experiments/audit_zero_dose_identity.py`：M1–M6 零剂量旁路审计；
- `experiments/run_s1_bounded_tuning.py`：S1 屏选、确认、压力测试和唯一配置冻结；
- `experiments/build_s1_reports.py`：S1 主表、失败案例、压力测试和复现清单生成器。
- `results/phase9/pelvis_m1_m7/s1/`：服务器 S1 冻结结果和最终报告目录。

## 结果解释边界

- 当前结果只覆盖 C0、两个动作、两个随机种子和 ±2° 范围；
- S0 是机制检查，不构成大样本统计证据；
- 物理门 v1 已对 84 条 S0 序列完成评价，`7/84`；v2 独立校准后为 `22/84`，两套数值表同时保留；
- S1 有界筛选已完成；M2 为不变量失败，M1/M3/M5/M6 为角度失败，M4 为物理失败；没有方法通过 S1；
- M7 是生成后编辑参考；
- S1 仍是受控小规模机制实验，不构成最终论文主表；S2 未启动。

更早的骨盆控制、接触投影和终端补偿实验历史保留在 `PROJECT_MEMORY.md` 与 Git 历史中。本 README 以当前 M1–M7 规模实验主线为准。
