# ViMoGen 骨盆姿态控制研究代码

本仓库保存基于 [MotrixLab/ViMoGen](https://github.com/MotrixLab/ViMoGen) 开展的骨盆姿态控制、276 维动作表示一致性、评价与可视化研究代码。

当前主线新增独立协议 `vimogen_absolute_mean_pelvis_v4_anatomical_local`：它在 v3 完整前向运动学和末端安全融合基础上，使用冻结的项目专用 LASI/RASI/LPSI/RPSI 标志，并提供局部骨盆主导与防作弊审计。v3 仍是只读历史基线。

主要提交顺序：`a052bd0` 归档保护 → `11e8fe0` 表示/评价基线 → `a2cb7e5` 脚本/测试 → `9bf94af` 解剖几何 → `55bc0d0` v4 引导/评价/标定 → `592cd30` 局部主导安全项 → `7544acc` 占比统计修正 → `65ee191` 视频标记 → `00d5e66` 训练入口 → `5dcd795` 配置边界修正。

## 存档原则

- 使用分层 Git 提交记录每一阶段的实现和验证变化。
- 不提交模型权重、SMPL/SMPL-X 受许可模型、数据集、实验视频或大体积结果。
- 不提交服务器地址、密码、令牌、私钥或本机连接脚本。
- `results/phase6/absolute_mean_pelvis_v3/` 仅选择性保存冻结协议与小型清单，不保存生成结果。

## 上游与许可

本仓库不是 ViMoGen 官方仓库。上游代码及其模型、数据和第三方依赖仍受各自条款约束；本仓库只记录本研究中的新增或修改内容，不对上游资产授予额外许可。

## sample94 完整步行诊断

本分支新增 `v3_1_walk_diagnostic`，使用 sample94 的完整 100 帧步行动作观察骨盆 +10°、全身姿态与足部自然度。该阶段固定为 `diagnostic_only=true`、`eligible=false`、`can_unlock_v3_2=false`；正式选定动作仍回退 M0，诊断候选单独保存。sample94 左脚只有 5 帧平足证据、右脚没有平足证据，因此它只负责直观诊断，不能替代 sample34122 的严格双脚接触门。

服务器 `attempt_01` 完成于源提交 `433319d`，100 帧求解耗时 `54.07 s`。接触阶段 RMS 为 `0.861 mm`，通过 1 mm 内部门；躯干阶段把接触改善到 `1.236 mm`，但仍超门，因此阶段保护恢复到接触阶段候选。最终状态为 `DIAGNOSTIC_COMPLETED`，求解状态为 `INFEASIBLE_WITHIN_BUDGET`。

自然度对照如下。速度单位换算为 mm/帧，加速度为 mm/帧²：

| 指标 | M0 | 诊断候选 | 结果 |
|---|---:|---:|---|
| 骨盆剂量 MAE / P95 | — | `0° / 0°` | PASS |
| 根速度 P95 | `10.79` | `46.97` | 约 `4.35×`，仅报告 |
| 平均关节速度 P95 | `19.82` | `64.58` | 约 `3.26×`，仅报告 |
| 根加速度 P95 | `5.74` | `54.85` | 约 `9.56×`，仅报告 |
| 平均关节加速度 P95 | `7.24` | `81.62` | 约 `11.27×`，仅报告 |
| 根轨迹长度 | `0.659 m` | `1.445 m` | 约 `2.19×`，仅报告 |
| 左脚足滑均值 / P95 | `23.58 / 29.16` | `53.19 / 107.30` | FAIL |
| 左脚离地均值 / P95 | `14.69 / 22.52 mm` | `64.80 / 157.62 mm` | FAIL |
| 右脚足滑均值 / P95 | `24.53 / 25.21` | `26.22 / 28.37` | NOT_EVALUABLE（仅 2 个帧对） |
| 右脚离地均值 / P95 | `17.84 / 23.85 mm` | `104.65 / 182.62 mm` | FAIL |
| 左右脚穿地 | `0` | `0` | PASS |
| 躯干方向变化 P95 | `0°` | `15.68°` | FAIL |
| 骨盆-颈部 / 头部变化 P95 | `0° / 0°` | `14.15° / 14.86°` | FAIL |
| 骨盆相对支撑漂移 P95 | `0` | `75.73 mm` | FAIL |
| 水平朝向变化 P95 | `0°` | `0.020°` | PASS |

### 仅根旋转复核与躯干/重心诊断

为确认“接触补偿本身是否造成抖动”，本轮又单独构造了一个仅改变根旋转的候选：保持 M0 的 `body_pose`、根平移和所有派生通道，只按冻结协议构造 `+10°` 根旋转并重新权威化。服务器专项测试为 `14 passed`，候选形状为 `100×276` 且全部有限。

| 指标（sample94，+10°） | M0 | 仅根旋转 | 诊断补偿 | 解释 |
|---|---:|---:|---:|---|
| 根速度 P95 | `10.79 mm/帧` | `10.79` | `46.97` | 仅根旋转不增加根部抖动 |
| 平均关节加速度 P95 | `7.24 mm/帧²` | `7.40` | `81.62` | 额外补偿显著放大抖动 |
| 左脚足滑均值 / P95 | `23.58 / 29.16` | `24.18 / 28.37` | `53.19 / 107.30` | 仅根旋转基本保持原有水平 |
| 左脚离地 P95 | `22.52 mm` | `80.17 mm` | `157.62 mm` | 根旋转本身会改变脚底高度 |
| 左脚穿地 P95 | `0` | `18.72 mm` | `0` | 仅根旋转存在明显穿地 |
| 躯干方向变化 P95 | `0°` | `10.00°` | `15.68°` | 两者都没有保持躯干直立 |
| 骨盆-颈部 / 头部变化 P95 | `0° / 0°` | `10.00° / 10.00°` | `14.15° / 14.86°` | 前倾是全身姿态问题 |

重心部分采用完整 SMPL-X 网格顶点几何中心作为可复现的诊断代理，不冒充物理质量模型；支撑区域为冻结 M0 平足接触帧的完整足底贴片凸包。左脚只有 5 个有效稳定帧，右脚没有平足证据，因此该指标只作诊断，不进入正式门：

- M0 重心代理在这 5 帧相对足底支撑多边形的内部比例为 `0%`，说明步行动作不能用“所有帧重心必须落在脚中心”解释。
- 仅根旋转的重心水平位移 P95 为 `41.2 mm`，相对冻结 M0 支撑的有符号边界距离 P95 为 `−67.3 mm`。
- 诊断补偿的对应数值为 `91.1 mm` 和 `−107.3 mm`，整体平衡代理反而更差。

因此，当前证据支持：接触补偿增加了抖动，但没有修复身体前倾；接触几何与躯干/全身平衡必须分开处理。重心约束值得加入下一轮，但应先作为稳定支撑帧上的软诊断/软约束，并使用接触置信度渐变，不能直接施加全序列硬约束。

三栏视频依次显示 M0、仅改变根旋转、诊断补偿。固定相机下，诊断补偿在大多数帧仍接近“仅根旋转”，整体前倾没有恢复，同时根平移和时间变化显著放大。这与运行记录一致：能恢复躯干的阶段 2 候选因接触超出 1 mm 而被整体回退。

- [正常速度三栏视频](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/videos/sample94_walk_M0_root_only_compensated.mp4)
- [慢放三栏视频](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/videos/sample94_walk_M0_root_only_compensated_slow.mp4)
- [足部局部视频](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/videos/sample94_walk_foot_local.mp4)
- [自然度对照表](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/evaluation/naturalness_comparison.csv)
- [完整评价说明](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/evaluation/README.md)
- [仅根旋转评价说明](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/evaluation_root_only_com_v2/README.md)
- [仅根旋转指标](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/evaluation_root_only_com_v2/metrics.json)
- [仅根旋转逐帧重心](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/evaluation_root_only_com_v2/com_support_per_frame.csv)
- [诊断补偿（含重心指标）](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/evaluation_compensated_com_v2/metrics.json)
- [诊断补偿逐帧重心](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/evaluation_compensated_com_v2/com_support_per_frame.csv)

下一步保持协议和阈值不变，改造阶段 2 为可行性保持线搜索：只接受不破坏接触门的躯干更新，并额外保存“回退前候选”用于四栏诊断。目标是从当前 `1.236 mm` 向 1 mm 内推进，而不是放宽接触阈值。随后先在 sample94 观察步态连续性，再回到 sample34122 完成正式双脚接触验证。

## 骨盆接触补偿 v3.0.1：本轮执行报告

### 目的

本轮不是继续增加源噪声损失，而是检验一个更基础的问题：在 ViMoGen 冻结的输出姿态空间中，固定的骨盆剂量、躯干保持和足部接触是否能同时成立。`sample34122 / seed0 / +10°` 是严格接触案例；`sample94` 保留为后续完整步行序列的可视化案例。只要严格案例的 v3.1 窗口门未通过，就不进入 v3.2 全序列补偿，也不训练残差适配器。

### 冻结协议与执行计划

- 从提交 `805a6a5` 建立分支 `codex/pelvis-contact-compensation-v3-0-1`，保留旧 v3 结果，不覆盖历史目录。
- 协议修订为 `vimogen_pelvis_contact_compensation_v3_0_1`。正剂量继续采用 v1.3 语义，根旋转由冻结的 M0 矢状面直接构造；接触阈值、离散 M0 掩码、足底贴片索引和信赖域均固定。
- v3.1 只解 sample34122 左右最长稳定窗口，固定使用 `+2° → +5° → +10°` 延续路径。优化顺序为接触/穿地 → 躯干与整体直立 → 姿态和时间平滑。
- 后续阶段增加了硬保护：若躯干优化使已满足的接触约束失效，恢复上一阶段候选，并将本剂量标记为不可行；不可行候选与 M0 回退严格分开。
- v3.2 只有在左右窗口均通过后才允许运行；sample94 的全序列补偿、采样中投影、物理模块和残差适配器均不在本轮范围内。

### 代码与验证

主要实现位于 `sampling/pelvis_contact_compensation_v3.py`、`evaluation/pelvis_contact_compensation_v3.py`、`scripts/run_pelvis_contact_compensation_v3.py` 和协议冻结脚本。新增整体直立指标（骨盆-颈部、骨盆-头部、骨盆相对支撑中心漂移）、连续接触置信度、精确旋转/平移范数投影、固定延续初始化和分层约束保护。

服务器运行使用 PyTorch `2.7.0+cu128`、RTX 4080 SUPER；冻结协议 SHA256 为 `6884e256e23f6c3d268c3e04c6ed6a22c565e9eccf676abc3386dccd181b937a`，SMPL-X 模型目录 SHA256 为 `c4721f0dbbc741438cac9961efea31d832aa212cf65e34a3f3be82706af55896`，足底贴片 SHA256 为 `1f76af485fa969fc4d813bd61415b69ac8baf5e8cce715ecdd170f9efd4a87ae`。

新增专项测试为 `11 passed`。原服务器工作区的既有完整回归为 `246 passed in 51.29 s`；独立动态树只包含本分支提交的研究代码，缺少若干历史工作区未归档的兼容模块，因此在该树直接收集全部旧测试会出现导入错误，这不改变 v3 专项测试结果。

### 结果

运行记录保存在服务器目录：

`/root/autodl-tmp/vimogen_pelvis_contact_v3_0_1_results/v3_1_window_feasibility/sample_34122/dose_+10deg/attempt_02/`

运行状态为 `STOP_V3_2`，`v3_2_allowed=false`。

- 左脚窗口为帧 `14–25`（稳定帧 8）。+10° 接触 RMS 为 `1.029 mm`，略高于 `1 mm` 门；因此左窗口未通过，后续阶段未执行。
- 右脚窗口为帧 `78–90`（稳定帧 11）。接触阶段达到 `0.831 mm`，但躯干阶段的候选将接触 RMS 拉高到约 `7.0 mm`；硬保护检测到回归并恢复接触阶段候选，记录为 `preserved_previous_stage=false`、`restored_to_previous_stage=true`，该剂量仍不可行。
- 对两侧最佳不可行候选的严格配对评价均为 `FAIL`。骨盆剂量本身精确为 `+10°`，但整体躯干/直立指标仍明显超门：左侧骨盆-颈部 P95 `12.94°`、骨盆-头部 P95 `15.25°`、支撑漂移 P95 `35.8 mm`；右侧分别为 `12.85°`、`12.83°`、`234.0 mm`。脚部一般接触、滑动或离地门也存在失败项。由于这些是最佳不可行候选，正式 `selected_motion.pt` 按协议回退为 M0，不冒充补偿成功。
- 诊断视频（窗口候选，不是 v3.2 成功结果）：[左脚 M0/最佳不可行候选](results/phase8/pelvis_contact_compensation_v3_0_1/v3_1_window_feasibility/sample_34122/dose_+10deg/attempt_02/left_best_infeasible_M0_vs_candidate.mp4)；[右脚 M0/最佳不可行候选](results/phase8/pelvis_contact_compensation_v3_0_1/v3_1_window_feasibility/sample_34122/dose_+10deg/attempt_02/right_best_infeasible_M0_vs_candidate.mp4)。

### 结论

本轮完成了协议冻结、符号和根旋转构造、接触窗口选择、范数信赖域、固定延续路径、严格 JSON/哈希记录、整体直立评价以及失败回退保护。没有完成 v3.1 可行性门，因此 v3.2 全序列补偿尚未执行，sample94 步行样本也没有产生新的补偿候选。

当前负结果的准确含义是“在冻结协议和当前求解器预算内未通过”，还不能单独宣称几何上不可达。右脚结果明确暴露了原实现的层级约束问题；修复后接触可以保持，但躯干目标仍无法在同一信赖域内满足。左脚则仍卡在约 0.03 mm 的接触门边缘。

### 下一步

1. 保持协议、阈值、案例和 M0 完全不变，加入带可行性线搜索的阶段锁定求解：每个躯干/平滑候选只有在接触、方向和穿地约束不超过上一阶段门时才接受；必要时改用约束 SQP 或接触约束的零空间步。
2. 对左右窗口分别输出逐点接触残差、雅可比秩和信赖域活跃边界，区分“优化预算不足”和“局部几何冲突”；优先把左 +10° 接触压到 1 mm 以内，并验证右侧在保持接触时能否降低整体直立残差。
3. 只有左右窗口在 `+2°、+5°、+10°` 均通过，才重跑 v3.1 完整记录并解锁 v3.2；随后才处理 sample94 全序列步行可视化。若仍失败，再基于冲突约束另行版本化 v3.3 方案，不修改本协议回溯结果。
