# ViMoGen 骨盆姿态控制研究代码

本仓库保存基于 [MotrixLab/ViMoGen](https://github.com/MotrixLab/ViMoGen) 开展的骨盆姿态控制、276 维动作表示一致性、评价与可视化研究代码。

当前主线新增独立协议 `vimogen_absolute_mean_pelvis_v4_anatomical_local`：它在 v3 完整前向运动学和末端安全融合基础上，使用冻结的项目专用 LASI/RASI/LPSI/RPSI 标志，并提供局部骨盆主导与防作弊审计。v3 仍是只读历史基线。

## ViMoGen 骨盆—接触时间一致性投影 v0.2（2026-09-04）

### 本次运行目的

本轮在冻结 v0.1 代码和 `attempt_08` 结果之外建立独立分支，协议为
`vimogen_pelvis_contact_flow_projection_v0_2_temporal_contact`，结果根目录为
`results/phase8/pelvis_contact_flow_projection_v0_2/`。目标只有两个：严格复现冻结 v3.0.1 的 M0，并在脚跟/脚尖位置约束上增加冻结连续接触帧对的三维位移约束，降低脚滑和抬脚。本轮不训练 ViMoGen、不优化初始噪声，也不加入重心、躯干或头部投影约束。

### 实验内容与固定条件

固定 sample34122、seed0、50 步、BF16、`sample_v1` 样本级初始噪声、原始双样本清单和冻结 v3.0.1 的 SMPL-X/接触贴片/地面高度。M0 复现门依次要求双样本批大小 2、单样本批大小 1，必要时再用冻结提交 `46a1b04` 重放；正式投影默认 `allow_m0_mismatch=false`。窗口前后各读取 1 帧固定上下文，只有窗口帧可修改。

### 具体实现

- `sampling/pelvis_contact_flow_projection_v0_1.py` 保持 v0.1 接口兼容，并加入 v0.2 时间接触协议、脚跟/脚尖位移残差、平足权重 1.0、一般接触权重 0.25、默认位移权重 `1e6`、1 mm/帧门和冻结上下文列；保留原有信赖域、非线性重线性化、有限值检查和逐分量回溯。
- `train_eval_vimogen.py` 的正式基线固定为 `official_pre_cast → authority_project → frozen physical M0`，并保存 `raw`、`official_pre_cast`、`official` 及权威重建产物。允许不匹配时运行记录标记为 `DIAGNOSTIC_INELIGIBLE`。
- 新增 `sampling/pelvis_contact_flow_projection_v0_2.py`、v0.2 运行器、M0 审计入口和冻结接触评价层。评价使用协议冻结的接触掩码、地面高度和窗口边界帧对；`NOT_EVALUABLE` 不计为正式通过。

### 测试与运行命令

静态 Python 编译和 `git diff --check` 已通过。服务器专项回归在最近三项审计测试加入前已通过 `21 passed`；新增测试已完成静态编译，但服务器复跑因连接不可用而待执行。本机没有 PyTorch，动态测试必须在服务器执行。正式入口为：

```text
python scripts/run_pelvis_contact_flow_projection_v0_2.py --metric kinematic_temporal --side left --target-delta-deg 2
python scripts/audit_pelvis_contact_m0_replay_v0_2.py --frozen-protocol <protocol> --run dual=<run>=1 --run singleton=<run>=0 --output <audit.json>
python scripts/evaluate_pelvis_contact_flow_projection_v0_1.py --run-root <run> --protocol-root <protocol>
```

### M0 复现结果与停止状态

当前服务器已完成两次无投影重放：

- 双样本批大小 2：`results/phase8/pelvis_contact_flow_projection_v0_2/m0_audit/dual_batch2/left/kinematic_temporal/dose_+2deg/attempt_01/`；
- 单样本批大小 1：`results/phase8/pelvis_contact_flow_projection_v0_2/m0_audit/singleton_batch1/left/kinematic_temporal/dose_+2deg/attempt_01/`。

两次重放的 `official_pre_cast → authority_project` 直接姿态最大差均为约 `1.8884e-2`，超过冻结门 `2e-3`；当前代码和批大小不是主要差异来源。审计文件为 `results/phase8/pelvis_contact_flow_projection_v0_2/m0_replay_audit.json`，总体状态为 `FAIL`。服务器连接随后不可用，冻结提交 `46a1b04` 的第三次受控重放尚未执行，因此阶段 A 尚未完成，阶段 B 端点投影和阶段 C–E 正式采样均按停止门未执行。

### 结果说明、不能说明什么

已验证的是：v0.2 的时间接触残差和严格评价边界已实现，当前双样本/单样本重放均能稳定产生可审计的 M0 差异。该差异说明当前环境或算子路径尚未达到冻结 M0 的逐样本复现要求；它不能说明时间投影几何不可行，也不能支持任何 +2°/+5°/+10° 正式效果结论。没有通过 `M0_PAIRING_PASS` 时，任何允许漂移的候选都只能作为诊断，不能算正式结果。

### 停止原因与下一步分流

先恢复服务器连接并使用冻结提交 `46a1b04`、原始双样本配置完成第三次重放；保存逐阶段、逐通道和逐帧差异以及样本级噪声哈希、有效帧掩码、检查点/均值/标准差/采样调度哈希。若第三次仍失败，停止正式采样并将问题归类为环境/算子复现阻塞；若通过，才按冻结端点可行性 → 左侧 +2° → 右侧和消融/高剂量的顺序继续。任何 v0.2.1 躯干安全包络或支撑关系约束都必须另建协议，不能覆盖本轮结果。

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

### v1.3 / v2 引导候选与直接 +10° 对照

为判断“旧的生成过程引导是否比直接旋转更自然”，本轮在 sample94/seed0/+10° 上统一复核 M0、仅根旋转、v1.3 分层根—脊柱引导、v2 源噪声引导和当前诊断补偿。每个历史候选严格配对自己的归档 M0；跨版本比较优先使用相对自身 M0 的倍率，避免把基线差异误算成候选效果。

下表显式保留 M0 行。括号内 `ΔM0` 表示候选减去它自己的配对 M0；负数只表示该单项数值下降，不代表整体接触已经通过。v1.3 与当前 M0 可视为同一基线，v2 必须配对它自己的归档 M0。

| 动作/基线 | 配对 M0 | 剂量 P95 误差 | 躯干变化 P95 | 关节加速度 P95（ΔM0） | 左足滑 P95（ΔM0） | 左离地 / 穿地 P95（ΔM0） | 重心代理位移 P95 |
|---|---|---:|---:|---:|---:|---:|---:|
| **M0（当前/v1.3）** | 自身 | — | `0°` | `7.241 mm/帧²（0）` | `29.155 mm/帧（0）` | `22.523 / 0 mm（0 / 0）` | `0 mm` |
| 仅根旋转 | 当前 M0 | `0°` | `10.00°` | `7.396（+0.156）` | `28.370（−0.785）` | `80.172 / 18.724（+57.649 / +18.724）` | `41.2 mm` |
| v1.3 引导 | v1.3 M0 | `0.126°` | `0.439°` | `7.439（+0.199）` | `26.763（−2.393）` | `55.156 / 13.513（+32.633 / +13.513）` | `13.3 mm` |
| **M0（v2）** | 自身 | — | `0°` | `7.460 mm/帧²（0）` | `29.368 mm/帧（0）` | `22.284 / 0 mm（0 / 0）` | `0 mm` |
| v2 源噪声 | v2 M0 | `1.449°` | `7.59°` | `9.718（+2.257）` | `45.591（+16.222）` | `41.220 / 11.472（+18.936 / +11.472）` | `234.3 mm` |
| 当前诊断补偿 | 当前 M0 | `0°` | `15.68°` | `81.621（+74.380）` | `107.301（+78.146）` | `157.617 / 0（+135.094 / 0）` | `91.1 mm` |

主要结论是：v1.3 明显优于直接 +10° 和当前诊断补偿，它既实现剂量，也基本保持躯干和原动作的时间平滑，是目前最合适的名义动作；但其双脚垂向接触仍未严格通过。v2 的结果是混合的：右脚离地等局部指标改善，但躯干、左足滑、抖动和整体漂移明显变差，不能判为整体更自然。当前诊断补偿在躯干、足部和时间指标上最差，不应继续作为下一阶段起点。

v2 的 M0 与当前/v1.3 M0 不是逐值相同，权威化后最大通道差约 `0.01481`、均方根差约 `0.00110`。归档确认两条路径的 seed、派生 seed、噪声键和噪声 SHA256 完全一致，所以这不是随机种子变化。现有证据把差异定位到生成路径和批组成：v1.3/current 来自双样本批的正式 BF16 采样，v2 来自单样本可微/正式重放；50 步中的数值差异会累积。尚未做受控的 batch1/batch2 交叉实验，因此不能把它进一步武断归结为“只由批大小导致”。

- [四栏正常速度视频](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/guided_comparison_v1/videos/sample94_M0_root_only_v1_3_v2.mp4)
- [四栏慢放视频](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/guided_comparison_v1/videos/sample94_M0_root_only_v1_3_v2_slow.mp4)
- [完整对照说明](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/guided_comparison_v1/README.md)
- [指标表](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/guided_comparison_v1/comparison.csv)
- [严格 JSON](results/phase8/pelvis_contact_walk_diagnostic/v3_1_walk_diagnostic/sample_94/dose_+10deg/attempt_01/guided_comparison_v1/comparison.json)

下一步保持协议和阈值不变，但调整技术起点：以 v1.3 引导候选作为名义动作，只在稳定接触期加入最小的下肢/根平移修正；每一步用可行性保持线搜索同时检查接触、躯干和时间平滑，不再从当前阶段 1 补偿候选继续堆叠。重心只在稳定支撑期作为软诊断/软约束。先在 sample94 检查完整步态，再回到 sample34122 完成正式双脚窗口，v3.2 继续锁定。

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

## ViMoGen Pelvis-Contact Sampling Projection v0.1：本次执行记录

### 目的

按照冻结方案验证一种独立的、采样过程内的骨盆接触投影：在不修改旧 v1.3、v2、v2.1、v3 或 v3.0.1 结果的前提下，使用 sample34122、seed0、左脚窗口和 `+2°` 剂量，检查骨盆剂量、稳定足跟/脚尖接触、穿地约束以及时间平滑是否能同时成立。方案规定首轮失败即停止后续剂量和方法扩展。

### 内容与实现

- 新协议名为 `vimogen_pelvis_contact_flow_projection_v0_1`，结果根目录为 `results/phase8/pelvis_contact_flow_projection_v0_1/`；投影只编辑根平移、根旋转、spine1–3、双侧髋/膝/踝/脚，接受更新后通过 SMPL-X/FK 权威重建 276D 表示和速度。
- 实现了 `x0_hat=x_sigma-sigma v`、速度重组、SO(3) 目标、冻结接触贴片、活动穿地等式近似、Euclidean 与 `kinematic_temporal` metric、最多 5 次重线性化、范数信赖域和固定回溯序列。SMPL-X/KKT 投影在 FP32 中执行；最终 clean endpoint 再做同一约束投影，避免末端积分残差重新破坏接触。
- 服务器运行器冻结了协议、配置、模型、检查点、样本噪声、调度器 sigma 序列和输入快照。由于当前服务器采样器与冻结 v3.0.1 M0 的重放存在直接通道最大约 `0.1583` 的数值漂移，本次明确使用“允许漂移”的探索分支：投影锚定当前重放 M0，冻结 M0 仍作为独立对照并记录 `MISMATCH_ALLOWED`，不把该运行标记为严格复现。

### 结果

- 服务器运行目录：`/root/autodl-tmp/vimogen_clean/results/phase8/pelvis_contact_flow_projection_v0_1/pilot_sample34122/left/kinematic_temporal/dose_+2deg/attempt_08/`；生成耗时约 `79.3 s`。专项测试与兼容回归合计 `36 passed`。
- 左窗口帧 `14–25` 的剂量控制通过：相对当前重放 M0 的剂量均值 `1.780°`、MAE `0.373°`、P95 `0.765°`；相对冻结 M0 的剂量均值 `1.731°`、MAE `0.406°`。表示一致性和有限值检查通过。
- 接触门未通过：左脚窗口评价为 `FAIL`，足滑 P95 约 `41.1 mm/帧`，离地 P95 约 `20.4 mm`；穿地 P95 为 `0`，但滑动和离地仍超配对 M0 阈值。整体严格状态为 `PRIMARY_FAIL_OR_NOT_EVALUABLE`。
- 按停止门，本次未继续右脚窗口、Euclidean metric、`+5°/+10°` 或视频扩展；之前的失败 attempt_01/02/03/05/06 均保留，未覆盖。该结果只说明“当前服务器重放漂移和求解器预算下，左窗口 +2° 未通过”，不能宣称几何上不可达，也不能作为严格 v0.1 成功证据。
