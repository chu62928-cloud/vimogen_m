# ViMoGen骨盆姿态控制 Project Memory

## 2026-09-01：骨盆接触补偿 v3.0–v3.2 已实现并在 v3.1 停止

- IMPLEMENTED：从提交 `805a6a5` 建立独立工作树与分支 `codex/pelvis-contact-compensation-v3`，远端最新提交为 `38f5e43`。新增 `evaluation/pelvis_contact_compensation_v3.py`、`sampling/pelvis_contact_compensation_v3.py`、协议冻结/运行/评价/渲染脚本及专项测试；ViMoGen 与 v1.3/v2.1 代码未覆盖。
- FROZEN：协议名为 `vimogen_pelvis_contact_compensation_v3`，结果根目录为 `results/phase8/pelvis_contact_compensation_v3/`。正剂量采用 `delta=M0_pitch-candidate_pitch`；目标根旋转为 `Rot(M0_right,-delta)@R0`；接触阈值为高度 25 mm、速度 30 mm/帧、平足高差 20 mm，首帧速度无效，最小证据 3 帧/连续帧对。足底贴片使用中性 SMPL-X 网格并写入哈希。
- VERIFIED：服务器冻结协议位于 `/root/autodl-tmp/vimogen_pelvis_contact_v3_results/protocol_v3_0_final/`。sample94 左脚平足 5 帧、右脚 0 帧（右脚 `NOT_EVALUABLE`）；sample34122 左右平足分别 8/11 帧，最长稳定段为左 `18–21`、右 `82–86`，均按冻结离散 M0 掩码选择。
- VERIFIED：服务器 v3 专项测试为 `8 passed in 2.64 s`；原服务器工作区只读完整回归为 `246 passed in 51.41 s`。从干净 `805a6a5` 提取树收集旧测试会因服务器脏工作区中的历史兼容模块未随提交归档而导入失败，该问题不属于 v3 代码；完整回归已在原工作区完成。
- NEGATIVE_RESULT/STOP：sample34122、seed0、+10° 的 v3.1 左右窗口均在名义信赖域 30°/5 cm 内 `INFEASIBLE_WITHIN_BUDGET`；固定延续路径 `+2°→+5°→+10°` 后接触位置 RMS 分别约 `4.481 mm`（左）与 `4.513 mm`（右），超过 1 mm 接触门，故运行记录 `attempt_07/run_record.json` 状态为 `STOP_V3_2`、`v3_2_allowed=false`。脚本已强制 v3.2 检查该门，当前尝试会明确拒绝，不生成冒充成功的候选。
- VERIFIED：动态运行树为 `/root/autodl-tmp/vimogen_pelvis_contact_v3`，模型/数据仅通过只读链接复用；运行记录包含协议哈希、足底贴片哈希、模型哈希和源提交 `d4983d4c574a3a84adc5c963457a82cd773a308f`。失败尝试目录保留，未覆盖；v3.2 与视频渲染因 v3.1 停止门未执行。
- DECISION：本轮停止在 v3.1 可达性否决点，不训练残差适配器，不进入 v3.3/v3.4、采样中投影或物理模块。若要继续，需先针对左右窗口的具体足底位置冲突另立诊断/算法变更并重新冻结协议，不能修改当前 v3 阈值后宣称通过。

## 2026-09-01：根—躯干相对角 v2.1 单例验证完成

- VERIFIED：已从冻结的 `codex/relative-root-forward-v2` 切出 `codex/relative-root-trunk-v2-1`。v2 的代码和历史产物保持冻结；本分支最终对 `sampling/relative_root_forward_guidance_v2.py` 无差异。v2.1 新增独立几何、配对自然性评价、源噪声优化器、服务器运行器、最终评价器、视频渲染器和测试。
- IMPLEMENTED：新增协议 `vimogen_relative_root_trunk_v2_1_minimal_source_noise`。目标是每帧冻结 M0 水平朝向，在 M0 矢状面内计算“spine1→neck 躯干轴到直接根前向轴”的绕 M0 右轴有符号相对角；优化损失只包含相对角误差，躯干和脚部只作外部门。
- IMPLEMENTED：配对自然性评价统一使用直接 `body_pose/root_rotation/root_translation` 权威化和 FK/SMPL-X 网格，不使用速度积分。M0 固定接触集合排除首帧速度，一般接触用于滑动/离地/穿地，平足只用于脚尖主导；均值和 P95 使用 `max(M0×5%,1 mm)` 容差，证据不足返回 `NOT_EVALUABLE`。
- VERIFIED：服务器专项几何/接触测试 `9 passed`；服务器完整回归 `246 passed in 47.19 s`。所有动态测试、模型运行、评价和 EGL 视频渲染均通过 `connect_server.py` 在服务器完成，本地仅作静态编译、哈希核对和报告整理。
- VERIFIED/NEGATIVE_RESULT：固定单例 `sample94/seed0/+10°`、50 步、120 次上限、步长 RMS `0.01`、源噪声 RMS≤`1.0`、8 级固定锚点缩小最终运行于 `attempt_03`，耗时 `271.220 s`，实际 4 次迭代，状态 `INFEASIBLE_WITHIN_BUDGET`，明确选择 `m0_fallback`。最佳不可行候选实际相对角剂量约 `+2.046°`，相对角 P95 误差约 `8.642°`；最终选定 M0 的相对角 MAE/P95 均为 `10.000°`，因此内部相对角门失败。
- VERIFIED：v2.1 最终外部门中，躯干三维方向 P95 `0.028°`、`q_rigid=0`、水平朝向 P95 `0.020°`、尾部额外旋转/俯仰跳变约为零、左右脚滑动/离地/穿地相对 M0 均 `PASS`；脚尖门为 `NOT_EVALUABLE`（右脚平足证据不足）。这些脚部门结果来自 M0 回退，不能解释为 v2.1 成功保持了生成候选自然性；总状态为 `FAIL`。
- VERIFIED：冻结 v2 `attempt_10` 的同一直接权威化配对重评仍失败：相对角 MAE/P95 `13.384°/14.541°`，实际相对角剂量 `-3.384°`，躯干三维方向 P95 `7.593°`、`q_rigid=0.645`；水平朝向 P95 `1.557°`、尾部额外旋转 `0.558°`、俯仰跳变 `0.660°`。左脚滑动均值由 `24.408` 增至 `30.027 mm/帧`并超过容差，旧自然性回归未被评价修正消除。
- VERIFIED：最终交付目录为 `results/phase7/relative_root_trunk_v2_1/diagnostics/`，含 `USER_REPORT.md`、`TEST_REPORT.md`、`attempt_03_final/gates.json`、`naturalness.json`、逐帧/逐脚 CSV、相对角曲线、回放记录和三栏/脚部局部 MP4。三栏视频和脚部视频均为 H.264、1920×1080、20 fps、100 帧、5 秒；哈希分别为 `b3ec722b9cdda8d011ecb0ee125c47359ec2fe66dca16c3e56aae28a678d49c4` 和 `152e97c87be749169f2a03548fcf1ff0f46b1383137786924c50b8e5d625c9ce`。
- DECISION：按停止条件停止纯源噪声相对角路线，不扩展其他七种组合、样本、种子或正式 MBench，不向本协议加入脚部/躯干优化损失。下一任务转向学习式全身适配器，或先做接触感知的全序列运动学/物理投影，再决定是否接回生成器。相对角几何的通过测试仅证明表示的几何性质，不证明本单例可达。
- COMMITTED：v2.1 代码与项目记忆已提交为 `295be19`（`feat: validate root-trunk relative-angle v2.1`），并已推送到远端 `codex/relative-root-trunk-v2-1`。工作区其余未跟踪文件为用户既有文件，未加入提交。

## 2026-08-31：v2 统一评价、正确性修复与可达性诊断完成

- VERIFIED：当前分支为 `codex/relative-root-forward-v2`；本轮代码提交为 `2404343`，v1.3 仍保持冻结。工作区中用户既有的未跟踪文件未被加入提交、删除或覆盖。
- IMPLEMENTED：修复 `sampling/relative_root_forward_guidance_v2.py` 的已接受候选丢失问题；加入掩码均方根归一化的负梯度回退，并保留同一验证式回溯规则。新增统一门控、严格 JSON、接触证据、表示因果审计、子空间探针和统一报告入口。
- VERIFIED：服务器环境为 `/root/miniconda3/envs/mdm5090/bin/python`、PyTorch `2.7.0+cu128`、RTX 4080 SUPER；所有动态测试、采样、诊断和视频渲染均通过 `connect_server.py` 在 `/root/autodl-tmp/vimogen_clean` 执行。
- VERIFIED：最终服务器完整回归为 `237 passed in 48.81 s`；本地仅执行静态编译与 `git diff --check`，均通过。
- VERIFIED：固定纯 v2 单例为 sample94、seed0、+10°、120 次迭代上限；服务器 attempt_10 完成 `40` 次迭代，状态 `FEASIBLE`，耗时 `1625.718 s`，因连续无验证下降停止。统一评价中根俯仰 MAE `0.653901°`、完整前向 P95 `1.768239°`、根航向 P95 `1.556506°`、剂量符号正确；尾部两项通过，但躯干方向 P95 `7.503897°`、`q_rigid=0.649745` 和自然性外审失败。
- VERIFIED/NEGATIVE_RESULT：服务器停止门 attempt_03 通过，位级复现为真，梯度非零 `27594/27600`，峰值保留显存 `9282 MiB`，但该停止门和纯 v2 运行均不等价于完整 v2 成功。最终 `gates.json` 固定 `counts_as_v2_success=false`，阶段 E 按失败门停止。
- VERIFIED/DIAGNOSTIC_ORACLE：运动学诊断在 100/100 帧满足显式躯干和固定脚关节代理约束，但 SLSQP 成功返回 `0/100`，线搜索警告单独记录；源噪声子空间诊断包含 16 个方向、真实 50 步响应矩阵、线性 SLSQP 和四个比例复验，最佳真实根俯仰 P95 `7.926897°`，未达目标。两类结果均标记 `role=diagnostic_oracle`、`counts_as_v2_success=false`，没有混入纯 v2 门。
- VERIFIED：最终交付目录为 `results/phase7/relative_root_forward_v2/diagnostics/relative_root_forward_v2_unified_final/`，含 `USER_REPORT.md`、`TEST_REPORT.md`、`gates.json`、`naturalness.json`、`causal_audit.json`、运动学/子空间诊断原始产物及网格视频。视频为 H.264、1920×1080、20 fps、100 帧、5 秒，SHA256=`3dbb2acfea71f8b2d8f7db326296cce05581a8a9eef5c2772903a55ea0f99371`。
- DECISION：不扩展其他七种验证组合、不增加种子、不进入正式 MBench、不添加躯干/足部优化损失；保留本轮负结果与诊断证据，后续若推进应另建全身适配器或物理投影版本。

## 2026-08-31：v2 分支分离与输出链路修复（进行中）

- VERIFIED：GitHub 分支 `codex/relative-root-forward-v1-3` 已冻结在 `46a1b04`；独立分支 `codex/relative-root-forward-v2` 已从 `f61126e` 创建并推送，当前修复提交为 `c30fdf1`。未清理工作区中的既有未跟踪文件。
- VERIFIED：服务器停止门 `attempt_03` 通过。当前服务器实际显卡为 RTX 4080 SUPER；三路输出逐位一致，梯度有限且非零元素 `27594/27600`，峰值保留显存 `9282 MiB`，总耗时 `18.711 s`。
- VERIFIED：先行失败回归捕获了 v2 输出路由缺少可测试边界；修复后服务器专项测试为 `6 passed in 3.17 s`，完整回归为 `228 passed in 52.20 s`。
- IMPLEMENTED：v2 输出选择边界不再被后续 M0 默认赋值覆盖；不可行回退使用零噪声改变量；优化器记录实际 BF16 噪声、显式评价末次候选、固定锚点进行独立缩小搜索，并加入自适应下降、有效帧掩码和时间上限。
- FAILED/DIAGNOSED：单例 `attempt_07` 因缺少 `time` 导入失败，`attempt_08` 因 detached 增量切断梯度失败；两次失败目录保留，未作效果结论。
- SUPERSEDED：服务器 `attempt_09` 已完成；最终状态、指标和停止决策见下方“v2 单例根控制通过、自然性门失败”。

## 2026-08-31：v2 单例根控制通过、自然性门失败

- VERIFIED：服务器单例 `attempt_09` 已完成，运行 `929.288 s`，实际完成 `27` 次迭代后达到可行候选；根俯仰 MAE `0.6803°`、俯仰 P95 `1.7331°`、完整前向 P95 `1.9923°`、剂量符号正确，源噪声 RMS `0.05670`。归档动作与 M0 不同，证明输出覆盖修复已生效。
- VERIFIED：外部根控制审计结果位于服务器 `/root/autodl-tmp/vimogen_clean/results/phase7/relative_root_forward_v2/minimal_source_noise/seed_000/delta_+10deg/attempt_09/source_noise_external_evaluation.json`；尾部额外 SO(3) 最大跳变 `0.611°`、额外俯仰步长最大 `0.677°`，尾部门通过。
- VERIFIED/NEGATIVE_RESULT：新增外部自然性审计后，脚尖主导回归为假，但自然性门未通过：左脚接触期滑动均值由 `27.998` 增至 `42.196 mm/帧`，接触期离地高度均值由 `11.836` 增至 `14.054 mm`，均超过“5%或1 mm”容差；穿地未增加。服务器产物为 `/root/autodl-tmp/vimogen_clean/results/phase7/relative_root_forward_v2/minimal_source_noise/seed_000/delta_+10deg/attempt_09/source_noise_naturalness_evaluation.json`。该结果只支持“根控制成功、自然性失败”，不支持完整 v2 成功。
- VERIFIED：服务器最新完整回归为 `230 passed in 48.86 s`；专项 v2 测试为 `8 passed in 3.10 s`。自然性脚本首次因 batch 维输入失败，修正后审计通过运行并保留失败历史。
- VERIFIED：为便于结果审查，服务器生成并核验了 v2 单例骨架视频 `sample94_v2_single_skeleton_M0_+10.mp4`（1920×1080、20 fps、5 s、438290 bytes）；本地副本位于 `results/phase7/relative_root_forward_v2/minimal_source_noise/seed_000/delta_+10deg/videos/`。该视频右侧为 v2 `+10°` 候选，中间为 M0 复制，属于审查产物，不改变自然性门失败结论。
- VERIFIED：应用户要求，改用服务器 EGL 软件渲染生成网格审查视频 `sample94_v2_single_mesh_M0_+10.mp4`（H.264、1920×1080、20 fps、5 s、505612 bytes）；本地副本与骨架视频同目录。首次渲染的竖直轴翻转已修正并重新核验，当前画面人物方向正常；该视频仍只是审查产物。
- COMMITTED：独立 v2 分支远端当前包含 `c30fdf1`（输出路由与自适应源噪声优化）及 `e42aec7`（自然性外部审计）；v1.3 分支仍冻结在 `46a1b04`。
- DECISION：因单例自然性门失败，按冻结停止条件不运行另外 7 个组合、不扩展五种子、不进入正式 MBench、不加入足部损失；应保留该负结果并另行规划统一全身适配器或物理投影。

## 2026-08-30：根前向 v2 可微停止门与最小源噪声实现

- VERIFIED：已在分支 `codex/relative-root-forward-v1-3` 独立新增可微完整采样副本与停止门运行器；官方 v1.x 采样默认路径未修改。停止门固定 sample94、seed0、batch=1、BF16、官方 50 步、外层梯度检查点，要求 `G(z0)` 的 `raw/official_pre_cast/official` 与官方 M0 逐位一致，并验证仅根前向目标对 `z0` 的梯度有限且非零、峰值保留显存低于 `28672 MiB`。
- FAILED/DIAGNOSED：服务器首次 `attempt_01` 已完成官方 M0，随后在进入可微链前被新代码过严的参考动作通道数校验拦截；ViMoGen 官方接口只要求 `ref_motion` 与源噪声前两维 `[B,T]` 相同，不要求通道同为 276。已将新路径校验修正为与官方 `FlowSampler` 一致；该失败没有产生可微复现、梯度或峰值显存结论。
- EVIDENCE：失败记录位于服务器 `/root/autodl-tmp/vimogen_clean/results/phase7/relative_root_forward_v2/gates/differentiable_50step/seed_000/attempt_01/run_record.json`；服务器动态运行始终通过本地 `connect_server.py` 发起。
- VERIFIED：修正后的 `attempt_02` 已通过：三路输出逐位一致；梯度有限且 `27598/27600` 非零；前向/反向 `5.700/5.899 s`；峰值保留显存 `9282 MiB`，低于 `28672 MiB`。服务器产物为 `/root/autodl-tmp/vimogen_clean/results/phase7/relative_root_forward_v2/gates/differentiable_50step/seed_000/attempt_02/`。
- IMPLEMENTED：新增 v2 源噪声协议 `vimogen_relative_root_forward_v2_minimal_source_noise`、可微采样副本、端到端运行器和外部评估器；v1.x 默认路径保持不变，v2 只优化 `z0+delta_z`，硬约束为根俯仰 MAE≤1°、完整前向 P95≤2°、符号正确，最终不可行时安全返回 M0。
- DECISION：当前证据只支持“可微链路与 v2 接口可运行，有限预算下根前向完整约束未达标”的负结果；在没有新的优化预算或算法决定前，不扩大到五种子/正式 MBench，不添加躯干或足部损失。

## 2026-08-30：v2 最小源噪声端到端冒烟与步长修正

- VERIFIED：停止门 `attempt_02` 已通过服务器运行：`raw/official_pre_cast/official` 均逐位一致；梯度有限、非零元素 `27598/27600`；前向/反向 `5.700/5.899 s`；峰值保留显存 `9282 MiB`，低于 `28672 MiB`。
- IMPLEMENTED：新增 `sampling/relative_root_forward_guidance_v2.py` 和服务器运行器 `scripts/run_relative_root_forward_v2_minimal_source_noise.py`；`relative_root_forward.protocol=vimogen_relative_root_forward_v2_minimal_source_noise` 时只优化 `z=z0+delta_z`，第一层根约束可行性，第二层沿可行方向最小源噪声距离，最终一次 `authority_project`。
- DIAGNOSED：服务器 `seed0/+10°` 两次迭代冒烟完成但仍不可行，原因是实现把 `step_rms` 误作全张量 L2 步长，实际每步 RMS 缩小约 `sqrt(27600)`；已修正为 `step_l2=step_rms*sqrt(numel)`。该冒烟不作为效果结论。
- EVIDENCE：冒烟产物位于服务器 `/root/autodl-tmp/vimogen_clean/results/phase7/relative_root_forward_v2/minimal_source_noise/seed_000/delta_+10deg/attempt_01/`；服务器动态运行均通过 `connect_server.py`。
- DIAGNOSED：服务器 `seed0/+10°` 已完成 2、4、8、12 次迭代以及步长/软最大温度变体；4 次默认变体内部根俯仰 MAE 可到约 `0.60°`，但完整前向 P95 约 `7.94°`；12 次温度 20 仍不可行。由于没有同时满足硬约束，最终按协议返回 M0；独立外部审计确认最终 P95 约 `10.00°`、尾部安全通过，不能宣称控制成功。
- VERIFIED：完整服务器回归为 `225 passed in 27.87s`；协议已冻结到 `results/phase7/relative_root_forward_v2/protocol.json`，SHA256=`9fda91439cdbfa4a878159bed73ba26940e30e3409fd6b7e12c92c8a3704aa79`。若继续推进，应先明确是否允许高成本迭代/优化器搜索，或按研究分流转向 v2.1 冲突感知源噪声诊断。
- COMMITTED：本地分支提交 `f61126e`（`feat: implement minimal source-noise root-forward v2`），仅包含本轮 v2 代码、运行器、外部评估器、协议冻结脚本和 3 项边界测试；远端分支尚未推送。

### Immediate continuation

1. 若继续 v2，先在冻结协议下决定高成本优化预算/优化器；不得把当前有限预算负结果扩写为控制成功。
2. 只有新预算在 sample94/34122、seed0/42 的完整前向与自然性外部门同时通过，才扩大到五种子；否则保留负结果并转向统一全身适配器或物理投影。

## 2026-08-30：v1 系列 GitHub 存档完成，v2 待新任务实施

- VERIFIED：v1、v1.1、v1.2、v1.3 当前代码已存档到 GitHub 分支 `codex/relative-root-forward-v1-3`，远端提交为 `46a1b04ddfe10088a7472b0c67b82ce82fb343d3`。该提交新增 v1.2/v1.3 主模块、服务器运行器、严格评估、运动学可行性、脚部接触诊断及专项测试，并继承该分支历史中的 v1/v1.1 存档。
- VERIFIED：存档前确认本地候选文件与服务器 `/root/autodl-tmp/vimogen_clean` 副本的 SHA256 全部一致；服务器专项测试为 `18 passed in 1.80s`，完整回归为 `222 passed in 29.51s`。
- DECISION：v1.x 代码与历史结果从此冻结。下一阶段按已批准的“约束优先源噪声优化”方案新增 `vimogen_relative_root_forward_v2_minimal_source_noise` 和 `vimogen_relative_root_forward_v2_1_conflict_aware_source_noise`，不得修改 v1.x 协议语义。
- VERIFIED：v2 可微链路、停止门和最小源噪声接口已实现；停止门要求 `G(z0)` 与官方 M0 逐位一致，所有动态测试、显存分析和真实生成继续通过 `connect_server.py` 在服务器执行。

## 2026-08-29：根前向精简引导文献审查

- RESEARCH：完成一手论文、官方项目页与官方代码审查，记录在 `docs/research/relative_root_forward_simplified_guidance_review.md`。覆盖 D-Flow、DNO、DARTControl、ProgMoGen、OmniControl、PhysDiff、PHC 与任务优先全身控制。
- DIAGNOSIS：v1.3 的约束扩张源于控制层级错误：直接根/脊柱姿态手术没有定义全身响应。OmniControl 的消融也报告仅解析空间引导会造成其他关节不协调和足滑；学习式全身真实性分支或生成潜空间控制才是结构性解法。
- RECOMMENDATION：下一步优先做“最小源噪声根控制”的小规模可证伪实验：冻结 ViMoGen，只优化确定性完整采样链的源噪声，只给根前向目标，以最小源噪声改动保持内容；躯干和足接触只作未参与优化的外部检验。若仍失败，不继续添加局部关节损失，而转向统一的学习式全身适配器或 PHC/PhysDiff 物理投影。
- RESOURCE_GATE：DNO 官方编辑实验为 300 次优化、RTX 3090 约 3 分钟，DDIM-10/batch16 约 18 GB；当前 ViMoGen 约 13 亿参数且官方 50 步，必须先测 10/20/50 步的显存、耗时与质量，超过当前服务器资源即保留负结果，不隐式退回逐关节手术。
- ALTERNATIVE：若目标优先是工程交付而非验证生成先验，可采用事件门控、接触优先的分层局部投影器统一 heel/toe 接触、地面和最小姿态改动；该路线更稳健但仍是显式动画约束，不作为最精简的研究主线。
- PROJECT_BASELINE：补充一个项目特有的解析强基线。SMPL-X 根的直接子节点为 `left_hip/right_hip/spine1`；根从 `R0` 改为 `R*` 时令子节点局部旋转 `Lc*=R*^T R0 Lc`，可严格保持三个子树的世界朝向。再让根绕双髋中点旋转，可近似保持双腿位置；按冻结静止骨架计算，`10°` 时左右髋锚点各自仅约 `1.06 mm` 不对称残差。该方法不需要逐项躯干/脚部损失，应作为运动学可行性和 v1.3 强对照，但不替代“生成先验是否自发协调”的最小源噪声主实验。

## 2026-08-29：v1.3 脚尖接触回归诊断

- NEGATIVE_RESULT：用户在侧视视频中观察到 `+10°` 后脚底接触趋向脚尖。服务器上的 SMPL-X 网格级诊断已复现该问题；它不是单纯的相机视角错觉。以 M0 中原本接近平足且低速的接触帧为固定集合，`+5°` 有 `25/450` 帧（5.6%）变为脚尖主导，`+10°` 有 `83/450` 帧（18.4%）变为脚尖主导；`+10°` 在 20 个“样本×种子×左右脚”组合中有 6 个达到严格回归判据。
- DOSE_RESPONSE：在具有至少 3 个 M0 平足接触帧的组合上，脚跟相对脚尖的高度差变化中位数随剂量呈稳定单调关系：`-10°=-19.01 mm`、`-5°=-8.47 mm`、`+5°=+8.56 mm`、`+10°=+18.93 mm`；足部向脚尖方向的俯仰变化中位数分别为 `-7.98°/-3.57°/+3.60°/+7.97°`。
- ROOT_CAUSE：v1.3 只对 `spine1/2/3` 做躯干反向补偿，腿、踝和足部没有接触约束或反向补偿，因此根俯仰会沿骨架父子链近似刚性地传递到下肢。严格回归行中足部世界俯仰约变化 `7.55°–8.80°`，但踝与足部局部旋转变化中位数通常只有约 `0.1°–0.3°`，排除了“扩散产生了数度异常踝关节局部转动”这一主因。根平移会影响离地或穿透，但不能解释脚跟—脚尖相对高度差，因而不是脚尖支撑的根因。
- REPRO：新增红灯式诊断脚本 `scripts/diagnose_relative_root_forward_v1_3_foot_contact.py`。服务器产物为 `/root/autodl-tmp/vimogen_clean/results/phase7/relative_root_forward_v1_3/diagnostics/foot_contact/toe_pivot_5seed.json`，本地副本为 `results/phase7/relative_root_forward_v1_3/diagnostics/foot_contact/toe_pivot_5seed.json`；命令带 `--fail-on-regression` 时已按预期非零退出并报告 `toe_contact_regression_detected=true`、`flagged_rows=6`。
- VISUAL_CHECK：seed0/sample34122 的第 75 帧网格和骨架截图位于 `results/phase7/relative_root_forward_v1_3/diagnostics/foot_contact/frames/`，可见 `+10°` 面板的足部相对 M0 明显向脚尖方向倾斜，与网格指标一致。
- SCOPE：本轮只完成诊断，没有修改 v1.3 算法。v1.3 的根前向与躯干抑制数值门槛仍成立，但若验收包含自然足底接触，则当前方案不能判定为完整成功。下一阶段应单独设计接触感知的下肢补偿，并先做脱离扩散的运动学可行性测试。

## 2026-08-28：根前向引导 v1.3 五种子严格验证完成

- IMPLEMENTED：新增协议 `vimogen_relative_root_forward_v1_3_shadow_pose_hierarchical`。模型端状态与物理影子状态分离；`body_pose/root_rotation/root_translation` 是直接姿态权威，`J/dJ/dR/dT` 只在影子和最终输出边界由 FK/前向差分派生，不再写回采样器，也不接受引导梯度。可编辑直接通道仅为根旋转及 `spine1/2/3` 局部旋转。
- IMPLEMENTED：根俯仰、水平偏航和三节脊柱补偿采用迭代阻尼最小范数雅可比求解（最多 4 次闭环）；统一保留 11 次二分回溯、2° 相邻修正投影、根/躯干物理信赖域和最终一次完整权威化。v1.3 不构造混合量纲总损失，`motion_weight` 默认明确为 0；完整 276D 改变量只作诊断。
- FIXED：sigma 活动窗口采用 `eps` 容差比较，避免实际调度值因浮点舍入低于 `sigma_min` 而错误跳过最后一个活动步。修正后 seed0/sample34122 的 +5° 完整前向 P95 从约 `4.1°` 降至约 `0.97°`。
- VERIFIED：所有动态测试通过服务器 `connect_server.py`：v1/v1.2/v1.3 专项 `18 passed`，最终完整回归 `222 passed in 27.12s`。本机仅作语法检查；没有把本机缺少 PyTorch 当作动态验证环境。
- VERIFIED：运动学可行性测试在冻结 M0 上完成 `±5°/±10°`，四种剂量根/完整前向误差均为数值零，脊柱预算和时间预算通过，说明 v1.3 自由度与求解器本身可达。
- CALIBRATION：固定参数为 `residual_gain=1.0`、`max_step_deg=6°`、`sigma=[0.0662879,0.65]`、`heading_gain=0.75`、`max_heading_step=2°`、`trunk_gain=0.75`、`max_trunk_step=6°`。复用同一 `sample-noise-v1` 规则，分别生成 seed `0,42,464229750,1057660199,1386772747` 的 M0 与 `-10/-5/+5/+10°`，不跨种子平均动作。
- VERIFIED：40 个组合严格门槛全部通过（服务器 `strict_5seed_gate.json`，`strict_pass=true`，`row_failures=0`，`monotonicity_failures=0`）。跨组合最差/均值：根俯仰 MAE `0.136°/0.086°`，根俯仰 P95 `0.348°/0.226°`，完整前向 P95 `1.091°/0.558°`，水平朝向 P95 `1.106°/0.546°`，躯干方向 P95 `0.987°/0.307°`，`q_rigid` 最大 `0.077`；尾部额外 SO(3) 跳变最大 `1.179°`、额外俯仰步长最大 `0.384°`。每行一致性和剂量符号均通过，`|10°|>|5°|` 单调性逐样本逐种子通过。
- AUDIT：最差完整前向 P95 的可视化选择为 sample94/seed42、sample34122/seed1057660199；q_rigid 全部低于 0.2，说明 v1.3 已抑制根带动躯干的刚性随动，但脚部位置/腿部方向仍只是报告项，不代表接触问题已解决，也不宣称完成骨盆相对躯干局部前倾。
- VIDEO：服务器已生成并核验 H.264、1920×1080、20fps 的 16 个侧视审查视频（seed0 两个样本 + 每个样本最差种子；±5°/±10°，网格/骨架，含根实际前向、目标前向、躯干前向箭头和曲线）。服务器目录为 `/root/autodl-tmp/vimogen_clean/results/phase7/relative_root_forward_v1_3/videos/audit/`，本地副本为 `results/phase7/relative_root_forward_v1_3/videos/audit/`。
- ARTIFACTS：实现文件为 `sampling/relative_root_forward_guidance_v1_3.py`；运行/评估接入在 `train_eval_vimogen.py`、`scripts/run_relative_root_forward_v1.py`、`scripts/evaluate_relative_root_forward_v1.py`；诊断、严格门槛和服务器批处理分别为 `scripts/kinematic_feasibility_v1_3.py`、`scripts/collect_relative_root_forward_v1_3_smoke.py`、`scripts/evaluate_relative_root_forward_v1_3_strict.py`、`scripts/run_relative_root_forward_v1_3_holdout_server.sh`。协议记录哈希为 `a4fc4bddffe7c4f68ef7c686edc97032e256dda153e2dc7039ddd5c6bf27dfdc`。
- SCOPE：v1、v1.1、v1.2 代码及失败结果保持冻结；v1.3 当前成功范围仅是两个样本、五种子、`±5°/±10°` 的根前向相对控制并抑制躯干刚性随动。未进入 20 条开发集或正式 MBench；后续若处理脚部位移，应另建接触感知根位置/腿部补偿协议。

## 2026-08-28：根前向相对引导 v1.1 残差自适应校准完成

- ARCHIVE：旧的 `vimogen_relative_root_forward_v1_pose_authoritative` 已单独归档到 GitHub 分支 `codex/archive-relative-root-forward-v1`，提交 `1f36e698a5b617f461ed888a3479a547c9015c8a`；当前 v1.1 实现在分支 `codex/relative-root-forward-v1-1`，最新提交 `f2e54a7` 已推送（含初始实现 `a236b90`）。模型权重、结果数据和凭据未上传。
- IMPLEMENTED：新增协议 `vimogen_relative_root_forward_v1_1_residual_adaptive`。直接 `body_pose/root_rotation/root_translation` 是唯一权威量，`J/dJ/dR/dT` 全部由 FK 和差分派生；引导只在冻结 M0 右轴上施加逐帧一维标量根旋转，速度不接受引导梯度。
- IMPLEMENTED：残差自适应提议为 `clamp(guidance_strength * residual_gain * signed_residual_deg, ±max_step_deg)`；支持 `[sigma_min,sigma_max]` 校准窗口、双损失回溯、传递增益记录和失败尝试保留。传递增益诊断改为同一步欧拉反事实计算，不再二次调用会推进调度器的 `step`。
- VERIFIED：所有动态测试通过服务器 `connect_server.py`：专项 `tests/test_relative_root_forward_v1.py` 为 `8 passed in 1.35s`，完整回归为 `212 passed in 23.98s`。本机只做语法检查，不作为动态测试环境。
- CALIBRATION：服务器产物位于 `/root/autodl-tmp/vimogen_clean/results/phase7/relative_root_forward_v1_1/`，复用同一份 `sample-noise-v1` 和 M0。当前根目标最优候选为 `residual_gain=1.0`、`max_step_deg=6°`、`sigma=[0.0662879,0.65]`；sample94/34122 的 `+5/+10/-5/-10°` 实际根变化约为 `+5.003/+9.995/-4.980/-9.972°`，符号和绝对剂量单调性均通过。
- VERIFIED：该候选的逐帧角度 MAE 约 `0.058–0.101°`、P95 约 `0.135–0.269°`，一致性 `J-FK/dJ/dR/dT` 和尾部额外跳变均通过硬门槛；但严格完整向量和水平朝向 P95 仍约 `1.85–2.65°`，正向 sample34122 超过 `2°`，故 v1.1 当前不能标记为完整控制成功，只能标记为“根前向目标达标、完整向量二级门槛未达标”。
- AUDIT：全身随动 `q_rigid` 接近 `0.93–1.02`，说明这一阶段主要产生根/全身刚性俯仰；该结果不宣称实现骨盆相对躯干局部前倾。尚未进入开发集、视频或正式 MBench。
- VIDEO：服务器已生成并核验 8 个 v1.1 侧视审查视频（sample94、sample34122；`M0|-5|+5` 与 `M0|-10|+10`；网格和骨架两种版本），位于 `/root/autodl-tmp/vimogen_clean/results/phase7/relative_root_forward_v1_1/videos/audit/`；本地副本位于 `results/phase7/relative_root_forward_v1_1/videos/audit/`。视频使用固定 M0 运动朝向相机，叠加实际根前向、目标前向和 `spine1→neck` 躯干前向。
- SEED_SCOPE：上述结果只使用 `seed=0`，不是多个 seed 的平均；两个样本在同一次 batch 中生成，M0 和所有剂量复用同一份 `sample-noise-v1` 缓存。

## 2026-08-28：根前向相对引导 v1 已在服务器完成动态回归与真实冒烟

- SERVER_SYNCED：本次实现已同步到服务器干净工作目录 `/root/autodl-tmp/vimogen_clean`；动态运行使用 `/root/miniconda3/envs/mdm5090/bin/python`（PyTorch `2.7.0+cu128`、CUDA 可用、pytest `9.1.1`），不是本机缺少依赖的 `/usr/local/bin/python`。
- VERIFIED：服务器专项 `tests/test_relative_root_forward_v1.py` 为 `5 passed in 1.38s`；恢复服务器原有 `motion_rep/m1_consistent.py` 后完整回归为 `209 passed in 23.72s`。服务器原 M1 实现保留在原路径，本次 v1 同步不得覆盖它；同步前备份位于 `diagnostics/phase7/relative_root_forward_v1/server_pre_sync_20260828/`。
- FIXED：`sampling/flow_sampler.py` 的 batch-invariant 轨迹合并增加标量字段分支，避免 `trace_enabled=true` 时对 `sigma/timestep` 在不存在的维度上 `cat`；`motion_rep/pose_authority.py` 的 SO(3) 测地角改为双精度 `atan2(sin,cos)`，避免近单位旋转被单精度 `acos` 虚报约 `0.03°`。
- REAL_SMOKE：服务器已用同一 sample-noise-v1 缓存完成 sample94 与转弯样本 34122 的 M0、`-5°`、`+5°`、`-10°`、`+10°` 五个运行，所有产物只写入 `results/phase7/relative_root_forward_v1/`；运行记录和模型输入摘要保存在各 `runs/smoke/seed_000/delta_*/attempt_01/`。
- VERIFIED：五个运行的 M0 一致化端点逐位相同；零剂量 `G0` 与 `M0_consistent` 逐位相同；最终候选的 `J-FK`、`dJ`、`dR`、`dT` 均通过硬门槛，尾部最后八个有效相邻帧的额外 SO(3)/俯仰跳变均小于 `2°`。汇总为服务器 `results/phase7/relative_root_forward_v1/summaries/smoke_validation.json`。
- NEGATIVE_RESULT：以协议默认 `guidance_strength=1`、单次 `1°` 基础提议和当前有效 sigma 窗口运行时，真实冒烟实际达到的根向下变化约为：`+5° -> [1.2754°, 1.2365°]`、`+10° -> [1.3290°, 1.2152°]`、`-5° -> [-1.4408°, -1.5497°]`、`-10° -> [-1.4186°, -1.5681°]`（两个数字分别对应 sample94/34122）。因此剂量单调性未通过，不能宣称已实现 `5°/10°` 精确控制；当前结果只证明姿态权威化、一致性和方向符号链路可运行，并暴露需要重新校准引导强度/步长或有效步数。
- PENDING：本轮尚未把未达到目标的结果升级为正式 Phase 7 成功，也未进入开发集、MBench 或正式视频；应先由用户决定是否允许在冻结协议之外做引导强度/步数校准实验。旧 v1-v4 结果和协议未覆盖。

## 2026-08-28：根前向相对引导 v1 姿态权威实现进行中

- IMPLEMENTED：新增 `motion_rep/pose_authority.py`，直接 `body_pose/root_rotation/root_translation` 作为唯一物理权威；通过轻量 SMPL-X 22 关节 FK 重建 `J/dJ/dR/dT`，显式持有最后有效的隐藏 T+1 姿态，并记录 276D 投影改变量与一致性门槛。
- IMPLEMENTED：新增 `sampling/relative_root_forward_guidance.py`，协议名为 `vimogen_relative_root_forward_v1_pose_authoritative`。M0 官方 FP32 输出只投影一次并冻结 `f/h/r/phi`；引导只在冻结 M0 右轴上优化逐帧标量根旋转，速度通道全部回算，支持 11 次二分回溯、0.05 标准化 RMS 上限和零剂量一致基线。
- IMPLEMENTED：`sampling/flow_sampler.py` 与 `train_eval_vimogen.py` 增加 `relative_root_forward` 配置入口、与 M1/绝对角/额外 reconciliation 的互斥检查、默认产物目录 `results/phase7/relative_root_forward_v1/`，并保留旧协议路径。
- IMPLEMENTED：新增 `evaluation/relative_root_forward_v1.py` 与离线评估脚本，覆盖控制曲线、水平朝向漂移、尾部额外跳变、全身刚性随动和足部接触审计；新增 `tests/test_relative_root_forward_v1.py`。
- IMPLEMENTED：最终 G0 汇总同时携带控制曲线、P95、相关/波动比例、尾部安全、全身随动和一致性硬门槛；零剂量明确返回冻结的 `M0_consistent`。
- RESTORED：将未跟踪的服务器追踪副本原样恢复为 `sampling/m1_guidance.py`；补充最小 `motion_rep/m1_consistent.py` 兼容入口，避免旧 M1 导入链断裂。
- PENDING：当前工作站没有 PyTorch/pytest 运行时，尚未执行本地动态测试；需在带项目依赖与模型权重的服务器上运行单元测试及 sample94/34122 的 M0、±5°、±10° 真实冒烟。旧 v1-v4 结果和协议不得覆盖。

## 2026-08-28：276 维人物前向/根前向来源审计

- VERIFIED：276 维不单独存储“人物前向”向量。`258:264` 直接保存规范化后的 SMPL-X `global_orient` 旋转矩阵的 Rot6D；`264:270` 保存 `dR_t=R_(t+1)R_t^T`。Rot6D 是旋转矩阵前两列按行交错展开，第三列由叉乘重建。
- VERIFIED：动作规范化只在首帧使用左右髋：`x=normalize(horizontal(J_right_hip-J_left_hip))`、`z=(0,0,1)`、`y=normalize(z×x)`，再以 `Q=[x^T;y^T;z^T]` 左乘整段关节、根旋转和根平移。因此首帧的骨盆横向被对齐到世界 `+x`，由左右髋和右手系确定的前方被对齐到世界 `+y`；肩关节索引虽声明但没有参与计算。
- VERIFIED：逐帧“根前向”定义为 `f_t=R_t e_z`，即规范化根旋转把 SMPL-X 根局部 `+z` 轴映射到世界后的方向；水平人物朝向为 `normalize(f_t-(f_t·e_z)e_z)`。它不使用鼻尖、ASIS/PSIS、运动轨迹或头部朝向。
- VERIFIED：HMR 路径先从 `global_orient/body_pose/transl` 经 SMPL-X 得到 22 关节，再用固定 `R_motionx_to_amass` 左乘转换坐标，之后执行上述首帧髋规范化。光学 Mocap 发布数据被官方文档声明为相同的全局/规范化 DART-276 表示，但仓库未提供生成已发布 AMASS/光学子集的完整源数据转换脚本，故源数据集级转换只能确认到发布说明，不能逐样本重放。
- VERIFIED：真实 sample94 M0 中，根旋转第三列的水平前向与逐帧左右髋推导前向余弦 `min/mean/max=0.935760/0.993374/0.999998`；平均根前向与整段位移方向余弦为 `0.991911`。二者在该样本中接近但定义上独立；后退、侧移和原地转身时不得用位移方向替代根前向。
- DECISION：生成模型会独立预测根旋转和关节位置等冗余通道，二者不保证严格一致；当前 v2/v3 权威边界以 `258:264` 根旋转和 `0:126` 局部姿态为权威，经 FK 重建关节和全部差分通道。

## 2026-08-28：v4 ASIS/PSIS 定义与模板坐标轴进一步审计

- CLARIFIED/VERIFIED：SMPL-X 官方顶点标志只提供鼻尖、眼、耳、手指和足部等映射，不提供 ASIS/PSIS。v4 的 LASI/RASI/LPSI/RPSI 是从中性模板表面人工选择并取组均值的项目专用虚拟标志，不是 SMPL-X 自动恢复或官方解剖标志。
- PROVENANCE/VERIFIED：原始候选规则仅保存在 2026-08-27 会话日志而未写成仓库脚本：在原始中性模板中限定 `-0.38<y<-0.18`、`0.11<|x|<0.22`，以 `z>0.07` 选 ASIS 候选、`z<-0.06` 选 PSIS 候选，再按 `x` 正负分左右并人工查看 `diagnostics/phase6_pelvis_candidates.png`。该图正确显示红色 ASIS 位于 `+z`、蓝色 PSIS 位于 `-z`。
- ROOT CAUSE REFINED：错误发生在候选编号转录到 `configs/pelvis_landmark_groups_v4.json` 时：图中/会话变量的 ASIS 正 `z` 编号被写入 LPSI/RPSI，PSIS 负 `z` 编号被写入 LASI/RASI。因候选生成规则未纳入版本控制且标定脚本没有前后语义断言，此互换未被自动发现。
- VERIFIED：此前 `P→A` 与鼻尖方向余弦 `-0.99952155` 的检查在中性模板中将 `y` 轴作为竖直轴并投影到 `x-z` 平面，因此它只是一项一次性的模板前后方向校验，不使用逐帧姿态下会随头部转动的鼻尖。
- COORDINATE FRAME CLARIFIED：`configs/pelvis_landmarks_v4.json` 中的标志点是原始 SMPL-X 根局部坐标（局部 `y` 向上、`+z` 向前），而 `up_axis=2` 指规范化 276 维世界坐标的 `+z` 向上。`anatomical_pelvis_geometry` 先用每帧规范化根旋转把局部标志变换到世界，再以世界 `+z` 测量，因此这两处轴定义本身不矛盾；此前将其判为第二类坐标轴错误的记录已更正。
- IMPACT/CORRECTION：交换 ASI/PSI 后角度逐帧满足 `new=-old`，仍可证明当前标签反转会翻转符号；但 v5 仍须显式记录“局部 SMPL-X `+z` 前方、局部 `+y` 上方、规范世界 `+y` 初始前方、世界 `+z` 上方”的变换链，并加入端到端坐标测试，避免再次混淆局部轴和世界轴。
- INVARIANT：鼻尖只允许用于中性模板的一次性方向一致性检查；逐帧骨盆前向与倾角必须完全由骨盆局部坐标/虚拟 ASIS-PSIS 定义，不得受转头、注视方向或运动轨迹方向影响。

## 2026-08-27：v4 解剖前后标志方向错误，现有 v4 结果不得继续解释为解剖前倾

- USER_REPORT/VERIFIED：用户指出视频中 P→A 骨盆前向线指向人物后侧。使用 SMPL-X 官方 `nose=9120` 顶点作为独立人体前方参考复核：冻结配置的 `A-P` 与“骨盆→鼻尖”水平向量余弦为 `-0.99952155`，几乎完全反向；这不是相机镜像。
- ROOT CAUSE：`configs/pelvis_landmark_groups_v4.json` 将负局部 z 顶点组标为 LASI/RASI、正局部 z 标为 LPSI/RPSI，并错误记录“人物朝局部 -z”。实际中性 SMPL-X 模板鼻尖相对骨盆位于局部 `+z`，故两组 ASIS/PSIS 标签前后颠倒。`anatomical_pelvis_geometry` 又从错误的 A-P 自身推导 heading，使原测试成为自洽但无法发现模板方向错误。
- IMPACT：交换 ASI/PSI 后，P→A 与鼻尖方向余弦恢复为 `+0.99952155`；sample94/+10° G0 的动作平均角从旧 `+9.931057°` 精确变为 `-9.931057°`，逐帧满足 `new=-old`。因此问题同时影响可视化、数值正负和引导语义，不能只改箭头或重新解释现有 v4 结果。
- DECISION GATE：现有 v4 代码、协议哈希、结果和视频保留为已发现标定方向错误的历史证据，不得作为解剖骨盆前倾结论。修复必须另建新协议（建议 v5），交换并重新审核 ASI/PSI 组，增加“P→A 与官方鼻尖前方同向”的硬标定测试，并从标定、几何、引导和 sample94 +5°/+10° 冒烟重新运行。

## 2026-08-27：v4 网格/骨架运动朝向与标记统一修正

- DIAGNOSED：同一条 sample94 M0 根轨迹在世界坐标末帧相对首帧沿 `+y` 移动约 `0.127`；网格路径经过 PyTorch3D 屏幕坐标后向左，手工骨架投影向右，根因是两条渲染路径的屏幕横轴符号相反，不是动作数据本身相反。
- FIXED：`scripts/render_absolute_mean_triptych.py` 新增基于 M0 根轨迹的序列级运动朝向估计；由该朝向构造一次固定侧视相机，网格和骨架共用，不再写死世界 `y` 轴或分别变换。手工投影加入与 PyTorch3D 一致的横轴符号，消除网格/骨架镜像。
- FIXED：主画面新增青色运动方向箭头，局部插图从固定 `front ->` 改为 `motion ->`；红色 `P -> A` 仍严格表示解剖骨盆线。人物逆着运动方向时二者相反会被明确展示，而不再把运动方向和解剖前方混为一谈。
- VERIFIED：新增 `tests/test_render_absolute_mean_triptych.py`，覆盖正/负运动朝向与 PyTorch3D 屏幕符号；服务器专项 `7 passed`，完整回归 `204 passed in 26.20s`。真实 sample94/+5° 和 +10° 网格、骨架视频均重新生成并核验为 H.264、20fps、1920x1080，临时复核产物位于 `results/phase6/absolute_mean_pelvis_v4/videos/smoke_motion_oriented/`。
- ARCHIVE：本次渲染修正已按独立批次提交并推送 GitHub，提交 `cf28af1`（`fix: unify motion-oriented mesh and skeleton display`）；v3 结果与代码未改写。
- ARCHIVE：随后补充运动朝向输入的有限值/非零校验，提交 `84b1730`（`test: validate motion display heading inputs`）。

## 2026-08-27：v4 解剖骨盆前倾控制与防作弊审计已实现

- ARCHIVE：已将此前代码按批次上传至 GitHub 仓库 [chu62928-cloud/vimogen_m](https://github.com/chu62928-cloud/vimogen_m)：`a052bd0`（归档保护）、`11e8fe0`（运动表示/评价基线）、`a2cb7e5`（实验脚本/回归测试）。v4 后续提交为 `9bf94af`（统一解剖几何）、`55bc0d0`（v4 引导/评价/标定）、`592cd30`（局部主导安全项）、`7544acc`（局部占比统计修正）、`65ee191`（视频审计标记）、`00d5e66`（训练入口 v4 分支）、`5dcd795`（关闭引导时终端边界修正）、`4913ada`（归档说明）、`73ee8ed`（固定安全项协议说明）。模型权重、视频结果、数据和凭据未上传。
- IMPLEMENTED：新增 `motion_rep/anatomical_pelvis.py`，冻结项目专用 LASI/RASI/LPSI/RPSI 虚拟标志、模板哈希和根关节；统一计算 `theta=atan2(-dot(A-P,up),dot(A-P,heading))`，固定“前侧向下为正”。新增躯干与双侧大腿角、局部变化占比、低信号标记和比例审计。
- IMPLEMENTED：新增独立协议 `vimogen_absolute_mean_pelvis_v4_anatomical_local`、受限活动通道引导、2°软防作弊损失、局部变化至少一半且同向的固定安全项、约束终端细化；v3 代码、协议、结果和视频未修改。
- CALIBRATION：中性模板 `SMPLX_NEUTRAL.npz` 哈希为 `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`；v4 协议哈希为 `4dfa8774008f843c1d483db971252b6808e7c9e4707384216b1e141ac040df99`，服务器标定摘要哈希为 `f19098a04dc67827d3f736875730d52215a48a9398e2c2e84dabce3699349e9a`。
- VERIFIED：服务器专项回归 `5 passed`，完整回归 `202 passed in 26.88s`。sample94/seed0 +5°（attempt_06）目标误差 `0.602609°`、曲线相关 `0.983021`、躯干/左大腿/右大腿变化 P95 为 `1.082785°/1.039001°/1.028196°`、局部占比 `0.794593`；+10°（attempt_02）目标误差 `0.068943°`、曲线相关 `0.983364`、P95 为 `0.821530°/1.192602°/1.328780°`、局部占比 `6.803685`（比例低信号帧约 `99%`，仅作审计）。两组 276D 一致性残差均通过 v3 容差。
- VERIFIED：v4 冒烟视频已生成四条 H.264、20fps、1920×1080 文件，位于服务器 `/root/autodl-tmp/vimogen_clean/results/phase6/absolute_mean_pelvis_v4/videos/smoke/`；网格和骨架复用同一 P/A 端点，主画面标出 P→A，局部插图固定前方向右、前倾向下。尚未运行开发集、主盲测和鲁棒性实验。

## 2026-08-27：官方 MBench v3 已安全暂停并完成断点归档

- VERIFIED：用户要求暂停后，恢复目录 `/root/autodl-tmp/vimogen_clean/results/phase5/mbench_publication_v1/official_motion_quality_v3_resume_20260826/` 已改为 `USER_STOPPED_ARCHIVED`；最终状态为 `completed_jobs=16/27`、`pending_jobs=7`、`active_workers=0`、`failed_jobs=0`。
- VERIFIED：暂停前运行中的 4 组保留为 `RUNNING` 记录，未伪造为完成：`m1_plus10/seed_000/absolute_position`、`m1_plus10/seed_000/velocity_integral`、`m1_plus5/seed_002/reconciled`、`m1_plus5/seed_002/velocity_integral`。下次恢复时按组级断点重跑这些记录和 7 个待运行组，已完成 16 组只读复用。
- VERIFIED：本次调度器、工作进程和评估进程均已退出；1691/1691 个 SMPLify 缓存通过完整性审核，隔离数为 0。
- ARCHIVE：停机证据归档于 `/root/autodl-tmp/vimogen_clean/diagnostics/phase5/mbench_publication_v1/official_motion_quality_v3_resume_stop_archives/user_stop_20260827T014948Z/`，包含 `pre_stop_manifest.json`、`post_stop_manifest.json`、`scheduler_state_before_stop.json` 和 `cache_manifest.json`；恢复目录本身的 `scheduler_state.json` 记录了 `post_stop_manifest.json` 路径。

## 2026-08-27：官方 MBench v3 恢复运行进度核验

- VERIFIED：恢复目录 `/root/autodl-tmp/vimogen_clean/results/phase5/mbench_publication_v1/official_motion_quality_v3_resume_20260826/` 调度器仍为 `RUNNING`；当前 `completed_jobs=16/27`、`active_workers=4`、`pending_jobs=7`、`failed_jobs=0`，缓存约 `1677/2700`。
- VERIFIED：旧目录 `/root/autodl-tmp/vimogen_clean/results/phase5/mbench_publication_v1/official_motion_quality_v3/` 仍为 `USER_STOPPED_ARCHIVED`、`completed_jobs=8`、`active_workers=0`，未被恢复运行写入。
- CURRENT_WORKERS：当前运行组为 `m1_plus5/seed_002/velocity_integral`、`m1_plus5/seed_002/reconciled`、`m1_plus10/seed_000/absolute_position`、`m1_plus10/seed_000/velocity_integral`；调度器因 GPU 利用率 100% 维持 4 路并行。

## 2026-08-27：已完成组的 MBench 双指标输出核验

- VERIFIED：恢复目录中 16 个 `COMPLETED` 组均已生成 `*_eval_results.json`、`*_full_info.json`、`*_per_motion_results.json` 和 `merged_per_motion_results.json`；`Body_Penetration` 与 `Pose_Quality` 均包含 aggregate 的 mean/std/num_samples 以及逐动作值（当前每组 100 个样本）。
- NOT_READY：剩余 11 个未完成组尚无完整双指标评价；官方三路总汇总需等待 27 组全部完成后由调度器生成。

## 2026-08-26：官方 MBench v3 非覆盖式恢复运行已启动

- USER_REQUEST：用户要求重新跑官方 MBench v3，优先补齐未完成组，已完成组不得覆盖。
- VERIFIED：旧目录 `/root/autodl-tmp/vimogen_clean/results/phase5/mbench_publication_v1/official_motion_quality_v3/` 保持原样，状态仍为 `USER_STOPPED_ARCHIVED`、8/27 组完成、0 失败、0 活跃工作进程。
- STARTED：新恢复目录 `/root/autodl-tmp/vimogen_clean/results/phase5/mbench_publication_v1/official_motion_quality_v3_resume_20260826/`，启动 PID `202496`；启动前仅复制旧目录中 8 个 `COMPLETED` 组，4 个中断组和 15 个无记录组在新目录重新运行。
- SAFETY：恢复清单为 `resume_manifest.json`，记录了 8 个已复制组的源记录 SHA256 以及 19 个待重跑组；源目录和已完成结果未写入。新调度器状态/日志在恢复目录的 `scheduler_state.json`、`scheduler.log`，摘要目标为 `diagnostics/phase5/mbench_publication_v1/official_motion_quality_summary_v3_resume_20260826.json`。
- STATUS_AT_START：新目录调度器先启动 4 个工作组，8 组完成、4 组运行、15 组等待；使用旧组织数据和旧官方输入，未覆盖已完成结果。GPU 利用率达到 100% 后调度器按资源门槛暂不追加并行度。

## 2026-08-26：v3 尾部安全融合修复与真实冒烟验证

- FIXED：用户观察到的 +10° 末尾突变已在独立协议 `vimogen_absolute_mean_pelvis_v3_tail_safe` 中修复；v2 代码、协议、结果和视频均未覆盖。
- ROOT CAUSE：v2 在窗口 9 融合时将隐藏的 T+1 姿态外推并端点复制进平均窗口，导致末帧直接根旋转梯度放大；v3 只对 T 个真实输出帧使用不复制端点的截断滑动平均，并将隐藏 T+1 姿态固定为最后一个融合输出姿态。
- IMPLEMENTED：`motion_rep/consistency_v3.py` 完整执行权威姿态 -> FK -> 重建 J -> 重建 dJ/dR/dT -> 重新封装 276D；`sampling/absolute_mean_pelvis_guidance_v3.py` 复用 v2 的绝对平均/局部矢状面角度损失，仅替换尾部安全边界；`train_eval_vimogen.py` 已增加显式 v3 协议入口。
- VERIFIED：服务器专项回归 `3 passed`，与既有测试合计 `197 passed in 10.27s`；无 GPU 的整套收集失败仅因项目既有的 T5 默认调用 `torch.cuda.current_device()`，不是 v3 回归失败。
- VERIFIED：真实 `sample94`、seed0、strength=1.0、shape=0.1 冒烟完成。+5° G0 平均误差 `0.498478°`、去均值曲线相关 `0.974486`、G0 尾部最大单帧跳变 `0.427399°`；+10° G0 平均误差 `0.880614°`、去均值曲线相关 `0.961426`、G0 尾部最大单帧跳变 `0.434017°`。两者 G0/G1 尾部均通过 `2°` 突变审查。
- VERIFIED：+5°/+10° G0 的 J-FK 最大残差分别为 `2.137969e-6 m`、`2.833386e-6 m`；dJ 最大残差分别为 `7.450726e-8 m`、`8.940697e-8 m`；dR 最大残差分别为 `3.249819e-5°`、`1.315431e-5°`；dT 最大残差均约 `2.98e-8 m`。
- ADDED：v3 协议及逐字节复用的数据清单冻结于服务器 `/root/autodl-tmp/vimogen_clean/results/phase6/absolute_mean_pelvis_v3/protocol.json`，SHA256=`fc04ae409adee777ef1a0d23e1b040d0d52a81c030b48b01674816d486d3dd28`；冒烟核验为 `.../summaries/smoke_validation.json`，回归日志为 `.../logs/full_pytest.log`。
- ADDED：真实 H.264、20fps、1920x1080 三路并排侧视视频（网格和骨架两种显示）已归档于 `results/phase6/absolute_mean_pelvis_v3/videos/smoke/`，包括 `sample94_target05_triptych_sagittal_side.mp4`、`sample94_target05_triptych_skeleton_sagittal_side.mp4`、`sample94_target10_triptych_sagittal_side.mp4`、`sample94_target10_triptych_skeleton_sagittal_side.mp4`；固定相机为真正矢状面侧视，骨架版突出显示根部和两侧髋部。
- DECISION：v3 冒烟已证明末端突变修复和 276D 一致性链路有效，但不能替代冻结协议要求的 20 条开发集门槛；尚未运行 seed1/2、40 条主盲测或 450 条鲁棒性实验，`result.md` 保持不更新。

## 2026-08-26：v3 骨盆角可视化增强

- CORRECTION：原视频只显示底部角度曲线，人体上没有标出骨盆角的几何对象，导致 +10° 肉眼不明显。
- ADDED：`scripts/render_absolute_mean_triptych.py` 现在在网格和骨架源视频上叠加同一角度定义的标记：红色箭头为去除当前人物朝向后的骨盆局部前向，青色箭头为局部水平朝向，红/青箭头夹角即 `pelvis_sagittal_tilt_degrees`；每帧左上角显示当前角度。
- VERIFIED：重新生成的 v3 +5°/+10° 网格和骨架侧视视频已覆盖到 `results/phase6/absolute_mean_pelvis_v3/videos/smoke/`；+10° G0/G1 尾帧分别显示约 `+8.5°`/`+9.4°`，箭头方向清晰且与曲线数值一致。旧无标记视频仍保留在各次 run 目录中。

## 2026-08-26：+10° 冒烟视频末端骨盆突变诊断

- VERIFIED：用户观察到的末端人物倾斜是真实运动学输出，而非侧视渲染错误。可重复的红色诊断条件为：+10° G0 最后 8 帧的最大角度步长为 `3.657274°`，超过 M0 的 `0.747942°` 三倍且大于 `2°`。G0 角度从第 95 帧 `10.413341°` 升至第 99 帧 `22.507975°`；G1 只在所有帧近似额外施加 `+0.660°`，不是根因。
- ROOT CAUSE VERIFIED：尖峰在 `guided_raw_norm_batch.pt` 的直接根旋转中已存在，尚未经过最终 G0 的 FK 重建/276D 重封装；G0 之后直接根旋转—旋转速度误差仅 `0.000540°`，说明完整 276D 一致化正确地保留并统一了异常姿态，而没有生成该异常。
- ROOT CAUSE VERIFIED：冻结的窗口 9 根旋转融合在末端构造隐藏 T+1 姿态时同时执行“末帧增量外推”和端点重复填充。对同一 M0 输入，当前窗口 9 的最后一帧直接根旋转梯度是倒数第二帧的 `6.2533` 倍；诊断性改为 window=1 后为 `1.0008` 倍，改为 hidden pose 持有末帧仍为 `2.5023` 倍。该边界放大与只约束平均角/去均值位置差而未约束角速度的损失共同使 +10° 修正集中到末帧。
- VERIFIED：+5° 同一动作的末端最大角度步长为 `0.551701°`，没有出现可视突变；说明该缺陷在较大控制需求下被放大。
- ADDED：只读诊断工具 `scripts/diagnose_absolute_mean_pelvis_v2_tail.py`；机器证据在 `diagnostics/phase6/absolute_mean_pelvis_v2/tail_instability/sample94_target10.json` 和 `sample94_target05.json`。未修改冻结 v2 协议、模型输出、开发集结果或 `result.md`。
- DECISION：若用户授权修复，必须建立新版本协议；优先审查融合末端边界（隐藏姿态不外推、损失图不让隐藏姿态污染末端输出）与显式 SO(3) 角速度/末端平滑项，再从单动作冒烟重新验证，不能直接覆写 v2。

## 2026-08-26：冒烟视频侧视相机修正

- CORRECTION：先前标为固定侧视的 v2 冒烟视频实际使用了正面投影；随后新增的第一次侧视矩阵因渲染器采用左乘行向量而将人物渲染成横向。两类问题均未改动模型、指标或冻结协议。
- VERIFIED：`scripts/render_absolute_mean_triptych.py` 已改为针对 z-up 坐标的真正矢状面侧视：图像水平轴取人物行走方向，图像竖直轴取世界 z 轴，深度轴取世界 x 轴，并根据 M0 根轨迹居中固定相机。
- VERIFIED：修正版视频已重新生成并核验为 H.264、20fps、1920×1080：`results/phase6/absolute_mean_pelvis_v2/videos/smoke/sample94_target05_triptych_sagittal_side.mp4` 与 `sample94_target10_triptych_sagittal_side.mp4`。人工抽帧确认人物竖直、侧身轮廓可见；原有无后缀视频保留为历史显示产物。

## 2026-08-26：用户要求安全停止官方 MBench v3，已完成并归档

- VERIFIED：已先停止旧 `absolute_mean_pelvis_v1` 自动冒烟看门程序 PID `91034` 及其父进程，避免 v3 停止后旧方案误启动。
- VERIFIED：官方 MBench v3 在用户要求停止前已完成 `8/27` 组、失败 `0` 组；停止前结构化状态、全部完成/未完成 `run_record.json` 及完成组文件哈希已冻结到服务器 `diagnostics/phase5/mbench_publication_v1/official_motion_quality_v3_user_stops/user_stop_20260826T071924Z/`。
- VERIFIED：已按精确进程身份停止调度器 PID `60732` 和四个工作进程组；复核不存在 `run_official_mbench_repair_parallel.py`、`run_official_mbench_repair_worker.py`、`evaluate_mbench.py` 或旧 phase-6 看门进程，GPU 计算进程为空。
- VERIFIED：v3 状态为 `USER_STOPPED_ARCHIVED`、`active_workers=0`、`completed_jobs=8`。四个被中断组的原记录继续保留为 `RUNNING`，未伪造为完成；最终汇总未生成，`result.md` 不更新。
- VERIFIED/CORRECTED：首次停机审计器误将官方 `joints` NumPy 数组当作非法类型，并把 987 个缓存可恢复地移动到停机归档；第一次更正又误假设每条动作固定 100 帧。最终按“字段允许张量/NumPy、首维可变、三字段帧数一致、尾部形状固定、全部有限值、SHA256 与移动前一致”复核后，987/987 缓存全部原位恢复，隔离数为 0。原始误判与更正证据均保留，最终更正为 `cache_audit_correction_v2.json`。
- DECISION：用户确认后续新底座必须执行“权威姿态 -> FK -> 重建 J -> 重建 dJ/dR/dT -> 重新封装 276D”；骨盆角改为去除逐帧人物朝向后在人物局部矢状面内计算的几何倾角，不能继续使用简单欧拉俯仰。现有 `absolute_mean_pelvis_v1` 保留为历史版本，不覆盖；新实现登记为 v2，先专项测试和单动作冒烟，再进入开发集。

## 2026-08-26：v2 底座、协议冻结与真实冒烟完成，开发网格已启动

- VERIFIED：v2 新增 `motion_rep/consistency_v2.py`，固定执行权威姿态 -> 可微 FK -> 重建 J -> 重建 dJ/dR/dT -> `finalize_motion` 重新封装 276D；中性 SMPL-X 22 关节骨架与真实 `smplx` FK 对照最大绝对差 `2.384e-7`。
- VERIFIED：v2 新增 `motion_rep/sagittal_pelvis_angle.py`，先由根旋转的水平投影估计人物航向，再在人物局部矢状面内计算骨盆倾角；已覆盖转弯、滚转、退化航向和 G1 修正轴测试，旧 v1 欧拉角文件未覆盖。
- VERIFIED：v2 协议已在模型运行前冻结到 `/root/autodl-tmp/vimogen_clean/results/phase6/absolute_mean_pelvis_v2/protocol.json`，SHA256=`359fa7be70137df01df9c95aa8b94c27ed2c1f08869965c252b87bd37376912b`；开发20、候选65、主盲测40、鲁棒性450和视频12 清单与 v1 冻结数据逐字节一致。
- VERIFIED：v2 专项与兼容测试合计 `41 passed in 5.15s`；两组真实动作冒烟均生成并离线评价。+5° G0 误差 `0.443919°`、曲线相关 `0.922073`；+10° G0 误差 `0.667998°`，但曲线相关 `0.460716`，故冒烟只证明接口/编码链路，不作为开发门槛证据。
- VERIFIED：真实冒烟 MP4 已核验为 H.264、20fps、1920x1080，路径为 `results/phase6/absolute_mean_pelvis_v2/videos/smoke/sample94_target05_triptych.mp4` 与 `sample94_target10_triptych.mp4`；三路为 M0/G0/G1，固定侧视相机并含角度曲线、目标线、当前/运行平均值和末端修正标记。
- VERIFIED：开发 seed0 3×3 网格共 18 个目标单元已全部完成，使用 `scripts/orchestrate_absolute_mean_pelvis_v2_parallel.py` 最多 2 路运行；显存峰值约 20.6 GB/32.6 GB、温度最高 48°C。所有单元的逐帧 CSV、逐动作 CSV、摘要和日志均保留，串行切换产生的中断尝试标记为 `INTERRUPTED_FOR_PARALLEL_SWITCH`。
- VERIFIED：seed0 选择汇总为 `results/phase6/absolute_mean_pelvis_v2/summaries/development_selection_seed0.json`，9 个参数点全部未通过双目标完整门槛；最近门槛点为 strength=2.0、shape=0.2（+5° 曲线相关 0.892、+10° 相关 0.837），+10° 目标的曲线保持和部分 <=2° 率是主要失败项。G0 的 J-FK、dJ、dR、dT 均在冻结容差内；G1 个别单元 dR 在 1e-4° 附近略超，不能标为已通过。失败审查为 `summaries/development_gate_failure_review.md`。
- VERIFIED：切换并行调度器后，v2/旧版兼容专项回归为 `46 passed in 5.42s`，日志为 `results/phase6/absolute_mean_pelvis_v2/logs/final_targeted_pytest.log`；开发选择、并行运行和失败审查摘要已同步到本地 `results/phase6/absolute_mean_pelvis_v2/`。
- DECISION：按冻结协议，开发门槛失败后不运行 seed1/2 验证、不进入 40 条主盲测或 450 条鲁棒性实验，不晋升 G1 为主方法；v2 冻结协议和根目录 `result.md` 均未改写。后续只能按“角度/坐标约定→权威边界→平滑削弱→梯度/标准化→损失冲突→目标可行性→融合窗口→骨盆-脊柱补偿”顺序另建新审查/版本。

> 阶段2更新（2026-08-14）：统一评价接口与 B0/B1/B2 简单基线已完成首个可验证里程碑；角度语义及 FK/残差主路线仍保持 PENDING。

## Last verified

- 2026-08-31（v2 统一评价与独立可达性诊断完成）；当前分支代码、服务器回归、固定纯 v2 单例、统一门、诊断原始产物和最终网格 MP4 已核对。

## Objective

- 完成 ViMoGen 276D 统一表示的真实恢复验证、M0/M1 的 MBench 三路表示比较，并维护可直接用于论文撰写的正式结果账本。不重新训练模型，主线控制条件为相对基线 0°/+5°/+10°。

## Current state

- VERIFIED：本地当前工作分支为 `codex/relative-root-forward-v2`，代码提交 `2404343`；v1、v1.1、v1.2、v1.3 代码和结果保持冻结。
- VERIFIED：v2 统一评价、正确性修复、服务器回归和固定纯 v2 单例已完成；完整成功门为 `FAIL`，不能把 attempt_10 写入正式成功结果。
- VERIFIED：独立运动学和源噪声子空间诊断已完成，原始矩阵、运动张量、门控和报告均在最终诊断目录；它们不能用于纯 v2 调参或成功计数。
- VERIFIED：服务器干净副本 `/root/autodl-tmp/vimogen_clean` 保留所有失败 attempt 和实验记录；没有覆盖旧 v1.x 结果或删除历史证据。
- VERIFIED：项目根目录的 `result.md` 仍只保存已核验的正式结果；本轮 v2 负结果和诊断写入 phase7 诊断目录及本项目记忆，不改写 v1.x 正式账本。
- VERIFIED：服务器环境和动态执行约束见本节顶部记录与最终 `EXECUTION_SPEC.md`；本地未把缺少模型运行时当作动态验证环境。

## Verified results and evidence

- VERIFIED：远端原始目录仍为未跟踪文件状态；`origin` 为官方 ViMoGen GitHub 仓库。
- VERIFIED：旧批次 `z0`、M0 raw/official 张量、sigma/timestep 日程均已保存并通过逐张量回归。
- VERIFIED：样本级噪声本身在批大小 `1/2/4` 下逐位一致（`atol=rtol=0`），并通过 M0 空速度采样器集成测试；正式报告位于 `tests/artifacts/phase0/sample_noise_protocol_v1/sample_noise_protocol_report.json`。
- VERIFIED：阶段 0/阶段1相关完整回归本轮为 `22 passed in 3.89s`；日志位于 `tests/artifacts/phase1/phase1_pytest.log`。
- VERIFIED：真实 ViMoGen 模型样本级噪声输出的批大小无关性已验证；阶段0旧黄金基线未替换。

## Decisions and invariants

- FROZEN：`result.md` 是正式论文关键结果的唯一权威账本；所有正式关键结果必须在核验后同步写入该文件。
- FROZEN：`RESULTS_REPORT.md` 保存工程过程，`PROJECT_MEMORY.md` 保存状态和证据索引；二者不能替代 `result.md` 的论文结果正文。
- FROZEN：没有样本数、统计量、机器产物路径和摘要值的结果不得标记为 VERIFIED 或 FROZEN；不覆盖旧冻结结果，协议变化建立新版本。

- FROZEN：主控制为相对基线 `0°/+5°/+10°`；正式主参照为 `M0_canonical`；M1-K 用于分离运动学处理与解剖启发旋转变量的作用。
- FROZEN：不修改旧服务器目录；先使用远端干净提交副本。
- FROZEN：所有引导状态保持标准化空间，几何损失在反标准化物理空间计算，重建后重新标准化再回到流采样。
- FROZEN：所有实验产物必须按 `results/`、`tests/`、`evaluation/`、`scripts/`、`diagnostics/` 分区；禁止在项目根目录新建 `stage0_*`、`diagnostic_*` 等临时结果目录。
- FROZEN：`vimogen-sample-noise-v1` 只能显式调用；默认旧批次随机数路径保持不变。新协议缓存不得写入或覆盖两个旧 M0 黄金目录。
- FROZEN：新协议以单样本 `[T,C]` 形状为键的一部分，在 CPU 上按派生种子生成，再搬到目标设备；缓存同时保存 `.pt` 与 JSON 清单并校验原始张量字节 SHA256。
- FROZEN：阶段 0 正式输出统一写入 `results/phase0/smoke_rot6d_fixed/{motion,videos,frames,prompts}/`，样本使用 `sample_000`、`sample_001` 等零填充编号；旧解码器结果只能放在 `diagnostics/phase0/rot6d_layout_bug/old_render/`，不得作为正式结果读取。
- FROZEN：测试代码放 `tests/`，测试生成文件放 `tests/artifacts/phase0/`；指标、统计和评估报告放 `evaluation/phase0/`；可重复运行的命令脚本放 `scripts/`；故障定位中间文件放 `diagnostics/`，并在阶段结束时归档或清理。

## Important files and paths

- `plan_optimized.md`：当前执行方案。
- `plan.md`：原始方案，保持不变。
- `connect_server.py`：服务器连接脚本；凭据不记录到项目记忆。
- 服务器原始目录：`/root/autodl-tmp/vimogen`。
- 干净开发副本：`/root/autodl-tmp/vimogen_clean`。
- 阶段 0 正式结果：`/root/autodl-tmp/vimogen_clean/results/phase0/smoke_rot6d_fixed/`。
- 阶段 0 诊断归档：`/root/autodl-tmp/vimogen_clean/diagnostics/phase0/`。
- 阶段 0 测试产物：`/root/autodl-tmp/vimogen_clean/tests/artifacts/phase0/`；评估目录：`/root/autodl-tmp/vimogen_clean/evaluation/phase0/`。
- M0 旧批次黄金捕获：`/root/autodl-tmp/vimogen_clean/results/phase0/m0_golden_legacy_batch_seed42_rerun01/`。
- M0 新采样器回放：`/root/autodl-tmp/vimogen_clean/results/phase0/m0_golden_replay_new_sampler_seed42_rerun03/`。
- M0 回归报告：`/root/autodl-tmp/vimogen_clean/results/phase0/m0_golden_replay_new_sampler_seed42_rerun03/golden_regression_report.json`。
- M0 测试日志：`/root/autodl-tmp/vimogen_clean/tests/artifacts/phase0/m0_golden_pytest.log`。
- 样本级噪声实现：`/root/autodl-tmp/vimogen_clean/sampling/noise_protocol.py`；验证脚本：`/root/autodl-tmp/vimogen_clean/scripts/validate_sample_noise_protocol.py`。
- 样本级噪声测试：`/root/autodl-tmp/vimogen_clean/tests/test_sample_noise_protocol.py` 与 `tests/test_sample_noise_flow_integration.py`。
- 样本级噪声测试产物：`/root/autodl-tmp/vimogen_clean/tests/artifacts/phase0/sample_noise_protocol_v1/`。
- 验证脚本首次导入路径失败已归档：`/root/autodl-tmp/vimogen_clean/diagnostics/phase0/sample_noise_protocol_failures/attempt_01_import_path/`。
- 本地检查图：`E:\博士\统计机器学习\vimogen0809\results\phase0\smoke_rot6d_fixed\frames\`；本地旧解码器对照图和视频归档在 `diagnostics\phase0\rot6d_layout_bug\`。

## External systems and background jobs

- VERIFIED：连接服务器使用本地 `python .\\connect_server.py --cmd "..."`；外部服务器状态需每次重新核查。
- VERIFIED：干净副本当前提交为 `7faceca`，相对官方 `origin/main` 提前 5 个提交；原始 `/root/autodl-tmp/vimogen` 仍为 `No commits yet on main`，本轮未修改。

## Immediate continuation

1. 继续阶段1的显式关节角度代理定义与人工坐标/正负标定，输出独立 `results/phase1/` 校准材料；不修改阶段0结果。
2. 用已记录的 `diagnostics/phase1/m0_representation_audit/summary.json` 解释 canonicalization、SMPL-X 模板和显式关节/FK 差异；在证据不足前保持“绝对 FK vs 增量残差”决策门为 PENDING。
3. 完成统一的 `T+1 → T×276` 最终化接口和回归，明确最后一帧速度策略、mask 和标准化/反标准化边界。
4. 阶段1通过前不开始 guidance、骨盆控制、M1/M2 或其它后续实验；不覆盖旧 M0 黄金目录和样本级噪声协议。

## 阶段 0 故障定位（2026-08-13）

- 已确认阶段 0 冒烟失败的根因不是 SMPL-X、SMPL 或 MBench 数据。干净 Git 副本缺少原始服务器目录中一段未提交的注意力回退修复。
- 当前环境 `FLASH_ATTN_2_AVAILABLE=False`、`FLASH_ATTN_3_AVAILABLE=False`，会进入 PyTorch `scaled_dot_product_attention` 回退路径。原始上游代码把 `[B,T,H,D]` 先展平为 `[B*T,H,D]`，再额外增加维度，导致 `B=8,T=100` 时输出序列错误变为 800；TM2M 参考编码器随后在 `[8,100,2048]` 与 `[1,800,2048]` 相加时报错。
- 已将原目录中已验证的回退修复同步到干净副本的 `models/transformer/wan/modules/tm2m_model.py` 与 `models/transformer/wan/modules/t2m_model.py`。修复同时保留批次/时间维度、支持有效长度掩码，并修正无 FlashAttention 时的维度换位。
- 修复后的独立 GPU 形状测试通过：输入 `[8,100,16,8]`，输出仍为 `[8,100,16,8]`，设备为 CUDA，数据类型为 bfloat16。完整官方冒烟尚待服务器 SSH 恢复后复跑。
- 随后发现旧版 `torchgeometry` 在 PyTorch 2.7 中对布尔掩码做减法，导致动作保存阶段失败；原始服务器目录已有未提交的 PyTorch3D 旋转转换实现。已同步到干净副本并通过 128 个 Rot6D 的轴角/矩阵有限值与形状回归。
- 干净副本已挂载原始目录的外部资源：`data/body_models/smplx`，以及 `data/body_models/smpl` 中 3 个 SMPL 性别模型；资源不复制进 Git。SMPL-X 和 SMPL 均能由当前 `smplx 0.1.28` 成功实例化。
- MBench 资源位于 `/root/autodl-tmp/vimogen/data/mbench`，共 1350 个 `.pt` 文件、约 784 MiB；干净副本通过符号链接复用。
- MBench 逐文件加载检查通过：450 个动作张量（均为二维、276 维）和 900 个文本特征张量，0 个损坏文件；这说明 MBench 下载本身不是本次故障来源。当前阶段使用 SMPL-X 路径；原始 `data/body_models` 未提供 `smplh` 子目录，若未来切换 SMPL-H 需另行准备模型。
- 阶段 0 官方冒烟最终通过：`stage0_smoke_final`，单卡 50 步采样、4 条动作可视化均完成；每条输出为 `(100,276)`，并成功生成 mp4。模型参数 2,258,214,420，验证损失 308.35107421875。
- `mdm5090`（Python 3.10.20）中 `flash_attn` 和 `flash_attn_interface` 均不可导入，ViMoGen 的两个 FlashAttention 标志均为 `False`；服务器 base Python 3.12 环境另有 `flash-attn 2.7.4`，但二进制不能直接复用于 Python 3.10。阶段 0 先固定 PyTorch SDPA 回退，不安装新 FlashAttention。
- 干净副本提交 `35391cd`（`fix attention fallback and rotation conversion compatibility`），工作树干净且仅比远端主分支多 1 个提交；原始目录没有修改。

## Blockers and unknowns

- VERIFIED：旧批次 M0 黄金回归已完成，报告和哈希位于正式结果目录。
- VERIFIED：新样本级噪声协议已作为非默认、显式接口实现，并通过噪声层及空采样器批大小无关测试。
- BLOCKED：真实 ViMoGen 的新噪声 M0 回归需要用户明确批准越过噪声语义决策门；当前没有改变默认路径，也没有创建新的正式 `results/phase0/` 运行。

- VERIFIED（更新）：用户已批准并已完成独立样本级噪声 M0 回归；以上旧 BLOCKED 条目为历史记录，不代表当前阻塞。
- PENDING（当前）：阶段1最终角度代理、统一最终化流程以及绝对 FK/增量残差路线尚未冻结。

## Update history

## 2026-08-23：M0 原始 276 维表示一致性审计完成

- VERIFIED：新增只读评价模块 `evaluation/representation_consistency.py`、命令行脚本 `scripts/audit_vimogen_representation.py` 和 6 项专项测试；专项测试 `6 passed`，全项目回归 `122 passed in 18.96s`。未修改模型、采样器或既有 M0/M1 结果。
- VERIFIED：在冻结 `nonturning_v10_user_approved` 数据上审计 M0 Raw/Official 各 60 条轨迹；同源参考 20 条；MBench 参考 450 条。机器报告和图表位于 `diagnostics/phase1/vimogen_representation_consistency_v2/`。
- VERIFIED：M0 Raw 的位置—速度整体 Frobenius 残差中位数为 `0.049902 m/frame`，相对直接帧间步长为 `0.403513`；Official 分别为 `0.023100 m/frame`、`0.214497`。Official 平滑降低了单步残差，但没有降低累计漂移。
- VERIFIED：直接位置—速度积分轨迹最终漂移中位数为 Raw `0.467329 m`、Official `0.472334 m`；轨迹斜率分别为 `0.003465`、`0.003486 m/frame`。逐帧曲线显示随时间持续分离。
- VERIFIED：M0 Raw/Official 的骨盆相对 FK 平均关节误差中位数分别为 `0.024369 m`、`0.023267 m`；同源参考为 `0.043824 m`，MBench 参考约 `5.77e-6 m`。因此 FK 绝对/相对误差必须区分参考数据族，不能单独归因于生成模型。
- VERIFIED：同源 20 条和 MBench 450 条参考动作的位置—速度残差及积分漂移均为浮点舍入量级；这表明当前 M0 的显著位置—速度矛盾主要在生成输出阶段出现，而不是 276 维前向差分构造普遍失效。
- VERIFIED：根平移速度残差中位数为 Raw `0.007275 m`、Official `0.003263 m`；根旋转单步残差为 Raw `0.316826°`、Official `0.198579°`；根旋转累计漂移约 `3.14°/3.20°`。
- PENDING：尚未据此修改表示或重新生成模型；下一步应先审阅报告中的 Raw/Official 配对差异和 FK 参考族差异，再决定是否注册一致化输出协议。

- 2026-08-14：以测试驱动方式新增显式 `vimogen-sample-noise-v1`：先得到缺少模块的失败测试，再实现逐样本确定性 CPU 噪声、缓存 JSON/张量双文件、哈希校验和回放；批大小 `1/2/4` 在噪声层与空速度 `FlowSampler` 层均逐位一致。完整阶段 0 测试为 `13 passed in 4.13s`，提交 `7faceca`。默认旧噪声路径和两个旧 M0 黄金目录保持不变；真实模型新噪声回归仍待用户批准。

- 2026-08-14：核对官方 ViMoGen 示例：仓库 `data_samples/example_archive.json` 提供与当前 M0 相同的4条提示词及文本嵌入，但不提供这4条提示词的官方生成动作或 MP4。官方项目页仅展示 Bodysurfing、Marching、Putting on Shoes、Squatting、Somersaulting、Juggling Balls 六条精选结果，不能作为 sample_000 的逐条基准。

- 2026-08-14：诊断 M0 `sample_000` 的文本—动作结果；右脚前段后移、骨盆下降与“坐下”一致，但末段未恢复到站立。`M0_raw` 与 `M0_official` 轨迹基本相同，问题不是官方平滑。当前渲染器使用固定相机矩阵，不读取提示词中的 `side view`；原始动作与对照视频位于服务器 `diagnostics/phase0/sample_000_raw_vs_official/`。

- 2026-08-13：完成原计划和优化方案审查，确认服务器真实代码与运行环境；创建项目记忆，准备阶段 0。
- 2026-08-13：在干净 Git 副本复现并定位 TM2M 形状错误；确认是缺少未提交的注意力回退修复，已同步修复并通过独立形状测试。
- 2026-08-13：定位并修复 PyTorch 2.7 与 torchgeometry 的布尔减法兼容问题；挂载并核验 SMPL-X/SMPL/MBench 资源；阶段 0 官方冒烟最终通过，提交 `35391cd`。
- 2026-08-13：用户反馈视频倒置/横躺。用同一条 MBench“下楼梯”动作分别走 PyTorch3D 默认 Rot6D 解码和 ViMoGen 原始交错布局解码，确认根因是布局不一致：项目使用前两列交错布局，当前 PyTorch3D 辅助函数按前两行解释；前者渲染为倒置，后者直立。不是 SMPL-X、SMPL 或 MBench 文件损坏。
- 2026-08-13：在干净副本修复 `motion_rep/rotation_transform.py`，恢复 ViMoGen 交错 Rot6D 编解码；新增 `tests/test_rotation_layout.py`。回归测试 `tests/test_rotation_layout.py tests/test_attention_fallback.py` 为 2 passed，修复提交 `c82fe40`。
- 2026-08-13：用修复后的解码器重新渲染原有四条 `(100,276)` 输出，视频位于服务器 `/root/autodl-tmp/vimogen_clean/stage0_smoke_rot6d_fixed/{0,1,2,3}/motion_gen_condition_on_text_depth_recover_velocity.mp4`。四条提示词分别是坐下、呼啦圈、捣土豆和爬梯子，并非普通“走路”；修复后人物姿态已直立，物体未渲染属于当前仅渲染人体网格的限制。
- 2026-08-13：按用户要求完成阶段 0 产物归档。正式动作、视频、抽帧和提示词统一迁移到 `/root/autodl-tmp/vimogen_clean/results/phase0/smoke_rot6d_fixed/`；旧错误视频、旋转布局对照和中间冒烟结果迁移到 `/root/autodl-tmp/vimogen_clean/diagnostics/phase0/`；本地同样建立 `results/`、`tests/`、`evaluation/`、`scripts/`、`diagnostics/` 分区。
- 2026-08-13：同步更新 `plan_optimized.md` 的文件结构，将原先笼统的 `artifacts/` 拆为正式结果、评估、测试产物和诊断目录，并冻结阶段/运行名/样本编号规则。
- 2026-08-14：将服务器端阶段 0 抽帧目录扁平化为 `results/phase0/smoke_rot6d_fixed/frames/sample_XXX_frame_YY.jpg`，与本地目录和项目记忆保持一致。
- 2026-08-14：完成 M0 黄金回归。旧实现实际批次 `z0` 为 `[4,100,276]`、`bfloat16`，SHA256=`29f1ac0bd7c474672e3fe45d7627fd2dee90f06ca6c9bf06b0c13807dbd3bdca`；`M0_raw` SHA256=`90e52cdd9686e75f88c6d8b23a48281d7c26dae924557b34fcb9c79a3227c827`，`M0_official` SHA256=`907c7ec997c8bf45aeef6e78d5c2fe640c21efe7945388094c240d835fc12fdb`；50步/shift=5.0日程哈希为 sigmas=`112da42580fbd48b3d39e2818cdb17276f938279a7995d9e9e79641eb63ff4f5`、timesteps=`70202b5d6f41214cfdcf674f883d03c294a62b771e3b0a05b8e15d4a88b35101`。新采样器与旧实现三组张量均 bitwise equal，最大绝对误差 0；冻结门槛 `atol=rtol=2e-2` 通过。
- 2026-08-14：新增 `sampling/flow_sampler.py`、外部 z0 接口、M0 raw/official 分界和回归测试；提交 `516c048`、`f0c7366`。全套测试命令共 8 项，结果 `8 passed`。本轮未实施新样本级噪声协议、guidance 或骨盆控制。

## 2026-08-14 阶段1表示与坐标校准首轮

- VERIFIED：在 `/root/autodl-tmp/vimogen_clean` 新增 `motion_rep/phase1.py`，把 276 维布局、SMPL-X 前 22 个关节顺序、A2/A3 主动集旋转切片、Rot6D 安全解码/重编码、SO(3) 左右乘编辑、前向差分、`T+1` 恢复、有效帧掩码和标准化边界集中为显式接口。未修改旧 M0 默认采样路径或旧结果目录。
- VERIFIED：新增 `tests/test_motion_representation_phase1.py`，覆盖 276 维索引断言、关节顺序（骨盆/左右髋/脊柱）、近共线 Rot6D 合法性和重编码、`+5°/+10°/-5°` 正负标定、前向差分与 `T+1` 回复、有效帧掩码、形状/dtype 和现有重建器边界。先出现缺失模块的红测，随后实现后为 `7 passed`。
- VERIFIED：阶段0相关回归命令共 `22 passed in 3.89s`，日志为 `tests/artifacts/phase1/phase1_pytest.log`。
- VERIFIED：只读诊断脚本 `scripts/phase1_representation_diagnostic.py` 对样本级 M0 batch4 的 raw/official 分别完成标准化反变换、Rot6D 正交性、速度差分和中性 SMPL-X 前22关节 FK 对照；汇总为 `diagnostics/phase1/m0_representation_audit/summary.json`。raw/official Rot6D 正交误差均小于 `5.4e-7`，说明旋转解码合法。
- VERIFIED：raw 显式关节与 FK 的物理单位误差 median=`0.0299 m`、p95=`0.1077 m`、max=`0.1822 m`；official median=`0.0299 m`、p95=`0.1071 m`、max=`0.1756 m`。这证明当前生成样本的显式关节位置与“直接用中性 SMPL-X 由旋转/FK 重建”并非逐位相同，且官方逐通道平滑不能消除该差异。
- VERIFIED：raw/official 关节速度一致性均值分别为 `0.00418/0.00179 m`，根平移速度为 `0.00310/0.00130 m`；根旋转相邻乘法的矩阵误差均值为 `0.00149/0.000884`。这些数值作为校准前审计证据，不是最终方法指标。
- PENDING/决策门：暂不无条件采用绝对 FK 重建。下一轮先用显式关节路径完成骨盆代理角的坐标/正负标定，同时保留 FK 作为诊断；只有在明确解释 canonicalization、SMPL-X 模板和根平移差异后，才决定绝对 FK 或增量残差主路线。不得把这次 FK 差异直接归因于模型文件损坏。
- 失败归档：诊断脚本首次直接运行时缺少项目根路径导致导入失败；修正为脚本自带根路径后成功，失败日志已归档在 `diagnostics/phase1/phase1_representation_failures/attempt_01_import_path.log`。

## 2026-08-14 阶段1角度与最终化第二轮

- VERIFIED：新增 `motion_rep/pelvis_angle.py`。当前明确记录 canonical 坐标 `x=左右、y=前进、z=向上`，并根据 SMPL-X 原生局部坐标和仓库的 SMPL-X→AMASS/DART 变换，将根 Rot6D 的候选局部前向轴设为 `+z`。这只是已证据支持的候选定义，角度语义仍未冻结。
- VERIFIED：代理角在行进坐标系中计算为 `atan2(前向轴的世界z分量, 水平投影范数)`；行进方向来自根平移速度，低于 `0.05 m/s`（20 fps 即 `0.0025 m/帧`）的帧被标记无效，方向可延续但不进入有效角度统计。合成 SO(3) 测试显示左乘 `+x` 的 `+5°/+10°/-5°` 分别恢复为同号角度，45°和180°偏航不改变零俯仰。
- VERIFIED：阶段0 M0 batch4 角度诊断结果写入 `results/phase1/pelvis_angle_calibration/angle_calibration.json`。官方样本中候选 `+z` 的有效帧水平倾角中位数约 `-3.41°`、标准差 `6.11°`，绝对行进对齐中位数 `0.789`；候选 `+y` 几乎竖直（倾角中位数 `84.17°`、对齐 `0.064`），候选 `+x` 对齐较弱（`0.603`）。这支持 `+z` 作为候选前向轴，但根姿态与行进方向并非完全一致，因此不把该诊断直接当作控制成功证据。
- VERIFIED：新增 `motion_rep/finalize.py` 的唯一最终化接口。输入为物理空间的 `[T+1,21,3,3]` 局部旋转、`[T+1,22,3]` 关节位置、`[T+1,3,3]` 根旋转、`[T+1,3]` 根平移（也支持批量）；先计算前向差分和根相对旋转，再输出 `[T,276]`，最后一帧仅作为最后一行速度的边界。有效掩码按相邻帧取与，标准化在几何重建后执行，无效行最终置零。
- VERIFIED：新增 `tests/test_phase1_angle_and_finalize.py`；包含角度方向、偏航不变性、T+1边界、标准化/dtype、批量形状，以及与旧 `collect_motion_rep_DART` 逐通道一致性测试。阶段1新增测试 `8 passed`；与阶段0回归合计 `30 passed in 4.00s`，日志 `tests/artifacts/phase1/phase1_pytest.log`。
- PENDING/决策门：代理角定义仍是“模型空间骨盆俯仰代理”，不是临床 ASIS–PSIS 角；绝对 FK 与增量残差路线仍均为 PENDING。尚未运行任何骨盆控制实验、guidance、M1/M2。

## Current continuation after phase1 second round

1. 继续只读审计 canonicalization、SMPL-X 模板轴和根姿态/行进方向不一致的来源；不得把 +z 候选直接升级为临床或控制终点。
2. 如需决定绝对 FK 或增量残差主路线，先补齐同一姿态流、同一模板和根平移的证据；在决策门通过前保持两条路线并列。
3. 可在阶段1范围内继续完善最终化边界、诊断和报告；不得开始 guidance、骨盆控制实验、M1/M2。

## 2026-08-14 阶段2评价系统与简单基线首轮

- VERIFIED：新增 `evaluation/phase2_metrics.py`，统一输出三类表示级指标：有效帧三元组上的二阶前向差分平滑度、显式关节与中性 SMPL-X FK 的米制欧氏误差、候选骨盆代理角摘要。每个结果带数值、样本数、单位和定义，避免把诊断量误写成官方指标。
- VERIFIED：指标清单明确 `official_fid=false`、`official_r_precision=false`；MBench 物理项只记为辅助项，要求后续若接入评估器时同时记录接触阈值、有效帧掩码和单位。本轮没有声称已有 FID、R-precision 或官方 MBench 分数。
- VERIFIED：新增 `motion_rep/baselines.py`：B0 不编辑但经过唯一 `T+1 -> T×276` 最终化；B1 只改根旋转 Rot6D、故意保留显式位置和速度冗余；B2 施加显式刚体 canonical 坐标变换后重新计算位置/速度并经过合法最终化。B2 当前是可审计的简单几何基线，不是绝对 FK 或增量残差路线的选择，`fk_route_decision` 仍为 `PENDING`。
- VERIFIED：先写红测（7 项导入失败），实现后 `tests/test_phase2_eval_baselines.py` 为 `8 passed`；与阶段0/阶段1回归合计 `38 passed in 3.91s`，日志为 `/root/autodl-tmp/vimogen_clean/tests/artifacts/phase2/phase2_pytest.log`。
- VERIFIED：在同一批样本级 M0 官方输出（4 条、`[4,100,276]`、相同文本和 z0、先反标准化到物理空间）上运行 B0/B1/B2，正式 delta=0 张量写入 `/root/autodl-tmp/vimogen_clean/results/phase2/baselines_official_batch4_delta0/`，共12个 `.pt` 文件；B0 与 B2 的每个样本哈希相同，证明 delta=0 下合法最终化路径一致。
- VERIFIED：正式基线诊断写入 `/root/autodl-tmp/vimogen_clean/diagnostics/phase2/baseline_metrics.json`，记录输入 SHA256=`a1abc5cf6a7743dc8a2551cf70467d87b0f90a1cb9f58d1345579fb379878745`、每个基线输出哈希、平滑度、FK差异和有效帧数。额外 `+5°` 仅作为 B1 故意不一致/B2 几何更新的诊断探针，未作为控制结果。
- VERIFIED：可复现脚本为 `/root/autodl-tmp/vimogen_clean/scripts/run_phase2_baselines.py`；实现哈希：`motion_rep/baselines.py`=`c815c0b9bf360b849444ea34b363f489e8362604fe3fb907cc23267b37a84658`，`evaluation/phase2_metrics.py`=`b6e99d5a3852c14859c6426835cfb42644c8df9ba43e13c5f1037e7a5c59fb98`，测试=`0afeb51896453490b9688edd18be2a81f87edbe80e5690a9f019ab53869e4ff8`，脚本=`27d9faaedba02e3c3d49909d1ea2cfe945823c145eba9ea8451c25588f27cacc`，诊断=`b9eb60de797ef113dabb0e309b324ba94c5900078493da1be9f28a3b60eb5202`，测试日志=`b46a6a7bdce689a31dc4e12ee12e0795e290615b19d8676b01489e71026c77b9`。
- PENDING：候选角度的临床/控制语义、绝对 FK 与增量残差主路线、MBench 官方物理评估器接入仍未决定；不得据此开始 guidance、骨盆控制主实验、M1/M2。

## 2026-08-14 阶段2决策门：合成正负与偏航探针

- VERIFIED：新增 `evaluation/phase2_probes.py`，把三个概念分开记录：根局部前向候选是 `R @ e_z`；canonical/world 编辑是左乘 `R_delta @ R`（`x=左右、y=前进约定、z=向上`）；行进 heading 只来自根平移速度的水平归一化，低于 `0.0025 m/帧` 的帧无效。没有把根姿态方向强行当作行进方向。
- VERIFIED：先写红测（3 项导入失败），实现后 `tests/test_phase2_decision_probe.py` 为 `3 passed`；与此前阶段0/1/2回归合计 `41 passed in 3.95s`，新增日志 `/root/autodl-tmp/vimogen_clean/tests/artifacts/phase2/phase2_decision_probe_pytest.log`。
- VERIFIED（合成中性姿态）：局部 `+z -> canonical +y`、heading=`+y` 时，左乘 canonical `+x` 的 `+5°`、`-5°` 分别得到 `+5.0000005°`、`-5.0000005°`；左乘 canonical `+z` 的 `45°`、`180°` 偏航均保持 `0°`。
- VERIFIED（M0 batch4 只读诊断）：使用与正式阶段2相同的 M0 official 输入和样本级 z0（输入 SHA256=`a1abc5cf6a7743dc8a2551cf70467d87b0f90a1cb9f58d1345579fb379878745`）。四样本中位角变化均值：B1 `+5° -> +4.8994°`、`-5° -> -4.9541°`；B2 分别 `+4.8839°`、`-4.8197°`。`yaw45` 的最大绝对中位变化 B1=`0`、B2=`1.2e-7°`；`yaw180` B1=`2.4e-7°`、B2=`9.5e-7°`。这些是候选模型空间诊断，不是控制成功结论。
- VERIFIED：通道更新关系符合基线定义：B1 的关节位置、关节速度、根平移和根平移速度变化恒为0；B2 同步改变位置/平移并由最终化重算速度。B0/B2 的 delta=0 正式结果未被覆盖。
- VERIFIED：显式关节-FK 误差仍为表示一致性问题而非模型损坏证据；四样本 delta=0 平均误差 B0/B2=`0.0954 m`、B1=`0.0869 m`，探针中刚体编辑后误差随姿态变化，尚不足以在绝对 FK 与增量残差之间作不可逆选择。
- VERIFIED：探针结果写入 `/root/autodl-tmp/vimogen_clean/results/phase2/decision_probe_v1/probe_outputs.pt`，诊断报告为 `/root/autodl-tmp/vimogen_clean/diagnostics/phase2/decision_probe_v1.json`，脚本为 `/root/autodl-tmp/vimogen_clean/scripts/run_phase2_decision_probe.py`。
- DECISION GATE：数学上的候选正负和偏航不变性探针通过（`VERIFIED_SYNTHETIC_ONLY` 与 `VERIFIED_SYNTHETIC_AND_M0_DIAGNOSTIC`）；临床/控制语义、绝对 FK/增量残差路线继续 `PENDING`。本轮没有触发需要用户作不可逆选择的阻塞，也没有开始 guidance、骨盆控制、M1/M2。

## 2026-08-14 样本级固定噪声阶段里程碑

- VERIFIED：用户已明确授权越过“新噪声语义”决策门；旧 M0 默认路径、旧黄金目录和原始服务器目录均未修改。
- VERIFIED：`train_eval_vimogen.py` 增加显式 `m0.noise_protocol: sample_v1` 适配；未配置该字段时仍使用旧批次生成器。入口脚本为 `scripts/run_m0_sample_noise_v1.py`。
- VERIFIED：真实模型已在样本级协议下完成批大小 1、2、4 三组独立运行，固定 50 步、shift=5.0、`denoising_strength=0.7`、BF16、seed=42、当前 PyTorch SDPA 回退。
- VERIFIED：为满足 plan_optimized.md 的“同一样本批大小无关”验收，样本级运行显式启用 `m0.batch_invariant: true`；FlowSampler 对每个样本走同一批大小为1的模型路径。旧 legacy_batch 路径不启用该选项。
- VERIFIED：批大小不变模式下，4 个样本的 z0、raw、official 跨 batch 1/2/4 均 bitwise equal；比较报告显示 `z0_bitwise_equal=true`、`outputs_within_tolerance=true`，`atol=rtol=2e-2`（实际输出差异为0）。
- VERIFIED：正式结果目录：`/root/autodl-tmp/vimogen_clean/results/phase0/m0_sample_noise_v1_batch1_invariant/`、`...batch2_invariant/`、`...batch4_invariant/`；每个目录含 `artifacts/batch_XXX/`、`noise_cache/`、`trainer/test_visualization/.../*.mp4`。
- VERIFIED：跨批大小报告：`/root/autodl-tmp/vimogen_clean/diagnostics/phase0/m0_sample_noise_v1_batch_compare_invariant.json`；运行日志：`diagnostics/phase0/m0_sample_noise_v1_batch{1,2,4}_invariant.log`。此前未启用批大小不变模式的运行保留在 `m0_sample_noise_v1_batch{1,2,4}/`，其输出差异（最大约0.082）作为 GPU 批处理数值差异诊断，不作为主结果。
- VERIFIED：新增/更新测试包括 `tests/test_m0_sample_noise_entrypoint.py`、`tests/test_flow_sampler.py`、`tests/test_sample_noise_protocol.py`；完整阶段0相关回归为 `16 passed`，日志 `tests/artifacts/phase0/sample_noise_protocol_v1/pytest_batch_invariant.log`。
- PENDING：样本级协议已完成并冻结；下一阶段按 plan_optimized.md 进入阶段1的角度、Rot6D 与 `T+1 -> T×276` 最终化校准。暂不开始 guidance、骨盆控制或 M1/M2。
- VERIFIED：总结果报告已建立并同步到本地 `RESULTS_REPORT.md` 与服务器 `/root/autodl-tmp/vimogen_clean/RESULTS_REPORT.md`；后续每个阶段完成后更新该报告和本记忆文件。
- VERIFIED：结果报告已扩充为阶段 0 总结（含冒烟视频、故障定位、M0 黄金回归、样本级噪声、限制和下一步），服务器文档提交为 `9a1d2ad`，报告 SHA256=`ad696a967a2580ea4b01c10a4980a44734a8c4c4aeab36a6197e5dbb6a2412d3`。
## 2026-08-14 Phase 3 M1 小规模原型

- VERIFIED：在保持阶段0 M0 与样本级 `z0` 协议不变的前提下，实现显式可选的 M1 近似清洁端点引导。公式为 `x0_hat = x_sigma - sigma * v_cfg`；钩子位于 CFG 速度合成之后、Euler 更新之前；模型参数不求梯度。
- VERIFIED：新增 `sampling/m1_guidance.py`、`sampling/flow_sampler.py` 可选 `m1_guidance` 参数和 `train_eval_vimogen.py` 的独立配置；M1 未启用时旧 M0 路径保持不变。标准化只在局部损失中反标准化，修正后重新标准化并回算速度。
- VERIFIED：M1 配置默认仍使用行进 heading；本轮四条非行走/停留片段的根平移通道为零，行进 heading 没有有效帧。因此 pilot 显式采用阶段2已验证的模型空间 `canonical_y` heading，并在 `m1_config.json` 中记录。这不是临床角度声明。
- VERIFIED：先红测后实现。M1/FlowSampler/配置回归在补充零目标严格空操作测试后为 `10 passed`；阶段0–3联合回归 `50 passed in 4.16s`，日志为 `/root/autodl-tmp/vimogen_clean/tests/artifacts/phase3/phase3_pytest_zero_noop_fix.log`。
- VERIFIED：成功 pilot 使用相同文本、seed=42、样本级 z0、50步和冻结 M0。正式目录：`results/phase3/m1_pilot/plus5_retry05/` 与 `results/phase3/m1_pilot/minus5_retry03/`；两组 z0 与阶段0 batch4 均逐位相同，M0 raw/official 也逐位相同。
- VERIFIED：评价脚本为 `/root/autodl-tmp/vimogen_clean/scripts/evaluate_m1_pilot.py`，指标为模型空间角度代理、二阶波形平滑度、显式关节—中性 SMPL-X FK 辅助误差、足部接触滑移/穿地启发式。绝对 FK 硬重建、官方 MBench 物理分数、FID、R-precision 和语义评估均未启用。
- VERIFIED：评价 JSON 为 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/m1_pilot_metrics.json`，完整清单为 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/m1_pilot_manifest.json`。官方输出的逐样本中位绝对角度误差：+5° 为 `1.794–1.824°`，-5° 为 `0.843–2.492°`；批内均值约 `1.50°/1.57°`，但 -5° 的最大样本超过 2°。
- VERIFIED：相对 M0 canonical 基线，官方 M1 FK 平均误差约 `0.0848–0.0900 m`，M0 约 `0.0956 m`；官方波形平滑度约 `0.02175–0.02188`，M0 约 `0.01826`；足部水平接触速度约左 `0.00389–0.00411 m/frame`、右 `0.00348–0.00349 m/frame`，需视为启发式而非物理结论。
- PENDING：本轮只能称为 M1 工程 pilot；按“逐样本不超过2°”的严格门槛，-5° sample_002 未通过。按批内中位数的宽松临时规则可继续调参，但完整 M1→M2 开发门槛仍为 `PENDING_M2_NOT_RUN`，不得开始 M2 或正式主实验。
- 失败试验均保留且写入 manifest：`plus5`（prefetch_factor/worker）、`plus5_retry01`（bool mask）、`plus5_retry02`（统计量切片）、`plus5_retry03/04`（FlowSampleResult 保存错误）、`minus5_retry01/02` 等，不纳入正式指标。

## Current continuation after phase3 M1 pilot

1. 保持 M0、样本级 z0、M1 代码和 pilot 结果可复现；先针对 -5° sample_002 的角度误差和足部滑移做只读诊断或超参数候选，不覆盖成功 pilot。
2. 在没有明确通过严格 M1 入口门槛前，不开始 M2、M1-K 或 guidance/骨盆控制主实验。
3. 继续保持绝对 FK 与增量残差路线为 PENDING；中性 SMPL-X FK 只作为辅助诊断。

- VERIFIED（补充）：M1 `target_delta_deg=0` 现严格返回未修正速度；该修复不改变 `+5°/-5°` pilot，也不改变 M0 默认路径。
- VERIFIED（提交）：零目标修复代码提交为 `650aff9`，结果报告同步提交为 `cf1363b`；服务器工作树仍只保留历史阶段产物目录未跟踪，不涉及原始目录。

## 2026-08-17 阶段3 M1 受控诊断

- VERIFIED：在不改变冻结 M0、样本级 `z0` 或既有 `m1_pilot` 目录的前提下，新增单变量诊断矩阵脚本 `scripts/run_m1_controlled_diagnosis.py` 与顺序运行脚本 `scripts/run_m1_controlled_matrix.sh`。配置变量分别为引导强度 `lambda_scale`、shift 后 `sigma` 窗口和端点修正 `max_correction_rms`；每项只改变一个变量。
- VERIFIED：先红后绿。新增 `tests/test_m1_controlled_diagnosis_config.py` 初次因脚本缺失为 `2 failed`；实现后配置测试为 `2 passed`。阶段3/M0 相关回归命令结果为 `16 passed in 4.30s`。
- VERIFIED：固定同一4条文本样本、seed=42、样本级 z0、50步、shift=5.0、denoising_strength=0.7、BF16、SDPA 和 `batch_invariant=true`，完成 `pilot_reference`、`strength_half`、`window_mid`、`rms_cap_half` 四组配置的 `+5°/-5°` 共8次真实模型运行。
- VERIFIED：正式结果位于 `/root/autodl-tmp/vimogen_clean/results/phase3/m1_controlled_diagnosis/*_retry02/`；指标位于 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/m1_controlled_diagnosis_metrics.json`；用户可读报告位于 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/M1_CONTROLLED_DIAGNOSIS_REPORT.md`。
- VERIFIED：8次运行的 `z0`、M0 raw、M0 official 均与阶段0 batch4 逐位一致；共同 z0 SHA256=`d2c026f951d7613c08937223d5e351d5081444102a0654d36153154c3228c10a`。M0 冻结未被污染。
- VERIFIED：当前 pilot 参考配置 `+5°/-5°` 平均中位绝对误差为 `1.504°/1.575°`，最大为 `1.824°/2.492°`，因此 `-5°` 仍未通过严格逐样本 `2°` 门槛。
- VERIFIED：`strength_half` 与 pilot reference 的 raw/official 输出逐位相同（最大差 `0`），说明当前 RMS cap=`0.05` 对引导强度变化形成饱和，单独减半 lambda 没有作用。
- VERIFIED：`window_mid`（sigma `[0.25,0.65]`）最大逐样本误差为 `1.729°/1.743°`；`rms_cap_half`（RMS cap=`0.025`）最大为 `1.284°/0.967°`。两者均满足当前 M1 入口门槛，状态为 `M1_ENTRY_GATE_CANDIDATE`，不是最终优选或 M2 授权。
- VERIFIED：资源记录写入每个运行的 `resource_metadata.json`；峰值 allocated 约 `9785 MiB`、reserved 约 `19046 MiB`，单次耗时约 `85–280 s`。官方 MBench/FID/R-precision/语义盲评仍未运行。
- VERIFIED：standalone torchrun 和未设置分布式环境的两次启动失败已归档在 `diagnostics/phase3/m1_controlled_diagnosis_failures/`，未计入正式结果。
- PENDING：尚未在 `window_mid` 与 `rms_cap_half` 中冻结最终 M1 开发集配置；不得开始 M2/M1-K/guidance 主实验。绝对 FK 仍只作辅助诊断，临床/解剖学角度语义仍不作结论。

## Current continuation after M1 controlled diagnosis

1. 比较 `window_mid` 与 `rms_cap_half` 的副作用并冻结一个 M1 候选；只允许有限、可审计的确认运行。
2. 更新 M1 入口门槛状态和结果报告；若候选保持逐样本 `≤2°`，再由主线决定是否开启 M1→M2 决策门。
3. 在明确授权前，不开始 M2-A2/A3、M1-K、源噪声优化或动作条件再生成。

## 2026-08-17 开发集候选准备

- VERIFIED：新增 `scripts/prepare_phase3_devset.py`，先写测试再实现；服务器固定环境下 `tests/test_prepare_phase3_devset.py` 为 `2 passed`。
- VERIFIED：脚本读取 `data/ViMoGen-228K/ViMoGen-228K.json`，按计划要求筛选步行主导文本、排除跑步/楼梯/坐下/爬行/跌倒或推搡/障碍/弯腰/跳跃及明显复杂动作，去除完全相同的规范化文本，并使用固定种子 `20260813` 选择五类各4条。
- VERIFIED：候选池规模为 straight_walk=`4534`、turning_walk=`2152`、speed_walk=`1138`、arms_while_walking=`1434`、stop_and_walk=`697`；候选总数20，正式协议为20条文本×3种子×3指令（0/+5/+10）=`180`个单元。
- VERIFIED：原始数据文件 SHA256=`5a65da2594f16c9cbf18957d87a2211d057b92b84766f2470642ba94574b9dc2`。
- PENDING：`candidate_v6` 仍是 `CANDIDATE_NOT_FROZEN`，目录为 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/devset_candidates/candidate_v6/`；candidate_v1–v5 作为筛选诊断历史保留。每条记录包含动作ID、文本、类别、匹配规则和两名审核人/裁决字段；必须完成两名独立审核和分歧裁决后才能冻结。
- IN PROGRESS：正式开发集未冻结前，不运行180个正式模型单元，不开始M1-K或M2。`window_mid` 仅作为符合原计划候选范围的优先配置；`rms_cap_half` 仍为超出计划固定RMS上限的探索性诊断。

## Current continuation after development-set candidate preparation

1. 人工复核 `candidate_v6/candidate_manifest.json`，记录每条文本的纳入/排除理由和近义重复处理；复核完成后另存不可覆盖的 `frozen_v1`，不得直接修改 candidate_v6。
2. 用冻结的20条文本完成阶段2 B0/B1/B2 开发基线（种子0/1/2，指令0/+5/+10），先验收0°空操作与剂量单调性。
3. 在开发集上按计划的强度/噪声窗口候选验证 M1，RMS上限保持0.05；只有开发集门槛通过后才实现M1-K，再进入M2-A2/A3。

## 2026-08-17 开发集协议矩阵

- VERIFIED：新增 `scripts/validate_phase3_devset.py`，验证候选/冻结状态、类别数量、ID和规范化文本唯一性，以及冻结状态下的人工审核字段。
- VERIFIED：新增 `tests/test_validate_phase3_devset.py`；候选生成和协议验证测试合计 `4 passed`。
- VERIFIED：全套项目回归为 `56 passed in 5.70s`，日志为 `tests/artifacts/phase3/devset_protocol_pytest.log`。
- VERIFIED：从 candidate_v5 生成180个单元的协议预览，路径为 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/devset_protocol/phase3_dev_matrix_preview_v1.json`；状态为 `PREVIEW_NOT_FORMAL`，`do_not_run_until_manifest_frozen=true`。
- FROZEN INVARIANT：矩阵固定方法集合为 `M0_official/B0/B1/B2/M1`，指令为 `0°/+5°/+10°`，种子为 `0/1/2`，每个单元要求相同的样本级 `vimogen-sample-noise-v1` 初始噪声键。

## 2026-08-17 候选人工审核表

- VERIFIED：新增 `scripts/render_phase3_devset_review.py`，从 candidate_v6 生成可读审核表；对应测试使候选/矩阵/审核工具专测达到 `5 passed`。
- VERIFIED：审核表路径为 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/devset_review/candidate_v6_review.md`。
- PENDING：审核表仍需两名独立审核人填写 `KEEP/DROP` 和原因，并由第三人处理分歧；当前不能把 candidate_v6 改名或标记为正式冻结集。

## 2026-08-18 reviewer1 单评审覆盖与开发集冻结

- VERIFIED：用户明确要求只采用 `diagnostics/phase3/candidate_v6_review_reviewer1.md`，忽略 reviewer2。本轮因此不是双评审冻结，而是明确记录的 `FROZEN_SINGLE_REVIEW_OVERRIDE`；不得在论文或报告中写成“两名审核人均完成”。
- VERIFIED：candidate_v6 原目录保持 `CANDIDATE_NOT_FROZEN`，未被修改。首次自动替换尝试因重复选择 DROP 的 ID `14567`、以及选择类别不纯的 straight_walk 文本而作废，记录在 `diagnostics/phase3/devset_frozen/frozen_v1_reviewer1/INVALID_AUTO_SELECTION.md`。
- VERIFIED：修正后的不可覆盖冻结清单为 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/devset_frozen/frozen_v2_reviewer1/frozen_manifest.json`，状态 `FROZEN_SINGLE_REVIEW_OVERRIDE`，20条、五类各4条。替换 ID 为 `1933`、`14053`、`48788`、`56467`；清单 SHA256=`5ad55b3565d467e8ae7b195eed9f8d7617477ad9dc22ed74bcabeb405242c628`。
- VERIFIED：reviewer1 证据文件 SHA256=`c6ab772328ebf57ecd0fa1cef24a5a511f855d43048764d1bfca3650650e4ea5`；原始 `ViMoGen-228K.json` SHA256 仍为 `5a65da2594f16c9cbf18957d87a2211d057b92b84766f2470642ba94574b9dc2`。
- VERIFIED：正式矩阵 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/devset_protocol/phase3_dev_matrix_frozen_v2_reviewer1.json` 状态 `FROZEN_MATRIX_READY`，180个单元（20文本×3种子×0/+5/+10），SHA256=`f06cf15f36611cbe972ae96ddd2909cd42386fc4bccc6025c0cb05a1737c2d54`。
- VERIFIED：纯文本输入清单和 T5 嵌入审计通过，输入审计 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/devset_inputs/frozen_v2_reviewer1/input_audit.json`，20条、每个嵌入二维且末维4096、无参考动作条件，SHA256=`cebdb61f858a57040fb3a3f7a4a758a081ead7e19b6452cf053666bad6136b9c`。
- VERIFIED：阶段2 B0/B1/B2 开发基线已完成。M0 使用 seed 0/1/2、sample_v1、batch_invariant、50步/shift=5/denoising_strength=0.7/SDPA；M0 运行目录 `/root/autodl-tmp/vimogen_clean/results/phase3/devset_baselines/frozen_v2_reviewer1/m0_retry01/`，离线基线目录 `/root/autodl-tmp/vimogen_clean/results/phase3/devset_baselines/frozen_v2_reviewer1/baselines_v1/`。
- VERIFIED：基线报告 SHA256=`a4db4c8a34f3b872d1032d25cf8603226be8ab25d6c6f2661a070c7ba5580990`，共记录 780 条输出记录（M0 raw/official 审计 + B0/B1/B2 的180个正式单元）。B0 的三个命令标签逐样本哈希相同；B0 与 B2 在 `0°` 下 60/60 个样本哈希相同。B1 仍是故意不一致诊断基线，不纳入足滑/自然度排名。
- VERIFIED：冻结相关工具和完整回归均通过；当前完整回归为 `59 passed in 4.45s`，新增输入/运行器专测为 `7 passed`。直接调用 `train_eval.main` 的一次启动失败仅因未设置分布式环境，已保留为失败记录；正确的单卡 `torch.distributed.run` 路径成功。
- PENDING：M1 尚未在 frozen_v2 开发集运行。按计划优先冻结已在受控诊断中通过、且仍在预注册范围内的 `window_mid`（sigma `[0.25,0.65]`、RMS cap `0.05`）；`rms_cap_half` 虽通过受控诊断，但属于超出预注册 RMS 上限的探索配置，暂不作为主线候选。M1-K/M2 仍未启动。



## 2026-08-18 reviewer1 M1 window_mid 开发集评估

- VERIFIED：仅按用户指定的 reviewer1 清单，在 `frozen_v2_reviewer1` 的20条纯文本开发集上完成 M1 `window_mid`；协议为 sigma `[0.25,0.65]`、RMS cap `0.05`、`canonical_y`、seed `0/1/2`、目标 `+5°/+10°`。
- VERIFIED：评估 JSON 为服务器 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/m1_devset_window_mid_metrics.json`；可读报告为 `diagnostics/phase3/M1_DEVSET_WINDOW_MID_REPORT.md`。120个单元中118个通过逐单元中位绝对误差≤2°，2个失败：`seed_001/+10°/34122=5.3027°`、`seed_002/+10°/14796=3.5040°`；总体均值 `0.9256°`，入口状态 `M1_ENTRY_GATE_FAILED`。
- VERIFIED：六组运行的 z0、M0 raw、M0 official 均逐位等于对应 M0 基线；失败不是噪声错配、M0漂移或输入审计错误。官方 FID、R-precision、MBench物理指标和语义盲评未运行。
- BLOCKED：在失败样本只读诊断完成前，不启动 M2、M1-K 或新的同开发集调参；失败样本不因结果不佳而排除，只有确认与方法无关的系统故障才允许按规则重跑。

## 2026-08-18 M1 失败单元只读诊断

- VERIFIED：诊断脚本 `scripts/diagnose_m1_devset_failures.py` 只读取既有张量和视频，不重跑模型；专测 `3 passed`（含评估器测试）。
- VERIFIED：两个失败均为 `turning_walk` 的 `+10°` 样本—种子交互：seed1/sample34122 官方误差5.3027°、实际中位变化6.9379°；seed2/sample14796 官方误差3.5040°、实际中位变化6.8928°。两者均100帧有效，M1 raw 与 official 误差接近，排除了官方平滑作为根因。
- VERIFIED：同类其他种子可通过（+10°下 sample34122 seed0误差0.450°，sample14796 seed0误差0.629°），因此不能归结为所有转弯动作失败；当前最稳妥表述是“存在样本—种子交互导致的欠校正”。
- BLOCKED：未发现与方法无关的系统故障；不重跑、不排除、不在同一开发集事后调参。失败诊断 JSON 为服务器 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/m1_devset_failure_diagnostics_v2.json`，可读报告为 `diagnostics/phase3/M1_DEVSET_FAILURE_DIAGNOSTICS.md`。

## 2026-08-18 转弯样本角度定义与欠校正原因审计

- VERIFIED：独立审计只读取冻结的 M0/B0/M1 `window_mid` 张量、输入文本和既有视频；没有重跑模型、调参、改写主指标或排除样本。脚本为 `scripts/audit_turning_m1_angles.py`，专测 `2 passed`，完整回归 `69 passed in 8.54s`。
- VERIFIED：`canonical_y` 与 `travel` 在共同有效帧上的代理角最大差约 `3.8×10^-6°`；水平偏航本身不是角度公式的主要错误来源，travel 主要影响低速帧掩码。
- VERIFIED：B0 会执行 `T+1 → T×276` 合法最终化，不能视作 M0 raw 的简单复制。24条 turning 记录中 M0 raw 与 B0 的角度中位差中位数为 `0.818°`、最大 `4.103°`。两条失败样本的 M1 端点速度—位置不一致，提示表示边界可能参与欠校正。
- VERIFIED：将既有 M1 official 事后重新经过 B0 只作为派生探针，两条失败误差约降至 `2.359°/1.198°`；这不能替换正式主指标，但说明下一版必须统一所有方法的合法最终化边界。
- BLOCKED：当前不能据此宣布 M1 通过；正式结果继续保持 `M1_ENTRY_GATE_FAILED`。后续若继续开发，应先注册“统一最终化边界”的新 M1 协议，并使用新的留出集验证，不能在 reviewer1 冻结集上调参。

## 2026-08-18 M1 unified finalization v1

- VERIFIED：新增 `motion_rep/unified_finalizer.py`，统一物理空间 `T+1 → T×276` 边界；显式声明根锚点、速度重算、mask 和标准化边界。协议文件为服务器 `diagnostics/phase3/devset_protocol/m1_unified_finalize_v1.json`。
- VERIFIED：统一 finalizer 与离线运行器专项测试 `8 passed`；完整工程回归 `77 passed in 4.68s`。B0 旧实现与 common finalizer 的最大绝对差约 `3.8×10^-6`，说明无编辑边界一致。
- VERIFIED：离线复用已有 M0/M1 张量完成 before/after 对照，未重新采样。旧边界120单元中2个失败、最大5.3027°；统一边界14个失败、最大3.2245°，平均误差由0.9256°变为1.1116°。
- BLOCKED：统一最终化解决了边界公平性问题，但没有使 M1 通过严格入口门槛，状态为 `M1_UNIFIED_FINALIZE_ENTRY_GATE_FAILED`。不得启动 M2/M1-K；若继续，必须注册新的算法协议并使用留出集，不能在 reviewer1 冻结集调参。
- VERIFIED：完整逐样本 B0/B1/B2/M1 离线矩阵保存在服务器 `diagnostics/phase3/m1_unified_finalize_v1/` 与 `results/phase3/m1_unified_finalize_v1/`；它是补充审计，不覆盖主评价 JSON。

## 2026-08-18 M1 冗余通道一致性审计与下一版边界探针

- VERIFIED：新增只读诊断脚本 `scripts/diagnose_m1_channel_consistency.py`，对 reviewer1 冻结开发集现有120个 M1 单元检查显式姿态与速度通道是否互相重建一致；没有重跑模型，也没有覆盖旧结果。诊断 JSON 为 `diagnostics/phase3/m1_channel_consistency_reviewer1_v1.json`。
- VERIFIED：M1 official 的跨记录中位一致性误差为：显式关节位置约 `0.0478 m`（最大记录中位 `0.0894 m`），根平移约 `0.0444 m`（最大记录中位 `0.1016 m`），根旋转约 `3.09°`（最大记录中位 `8.77°`）。这证实 M1 输出同时包含“直接姿态视图”和“速度积分视图”，两者不能无声明地混用。
- VERIFIED：新增 `motion_rep/consistent_finalizer.py`，采用“直接姿态权威、末帧恒速度外推、重新计算全部速度通道”的独立候选边界；先行回归 `3 passed`。它不改变旧 `unified_finalizer.py`、旧 M0/M1 结果或默认采样器。
- VERIFIED：在同一 reviewer1 数据上，直接姿态权威的 B0/M1 配对为 `34/120` 超过2°，最大 `4.9347°`、平均 `1.6872°`；因此它不能被宣称为通过，也不应在冻结集上继续调参。
- VERIFIED：新增边界组合对照 `scripts/compare_m1_boundary_effects.py` 与 `diagnostics/phase3/m1_boundary_effects_reviewer1_v1.json`。五种组合结果为：速度—速度 `14/120` 失败、直接—直接 `34/120` 失败、旧B0—直接M1 `2/120` 失败、速度B0—直接M1 `2/120` 失败、直接B0—速度M1 `53/120` 失败。该对照证明原先的2/120结果混合了不同的通道权威，不能作为统一边界的证据。
- BLOCKED：M1 仍未通过入口门槛；当前不能启动 M2/M1-K，也不能把“换最终化后失败数变化”当作模型改进。下一步只能注册一个明确的通道权威协议，在新的留出集上验证；reviewer1 冻结集只作审计，不再调参。

## 2026-08-18 M1-v2 速度权威采样内一致化

- VERIFIED：新增显式可选协议 `consistency_mode=velocity_authoritative_v2`，代码提交为 `5a45eef2954b209aeae40055a1935778ee19db0f`。默认仍为 `legacy`；完全缺少 `m1` 配置时保持禁用，旧 M0 和旧 M1 默认语义未改变。
- VERIFIED：每个启用采样步保留引导前 `x0` 作为速度权威，从其速度通道恢复物理空间 `T+1` PoseStream；用 `R_guided_direct @ R_authority_direct^T` 得到逐帧左乘根旋转增量，隐藏边界沿用最后一个有效增量，再统一重算关节、根旋转和根平移三类速度并重新标准化。关节位置和根平移不随根旋转做刚体变化，因此本轮不是 M1-K/M2，也不宣称根—关节绝对 FK 一致。
- VERIFIED：严格测试驱动开发。改动前全回归 `85 passed`；最终 M1-v2 专项 `13 passed`，覆盖非交换 SO(3)、隐藏 `T+1`、三类速度、标准化、补齐行、关闭/零目标空操作、批大小不变、OmegaConf、CPU/CUDA BF16 自动混合精度；最终全回归 `98 passed in 5.72s`。日志为 `tests/artifacts/phase3/m1_v2_pytest.log`，SHA256=`22b0c9073c03801ade8815e8e1b1ec0a8531add3b3f337501872d7e07069d8b1`。
- VERIFIED：代码审查发现并修复两项高风险问题：早期 v1 错把引导后直接根姿态当成权威；外层自动混合精度会把 `.float()` 后的矩阵运算再次降为 BF16。最终版改为双输入“引导前速度权威 + 引导根旋转增量”，并在局部几何图显式关闭自动混合精度。
- VERIFIED：只读取历史 4 样本 `+5°/-5°` M0/M1 张量完成离线兼容性 smoke；正式 v4 位于 `results/phase3/m1_v2_pilot/offline_smoke_v4/`，诊断为 `diagnostics/phase3/m1_v2/offline_smoke_v4.json`（SHA256=`ecacc183c0b430c4dc1385def56c94fae71d31c92f81e9ec1207c6ed6e32700a`）。raw 处理前关节/根平移/根旋转中位不一致约为 `0.0476–0.0483 m`、`0.0321–0.0326 m`、`2.78–2.96°`；处理后约为 `3.73e-9 m / 0 / 0°`。
- VERIFIED：v1–v3 非正式版本已从 `results/` 移到 `diagnostics/phase3/m1_v2/attempts/`，未删除；`results/phase3/m1_v2_pilot/` 只保留正式离线 v4。完整报告为 `diagnostics/phase3/m1_v2/M1_V2_IMPLEMENTATION_REPORT.md` 和 `m1_v2_implementation_report.json`。
- PENDING：离线 smoke 没有重跑 ViMoGen，不等价于每个流步骤运行 M1-v2，也未生成 MP4、未评价角度门槛或自然度。状态为 `PENDING_REAL_MODEL_PILOT`；不得据此宣布 M1 通过，不启动 M1-K/M2，不在 reviewer1 冻结集调参。

## Current continuation after M1-v2 implementation

1. 只用旧 4 样本运行一次真实 M1-v2 pilot：seed=42、sample_v1 z0、batch_invariant、50步、shift=5.0、denoising_strength=0.7、BF16、SDPA、sigma `[0.25,0.65]`、RMS cap `0.05`；结果必须写新且不可覆盖目录。
2. 先核对 M0/z0 是否与冻结基线逐位一致，再评价 M1-v2 的三类一致性和模型空间角度误差；离线 v4 只作输入兼容性证据。
3. 真实 4 样本 pilot 通过工程门后，冻结新的留出集验证 M1-v2；reviewer1 集不再调参。入口门槛明确前不开始 M1-K/M2。

## 2026-08-18 M1-v2 真实 pilot handoff

- IN PROGRESS：已将下一阶段交接至独立任务窗口 `/root/phase3_m1v2_real_pilot`。任务只运行旧4样本的真实 M1-v2 pilot，不运行新留出集、不在 reviewer1 冻结集调参、不启动 M1-K/M2。
- FROZEN PROTOCOL：seed=42、sample_v1 样本级 z0、batch_invariant、50步、shift=5.0、denoising_strength=0.7、BF16、SDPA、sigma `[0.25,0.65]`、RMS cap=0.05、canonical_y、目标 `+5°/-5°`、`consistency_mode=velocity_authoritative_v2`。
- ACCEPTANCE：新运行的 z0/M0 raw/official 必须与冻结基线逐位一致；结果写入新目录 `results/phase3/m1_v2_pilot/`，诊断写入 `diagnostics/phase3/m1_v2/real_pilot01/`；完成后更新本文件和 `RESULTS_REPORT.md`，明确工程门状态。真实 pilot 未完成前，不得宣称 M1 通过。

## 2026-08-22 M1-v2 pilot partial result and connection blocker

- VERIFIED：独立任务已完成 `+5°` 条件的真实模型运行。结果目录为 `/root/autodl-tmp/vimogen_clean/results/phase3/m1_v2_pilot/plus5_realpilot01/`，包含 M0/M1 张量、配置和4个 MP4。样本级 z0 与冻结缓存逐样本 bitwise equal；M0 raw/official 与阶段0黄金结果逐位一致。
- BLOCKED：`-5°` 条件尚未运行，因此四样本 pilot 未完成，尚不能评价 M1-v2 入口或自然度。按 `connect_server.py` 连续重试均失败：`NoValidConnectionsError`，TCP `connect.westd.seetacloud.com:27750` 不可达；未改变协议、未运行留出集、未修改原始目录。
- Immediate continuation：服务器恢复后先运行 `-5°`，再核对两方向结果并生成 `diagnostics/phase3/m1_v2/real_pilot01/` 的完整诊断；若服务器持续不可达，保留当前部分结果并等待恢复。

## 2026-08-22 M1-v2 real pilot completed

- VERIFIED：使用更新后的 `connect_server.py` 成功连接服务器；`+5°` 和 `-5°` 两个方向均完成真实模型运行，未覆盖旧目录。结果分别为 `/root/autodl-tmp/vimogen_clean/results/phase3/m1_v2_pilot/plus5_realpilot01/` 与 `minus5_realpilot01/`，每个方向包含 M0/M1 张量、配置和4个 MP4。
- VERIFIED：两方向的 z0、M0 raw、M0 official 都与冻结阶段0基线逐位一致。共同哈希：z0=`ca88081027e610a091033e81049a706bc9038199451e9a9c75f14dd41a3c87da`，M0 raw=`b1459943fabdc8d303c278c34e861589abf4e76675243baafead081ca5e5ef82`，M0 official=`a1abc5cf6a7743dc8a2551cf70467d87b0f90a1cb9f58d1345579fb379878745`。
- VERIFIED：诊断脚本和测试已完成；`tests/test_evaluate_m1_v2_real_pilot.py` 为 `2 passed`。报告和指标位于 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/m1_v2/real_pilot01/REPORT.md` 与 `metrics.json`。
- VERIFIED：官方输出的模型空间代理角逐样本中位绝对误差为：`+5°` `[0.9701, 1.7000, 0.6862, 0.4105]°`，`-5°` `[0.5125, 0.6270, 0.6395, 1.2495]°`；四样本 pilot 的角度结果均低于2°，但这不是新留出集结论。
- PENDING：通道一致性仍被诊断标记为 FAIL。官方输出平均中位残差约为 joint `0.0037/0.0035 m`、root translation `0.0029/0.0028 m`、root rotation `0.20/0.19°`（+5/-5）；raw输出更高，说明采样内一致化显著降低但没有在最终输出上达到严格零残差。M1-v2 状态仍为 `ENGINEERING_PILOT_ONLY`，不能据此宣称入口通过或立即启动大留出集。

## 2026-08-22 M0 baseline channel-consistency audit

- VERIFIED：使用与 M1-v2 相同的 `channel_consistency` 公式，只读检查冻结 M0 batch4。未加 M1 的 M0 raw 已存在非零残差：跨4样本中位数约为 joint `0.0081 m`、root translation `0.0059 m`、root rotation `0.245°`；M0 official 约为 joint `0.0033 m`、root translation `0.0026 m`、root rotation `0.137°`。
- INTERPRETATION：通道不一致是 ViMoGen 原始276维预测/逐通道后处理的既有表示问题，不是本项目修改某个局部后才首次产生。M1-v2 会改变残差大小和角度轨迹，但不能把全部残差归因于 M1；M0、M1 raw、M1 official 必须使用相同诊断并分别报告。
- PENDING：M0 official 的逐通道平滑会降低但不保证一致性；若要达到严格一致，需要在最终输出后重新恢复 `T+1` 姿态并重算速度，或把该指标定位为表示审计而非 M1 特有失败门槛。当前不改变冻结 M0，继续先完成 M1-v2 残差来源审计。

## 2026-08-22 M1-v2 逐步残差来源追踪

- VERIFIED：为 `FlowSampler` 增加默认关闭的可选逐步 trace；关闭时旧路径输出保持逐位不变。trace 记录每个采样步的 `sigma`、`x_sigma`、模型速度、`x0_hat`、引导端点、协调后端点、修正速度、Euler 后状态和下一步模型端点。代码涉及 `sampling/flow_sampler.py`、`sampling/m1_guidance.py`、`train_eval_vimogen.py`。
- VERIFIED：新增测试 `tests/test_m1_v2_residual_trace.py` 和 `tests/test_m1_v2_trace.py`；针对性测试 22 项通过，最终全回归 `107 passed in 5.63s`。最终日志为 `tests/artifacts/phase3/m1_v2_residual_trace_full_final.log`，SHA256=`6fc2ac7aa602cf198c9d223e29e396e0f4462e06b09cbc8ae926b5d1303930ef`。
- VERIFIED：使用固定 sample_v1 噪声、seed=42、50步、shift=5.0、denoising_strength=0.7、BF16、SDPA、M1-v2 `sigma [0.25,0.65]`/RMS cap=0.05，运行旧4条样本的 `+5°` 追踪；结果写入新目录 `results/phase3/m1_v2_residual_trace/sample_000_plus5_v1/`，没有覆盖旧 pilot。每个 batch 保存 `m1_trace.pt`，并生成 MP4。
- VERIFIED：端点重构误差最大约 `7.6e-6`（标准化空间），同一修正速度经过 Euler 后的保持误差最大约 `7.6e-6`；均为浮点舍入量级，排除“协调接口或 Euler 更新是主要残差来源”。
- VERIFIED：下一步模型重新预测的端点差异最大为 `23.4–66.4`（标准化空间；各样本中位约 `0.018–0.025`），显著大于前两项，说明主要残差来源是模型在 Euler 后再次独立预测全部276个通道，而不是 M1-v2 端点重构。
- VERIFIED：最终 raw→official 逐通道平滑仍产生可见变化（标准化空间中位约 `0.0098–0.0124`），并能降低但不能消除通道残差；追踪后 raw/official 的物理通道一致性仍需作为表示审计报告，不能改写冻结 M0。
- VERIFIED：诊断脚本与报告：`scripts/run_m1_v2_residual_trace.py`、`scripts/evaluate_m1_v2_residual_trace.py`、`diagnostics/phase3/m1_v2/residual_trace/sample_000_plus5_v1.json`、`diagnostics/phase3/m1_v2/residual_trace/M1_V2_RESIDUAL_TRACE.md`。关键哈希：评估 JSON=`93bf735e7383246b231f8e31e20774f33ff92ea21497fabe7650161961609b3d`，报告=`b833a46dc4d4c95ee22ea39742078fa02cc817a699772780474a28781ef75e98`。
- PENDING：本轮仅追踪 `+5°` 旧4样本，尚未追踪 `-5°`，也未运行新留出集；不据此调参、不启动 M1-K/M2。下一步优先把“模型重预测是主导来源”写入 M1-v2 方法边界，并决定是否需要独立 `-5°` 复核；若复核一致，再注册新留出集协议。
## 2026-08-22：M1-v2 逐步追踪完成

- 服务器 clean 副本已加入默认关闭的 `trace_enabled`/`M1TraceRecorder`，并保存每个采样步的 `x_sigma`、CFG 速度、`x0_hat`、`x0_guided`、`x0_reconciled`、修正速度、`x_next`、下一步模型 `x0`。
- 追踪专项 5 项、全量测试 107 项通过；关闭追踪时与既有输出逐位一致。
- 正负 trace 产物：`/root/autodl-tmp/vimogen_clean/results/phase3/m1_v2_residual_trace/sample_000_plus5_v3/`、`sample_000_minus5_v1/`。
- Euler 重建误差为 0；下一步模型重预测漂移标准化空间 RMS 中位数约 0.041–0.104，最大约 0.636–1.312；一致性投影变化约 0.21–1.04。结论：Euler 不是当前主要未知源，模型重预测漂移与原始表示残差应分开报告。
- 仍为 `PENDING`：未运行新留出集，不开始 M1-K/M2；先冻结残差定义和验收门槛。
## 2026-08-22：相对 M0 门槛与非转弯留出集候选

- 新增相对 M0 的三类额外残差指标：关节位置、根平移各允许 `+0.001 m`，根旋转允许 `+0.1°`。
- 生成非转弯留出集候选：服务器 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/devset_holdout_candidates/nonturning_v9/`，20 条文本、4 类各5条、0条转弯、0个缺失动作文件，且不与 reviewer1 开发集重复。
- 候选尚未冻结、尚未运行模型；全量测试 `111 passed`。冻结后才运行 M0/M1 配对验证。

## 2026-08-22：留出集候选规则修正（12989）

- 用户复核发现旧候选 `nonturning_v9` 的 `id=12989` 文本为 `starting with left foot, then walking four steps`，没有停止动作；将其归入 `stop_and_walk` 是筛选规则误判，不是动作数据本身的问题。
- 已收紧 `stop_and_walk` 规则：必须出现明确的 `stop/stopping/stopped/pause/paused/halt` 等停止或暂停词；仅有 `start/starting/begin` 不再满足条件。
- 新候选清单位于服务器 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/devset_holdout_candidates/nonturning_v10/`。共20条、四类各5条，`12989` 已移除；清单仍为 `HOLDOUT_CANDIDATE_NOT_FROZEN`，需人工确认后才可用于正式模型验证。旧 v9 保留作历史诊断，不再作为正式候选。
- 修正规则后专项测试与全量回归均通过：`111 passed`；日志为服务器 `/root/autodl-tmp/vimogen_clean/tests/artifacts/phase3/m1_gate_and_holdout_full_v2.log`。

## 2026-08-22：非转弯留出集已冻结并完成 M1-v2 验证

- VERIFIED：用户确认 `nonturning_v10` 候选集通过，已冻结为 `FROZEN_SINGLE_REVIEW_OVERRIDE`。20 条文本、四类各5条，排除转弯样本；`12989` 不在清单中。
- VERIFIED：文本输入与嵌入审计通过，20/20 条为纯文本条件，输入哈希为 `705ef8fae0538e7e8e6b50e146ffe7a851fe9d934223e261c5cf3cf66b2e1cf2`。
- VERIFIED：M0 三个种子与 M1 六个条件（seed 0/1/2 × +5/+10）全部完成；M1 中保存的 z0、M0 raw、M0 official 均与对应 M0 基线逐位一致。
- VERIFIED：120 个模型空间代理角单元的最大中位绝对误差为 `1.8007°`、平均 `0.8563°`，严格 `2°` 入口门通过：`M1_ENTRY_GATE_PASSED`。
- VERIFIED：相对 M0 的通道额外残差仍不是所有单元都低于固定工程上限（关节/根平移各 `+0.001 m`，根旋转 `+0.1°`）；保留为表示审计，不宣称通道严格一致。FK 仅作辅助指标，不做硬重建。
- 详细报告：服务器 `/root/autodl-tmp/vimogen_clean/diagnostics/phase3/nonturning_v10_user_approved/M1_NON_TURNING_HOLDOUT_REPORT.md`。结果位于服务器 `results/phase3/devset_baselines/nonturning_v10_user_approved/m0_v2/` 和 `results/phase3/devset_m1/nonturning_v10_user_approved/window_mid_v1/`。
- PENDING：入口角度门通过不等于语义或自然度充分验证；当前不启动 M1-K/M2，不在该留出集上继续调参。

## 2026-08-22：nonturning_v10 通道额外残差重新定义与定位

- VERIFIED：新增只读诊断 `scripts/diagnose_nonturning_v10_residuals.py`，把真正的相邻帧局部通道残差与从首帧积分整段后的累计轨迹漂移分开。新脚本对旧配对报告的累计数值逐值复现，最大绝对差为 `0.0`。
- VERIFIED：在 `nonturning_v10_user_approved` 的 120 个配对单元上，M1 official 相对 M0 的局部额外残差三通道均为 `0/120` 失败；最大额外值为关节 `0.000299 m`、根平移 `0.000691 m`、根旋转 `0.03835°`。旧报告的 `44/120`、`40/120`、`21/120` 失败实际对应累计积分漂移，不应继续称为速度残差。
- VERIFIED：raw 局部残差仅根平移有 `10/120` 失败，分散在 8 个样本中；只有 `12581`、`13016` 各重复 2 次，且只有 `13016/seed2` 跨 `+5°/+10°` 重复。类别、种子和角度的探索性关联均未达到 `p<0.05`，不能归因为固定 TXT 或固定动作类别，也不应删除文本或重抽留出集。
- VERIFIED：当前视频与 MBench 路径均设置 `recover_from_velocity=True`，实际下游以速度恢复轨迹为权威。旧累计漂移仍是有意义的下游风险，但必须作为独立指标解释。
- VERIFIED：新增测试 `tests/test_nonturning_residual_diagnosis.py`，专项 `2 passed`，全项目 `113 passed in 6.00s`。报告为 `diagnostics/phase3/nonturning_v10_user_approved/M1_RESIDUAL_DIAGNOSIS_V1.md`，机器报告为 `m1_residual_local_vs_accumulated_v1.json`。
- PENDING：不改变已冻结 M1-v2。下一候选应是独立 M1-v3 离线终端同步：official 平滑后保留速度通道和首帧锚点，用速度恢复的轨迹同步直接关节位置、根平移与根旋转；M0/M1 同接口、新目录、不覆盖旧结果。先验证 99 个可观测相邻帧残差接近数值精度、速度通道不变、渲染路径不变，并重新检查每单元角度误差 `<=2°`，通过后才决定是否成为正式协议。

## 2026-08-22：M1-v3 速度权威终端同步候选

- VERIFIED：新增 `motion_rep/m1_v3_terminal.py`。M1-v3 在 official 平滑后保留首帧锚点和原始速度通道，使用速度恢复后续直接关节位置、根平移和根旋转；速度通道与局部身体旋转逐位保持不变。它是离线候选，不接入默认采样、不覆盖 M1-v2。
- VERIFIED：红测先因缺少模块失败；实现后专项测试通过。一次批量切片测试错误已修正，最终专项测试 `3 passed`。
- VERIFIED：候选结果位于服务器 `results/phase3/m1_v3_terminal_sync/nonturning_v10_user_approved/v3/`。120 个单元的局部和累计额外通道残差均为 0 失败；速度通道最大绝对变化为 0；速度恢复的关节、根平移和根旋转轨迹逐位不变。
- VERIFIED：与冻结 M1 相同的 `canonical_y` 角度门复核中，M1-v3 最大中位绝对误差 `3.4102°`、平均 `1.1478°`，通过 `103/120`，未通过严格 `<=2°` 门。冻结 M1-v2 为 `1.8007°` 最大、`0.8563°` 平均、`120/120` 通过。
- DECISION：M1-v3 的表示一致性通过，但控制角度失败；当前不能进入 M2。根因是 M1-v2 的直接根旋转控制与视频实际使用的速度恢复根旋转不一致。下一步先解决根旋转速度路径并在同一留出集复验，不能把 M1-v3 失败归咎于某条 TXT，也不能绕过角度门进入 M2。
- 产物：`diagnostics/phase3/nonturning_v10_user_approved/M1_V3_REPORT.md`、`m1_v3_terminal_sync_v3.json`、`m1_v3_terminal_sync_angle_v2.json`；脚本 `scripts/run_m1_v3_terminal_sync.py`、`scripts/evaluate_m1_v3_terminal_sync.py`；测试 `tests/test_m1_v3_terminal_sync.py`。

## 2026-08-24：276D 表示验证协议实现（本地待同步）

- VERIFIED：新增 `evaluation/vimogen_representation_protocol.py`，定义 ViMoGen-228K 光学动作捕捉数据的来源隔离协议：`representation_dev_v1`（20,000 条开发样本）、`representation_val_v1`（KIT-ML/LAFAN1/BEHAVE 完整来源，共 10,133 条）和 `representation_test_v1`（FIT3D/Mixamo/HumanSC3D/ARCTIC/RICH/EMDB 完整来源，共 5,240 条）。
- VERIFIED：协议校验样本编号、路径、来源、光学子集、发布版来源计数；支持按张量内容 SHA256、帧数、276 维和有限值检查，并检查三个划分之间及 MBench 之间的重复。
- VERIFIED：新增 `scripts/build_vimogen_representation_splits.py` 与 `scripts/validate_vimogen_representation_splits.py`，支持候选清单、可选全量长度扫描、选中动作物化、冻结清单和机器可读摘要；默认不会覆盖既有结果目录。
- VERIFIED：新增 `motion_rep/reconciliation.py`，实现控制感知统一表示：以速度积分轨迹承载局部动态，以直接轨迹减速度轨迹的平滑修正恢复长期锚点，根旋转在 SO(3) 中左乘修正，最后统一重算全部速度通道。该模块默认不接入旧采样路径。
- VERIFIED：新增对应测试文件并完成 Python 语法检查；本地环境缺少 PyTorch/pytest，尚未完成服务器端专项测试。
- BLOCKED：向 `/root/autodl-tmp/vimogen_clean` 上传新增代码被安全策略拦截，需要用户明确授权后才能同步并在服务器环境运行测试和生成真实数据清单。上传前未改变服务器文件或旧实验结果。

### Immediate continuation

1. 获得用户明确授权后，顺序上传 6 个新增文件到服务器干净副本。
2. 在服务器运行协议专项测试和统一模块专项测试。
3. 先用不冻结模式检查完整 ViMoGen-228K 清单计数，再运行选中动作物化与去重审计。
4. 只有审计通过后，写入新的 `diagnostics/phase4/representation_protocol_v1/` 清单；不覆盖阶段0/阶段3结果。

## 2026-08-24：来源隔离真实数据协议与恢复盲测完成

- VERIFIED：新增代码已上传到服务器干净副本：`evaluation/vimogen_representation_protocol.py`、`scripts/build_vimogen_representation_splits.py`、`scripts/validate_vimogen_representation_splits.py`、`motion_rep/reconciliation.py`、`evaluation/representation_recovery.py` 及对应评估/测试脚本。
- VERIFIED：专项测试最终 `11 passed`；全项目回归最终 `133 passed in 20.18s`。
- VERIFIED：冻结清单位于服务器 `/root/autodl-tmp/vimogen_clean/diagnostics/phase4/representation_protocol_v1/frozen_v1/`，注释 SHA256=`5a65da2594f16c9cbf18957d87a2211d057b92b84766f2470642ba94574b9dc2`，有效张量哈希总数 `33,809`，跨三个划分与 MBench 内容重合审计通过。
- VERIFIED：有效样本数为开发 `20,000`、验证 `9,378`、最终盲测 `4,431`。原始来源覆盖数仍分别可审计；验证集排除 `755` 个完全重复张量，最终盲测排除 `809` 个，均记录在 `excluded_duplicate_rows`，没有静默删除。
- VERIFIED：开发校准报告为 `diagnostics/phase4/representation_recovery_v1/dev_calibration_512.json`；固定扰动参数只由开发集估计，验证/盲测未重新估计。
- VERIFIED：验证集恢复评估 `9,378` 条完成，盲测恢复评估 `4,431` 条完成。盲测中位关节位置 RMSE：绝对位置 `0.003591 m`、速度积分 `0.030184 m`、融合 `0.003437 m`；根平移 RMSE：`0.003581 / 0.028092 / 0.003420 m`。融合同时优于两基线。
- VERIFIED：预注册配对自助法 2,000 次完成；盲测融合相对绝对位置的关节 RMSE 中位差为 `-0.000158 m`，95%区间 `[-0.000159,-0.000157]`；相对速度积分为 `-0.026737 m`，95%区间 `[-0.026808,-0.026648]`。根平移对应差值也均为负，区间不跨零。
- VERIFIED：盲测报告为 `/root/autodl-tmp/vimogen_clean/diagnostics/phase4/representation_recovery_v1/test_v1.json`，自助法汇总为 `/root/autodl-tmp/vimogen_clean/diagnostics/phase4/representation_recovery_v1/bootstrap/test_v1_bootstrap.json`；验证集对应文件为 `val_v1.json` 和 `bootstrap/val_v1_bootstrap.json`。
- DECISION：新的 276D 统一表示在本项目的来源隔离盲测上通过“真实恢复误差更小”这一主要门槛；速度积分仍保留局部平滑优势，融合方案应以“准确性优先、平滑性不恶化”为最终解释。该结果证明表示恢复能力，不证明原始生成模型训练外泛化；MBench 继续承担生成质量评价。

### Immediate continuation

1. 将 `motion_rep.reconciliation.reconcile_motion_tensor` 接入生成结束、渲染、MBench 和肌肉代理模型的共同数据边界，默认旧路径继续保持不变，先以显式配置启用。
2. 对新生成动作运行 M0/M1 配对实验和 MBench；保持当前冻结盲测清单不再调参。
3. 若新增生成实验需要改权重或窗口，建立新的协议版本，不回写 `representation_test_v1` 的结论。

## 2026-08-24：统一表示生成边界接入完成

- VERIFIED：`sampling/flow_sampler.py` 新增默认关闭的 `reconciliation_config`、`motion_mean` 和 `motion_std` 接口。启用后，采样器保留 `raw`、`official_pre_cast`、`official` 审计输出，并额外返回 `reconciled` 及协议标识 `vimogen_276d_control_aware_reconciliation_v1`。
- VERIFIED：`train_eval_vimogen.py` 的 M0/M1 生成边界已传递统一表示配置；下游生成张量优先读取 `reconciled`，未配置时继续读取 `official`。历史 M0/M1 审计张量不被覆盖，旧路径保持默认关闭。
- VERIFIED：修正批量输入下每个样本 `[B,276]` 均值/标准差到 `[B,1,276]` 的显式广播；新增采样器显式接入测试和批量统计量测试。
- VERIFIED：针对性测试 `14 passed`；服务器完整回归 `135 passed in 20.44s`。采样器显式统一表示测试 `3 passed`。
- PENDING：尚未用新配置运行新的大规模 M0/M1 真实生成和 MBench；这属于下一阶段生成质量实验，不改变已冻结的真实动作恢复盲测结论。

### Immediate continuation

1. 复制一份冻结配置，显式开启 `representation.reconciliation`，运行少量 M0/M1 生成样本。
2. 核对渲染、MBench 和肌肉代理模型都读取同一 `reconciled` 张量，并保存 raw/official/reconciled 三路审计文件。
3. 通过小规模工程检查后，再运行正式生成质量实验；任何权重或窗口变化都建立新的协议版本，不回写 `representation_test_v1`。

## 2026-08-24：论文结果账本与 MBench 三路实验入口

- VERIFIED：项目根目录已建立 [`result.md`](result.md)，作为正式论文关键结果的唯一权威账本；`RESULTS_REPORT.md` 仅保留工程过程，`PROJECT_MEMORY.md` 只记录状态和索引。正式结果必须具备冻结协议、完整样本/排除/失败计数、机器可读逐样本产物、统计区间、代码版本和摘要值后，才可标记 `VERIFIED` 或 `FROZEN`。
- VERIFIED：新增三路 MBench 离线转换入口 `evaluation/mbench_threeway.py`、配对统计 `evaluation/mbench_threeway_stats.py`、组织与汇总脚本 `scripts/organize_mbench_threeway.py`、`scripts/summarize_mbench_threeway.py`，固定绝对位置、速度积分和窗口9/锚定权重1.0的融合表示。三路均从同一原始276D张量派生，并重新计算速度；正式结果禁止静默覆盖。
- VERIFIED：新增专项测试 `tests/test_mbench_threeway.py`，服务器端已通过 `4 passed`；四动作三路转换冒烟审计通过，服务器产物位于 `diagnostics/phase4/mbench_threeway_smoke/m0_seed0/`，三路最终位置—速度残差最大值为数值零。
- VERIFIED：MBench 正式批处理已增加 `eval_steps=1` 和关闭视频/中间 artifact 的配置，默认训练/历史可视化行为保持不变。先前的 `raw`、`raw_novideo`、`raw_novideo_v2`、`raw_novideo_v3`、`raw_novideo_v4` 运行均为性能试验或已中止，不纳入正式结果；v4 日志记录了收到 SIGTERM 的安全停止，未产生正式结论。
- VERIFIED：服务器专项测试在新增官方统计脚本和归档路径修复后为 `5 passed`；服务器代码语法检查通过。随后服务器全量回归为 `140 passed in 22.75s`。
- VERIFIED：已排除的 v9 归档通过离线物理化烟测：64 条归一化批次恢复为 `[T,276]` 物理张量，有限值检查通过；烟测输出位于服务器 `diagnostics/phase5/mbench_publication_v1/materialize_smoke_v9/`，不属于正式结果。
- EXCLUDED：`raw_novideo_v5`、`raw_novideo_v6`、`raw_novideo_v7`、`raw_novideo_v8` 为保存性能试验，均未纳入正式结果；`raw_novideo_v9` 已确认只保存第一验证批（64/450），标记为不完整并停止，不得用于论文结论。
- IN PROGRESS：正式批处理已切换至 `raw_novideo_v10`。无视频路径现在保存每个验证批的单一归一化276D归档，验证循环不再提前中断；离线脚本 `scripts/materialize_mbench_physical.py` 将合并全部批次并恢复物理276D文件。完成前，MBench 自然性、生成内部漂移和骨盆控制仍标记为 `PENDING`，不得写入论文摘要或结论。

## 2026-08-24：正式 MBench 生成状态复核

- IN PROGRESS：服务器复核时，`raw_novideo_v10/m0/seed_000` 仍在运行，已写入 6/8 个验证批归档（约 384/450 条）；未见失败记录。其余 8 个组合尚未开始或完成。
- PENDING：`physical_v1`、`organized_v1`、官方 MBench 结果和统计汇总尚未生成；后台后处理脚本仍处于等待全部 9 个运行记录完成的状态。因此 `result.md` 中的 MBench 结果继续保持 `PENDING`。

## 2026-08-24：M1 artifact 配置修复与 v11 续跑

- VERIFIED：`raw_novideo_v10/m0/seed_000` 最终完成全部 8 个批次（450/450，耗时约 2,187 秒）。
- EXCLUDED：`raw_novideo_v10/m1_plus5/seed_000` 在生成前因 `m1.enabled requires m1.artifact_dir` 失败，未产生有效动作归档；v10 不作为完整正式协议目录。
- VERIFIED：修正无视频模式下 M1 不必保存中间 artifact 的条件，服务器专项测试仍为 `5 passed`，全量回归仍为 `140 passed`。
- IN PROGRESS：新正式目录 `raw_novideo_v11` 已复制并审计 v10 的完整 M0/seed0（注明 `copied_from`），随后启动修复后的 M1 和其余条件/种子；对应后处理日志为 `diagnostics/phase5/mbench_publication_v1/postprocess_v11.log`。

## 2026-08-24：M1 FlowSampleResult 修复与 v12 续跑

- EXCLUDED：`raw_novideo_v11/m1_plus5/seed_000` 在 M0 采样后失败，原因是无视频模式下 M0 未返回 M1 所需的 `FlowSampleResult`；没有产生有效归档。
- VERIFIED：修复 M1 模式下即使不保存中间 artifact 也强制返回 M0 审计结构；服务器专项测试 `5 passed`，代码语法检查通过。
- IN PROGRESS：`raw_novideo_v12` 已复用已完成的 M0/seed0；M1+5/seed0、M1+10/seed0 和 M0/seed1 已各完成 8/8 批（450/450）。当前 `m1_plus5/seed1` 与新接续的 `m1_plus10/seed1` 两路并行运行，正式目录为 4 个已完成运行、2 个运行中运行、3 个待运行组合。
- VERIFIED：两路并行启动后显存约 11.8 GiB、剩余约 20.4 GiB；两个 `run_record.json` 均为 `RUNNING`，GPU 进程均存在，未见失败记录。后处理日志仍为 `completed=4 failed=0`，等待全部 9 个组合完成。
- VERIFIED：后处理进程仍在运行，最新日志为 `completed=3 failed=0`，说明当前没有新的失败记录，正在等待全部 9 个组合完成后再进行物理化、三路组织和官方评价。
- PENDING：`physical_v1`、`organized_v1`、官方 MBench 结果、漂移统计和配对统计尚未生成；`result.md` 中的 MBench 结果不得提前引用。

## 2026-08-25：v12 seed2 调度恢复

- VERIFIED：截至服务器 2026-08-25 09:26，`m0/seed0-1`、`m1_plus5/seed0-1`、`m1_plus10/seed0-1` 共 6 个组合已完成，每个均为 8/8 批（450/450）；无失败记录。
- BLOCKER RESOLVED：原批处理控制进程在 6 个组合完成后退出，导致三个 seed2 组合未启动；不是生成模型失败。已重新启动 `run_mbench_ablation_batch.sh`，并确认 `m0/seed2` 已进入运行。
- VERIFIED：`m0/seed2` 已完成 8/8 批（450/450）。
- VERIFIED：`m0/seed2` 与 `m1_plus5/seed2` 已完成 8/8 批（450/450），使正式生成完成 8/9 个组合。
- VERIFIED：最后的 `m1_plus10/seed2` 已完成 8/8 批（450/450），正式生成目录 9/9 个组合全部完成、无失败记录；每个组合均有完整 8 个归档批次。
- IN PROGRESS：已启动正式后处理；生成记录审计为 `status=VALID, run_count=9`，`physical_v1` 和 `organized_v1` 已创建，当前正在执行三路组织，尚未完成官方 MBench 评价和统计汇总。
- PENDING：官方 MBench 结果、漂移统计和配对统计完成并通过自动一致性审计前，`result.md` 中的 MBench 结果仍不得引用。

## 2026-08-25：后处理完成与官方 MBench 环境修复

- VERIFIED：物理化、三路组织和漂移统计已完成；9 个组织清单均为 `record_count=450, error_count=0`，漂移汇总位于服务器 `diagnostics/phase5/mbench_publication_v1/drift_summary.json`。
- EXCLUDED：官方 MBench 首次运行因服务器缺少 `OSMesa` 失败；已确认不是动作数据或算法错误。
- VERIFIED：服务器 EGL 最小离屏渲染测试通过；已将 `mbench/render.py` 改为尊重显式后端并默认使用 EGL，并安装公开依赖 `shapely==2.1.2`。
- VERIFIED：服务器 `nproc/lscpu` 显示宿主机可见 208 个逻辑 CPU，但容器实际 CPU 配额为 `cpu.max=2500000 100000`，即 25 vCPU；后续资源判断以 25 vCPU 为准。
- IN PROGRESS：官方 MBench 27 个组合已断点重跑；6 个并行 worker 已正常完成。当前 23/27 个评价组合已完成、1 个正在运行、3 个待处理，无失败记录；当前运行主进程 CPU 约 213%，并行执行不改变评价协议。尾部 watcher 已启动，等待主脚本进入 `m1_plus5/seed2/absolute_position` 后并行运行另外两种表示。
- VERIFIED：新增 `scripts/run_official_mbench_worker.py`，每个 worker 独占一个条件/种子/表示目录，拒绝覆盖 `COMPLETED` 或 `RUNNING` 记录；当前并行化不改变评价协议，只改变执行顺序。
- PENDING：官方运动质量评价和统计完成前，`result.md` 中的 MBench 自然性结果仍不得标记为 `VERIFIED` 或 `FROZEN`。

## 2026-08-25：官方 MBench 评价状态复核（并行阶段结束）

- VERIFIED：服务器 `results/phase5/mbench_publication_v1/official_motion_quality_v1/` 下 27/27 个“条件 × 随机种子 × 表示方法”目录均已生成 `run_record.json`、`eval_results.json`、`per_motion_results.json` 和 `full_info.json`，记录状态均为 `COMPLETED`，没有记录级失败。
- VERIFIED：并行 worker 已完成其负责的尾部任务；此前启动的串行主程序未感知并行结果，继续重复运行 `m1_plus5/seed2/velocity_integral`。该重复进程已停止，未删除或覆盖已生成产物。这解释了并行进程数和显存占用下降：并行任务已经结束，而残留串行任务只是重复计算。
- BLOCKED FOR FORMAL REPORTING：27 个 `evaluate.log` 均报告 `Body_Penetration` 和 `Pose_Quality` 因缺少 `mesh_intersection` 而为错误结果；因此当前只能确认其它可用 MBench 维度已产出，不能把 27 个目录称为“完整官方 MBench 结果”。
- PENDING：`official_motion_quality_summary.json` 尚未生成；需要先补齐或明确排除 `mesh_intersection` 依赖，再运行官方汇总、提示词级配对统计和自动一致性审计。`result.md` 中所有 MBench 论文结果继续保持 `PENDING`。

## 2026-08-25：MBench 缺失指标修复与输入边界复核

- VERIFIED：确认缺失原因是 `mdm5090` 环境没有 `mesh_intersection`；不是动作文件、三路转换或生成失败。
- VERIFIED：从官方 `torch-mesh-isect` 源码构建 CUDA 扩展，并针对当前 PyTorch 2.7/ CUDA 12.8 修正 `AT_CHECK`、`Tensor.type()` 兼容性；补齐公开 CUDA Samples `helper_math.h` 后安装成功。
- VERIFIED：`mesh_intersection` 导入测试和 CUDA BVH 最小碰撞测试通过。
- EXCLUDED：首次补跑只完成了依赖烟测，随后停止；原因不是 CUDA 扩展，而是 `organized_v1` 只有 12,150 个 `.npy` 关节轨迹、没有官方 `Body_Penetration`/`Pose_Quality` 所需的 SMPLify `.pt` 文件。相关 v2 记录已标为 `EXCLUDED`，未产生可用于论文的补跑指标。
- VERIFIED：已从现有物理化 276D 动作中核对出可生成项目本地 SMPL 数据的路径（`motion_rep.retarget_motion.motion_rep_to_SMPL` + SMPL 模型），但这属于新的“直接由 276D 生成 SMPL”辅助协议，不能冒充官方 SMPLify 输入协议。
- VERIFIED：NRDF 官方模型配置和权重已下载、上传并放置到官方代码期望的嵌套路径；`load_model(...)` 已成功加载。该步骤不需要重新生成动作。
- BLOCKED FOR FORMAL REPORTING：正式官方 Body Penetration/Pose Quality 仍缺 SMPLify `.pt` 输入。必须在“严格补齐官方 SMPLify 输入”与“新增项目本地直接 SMPL 辅助协议”之间作协议选择；在选择前不估计正式完成时间，也不把 v1/v2 写入 `result.md`。
- PENDING：MBench 其它已完成维度、累计漂移和真实动作盲测仍可保留，但 MBench 缺失指标修复、官方汇总和论文统计尚未完成。

### Immediate continuation

1. 先由用户确认 MBench 缺失指标采用严格 SMPLify 方案，还是采用明确标注的项目本地直接 SMPL 辅助方案。
2. 方案确认后只在新版本目录生成输入、运行缺失指标并完成逐样本合并；不覆盖 `official_motion_quality_v1` 或已排除的 v2。
3. 运行官方汇总、提示词级配对统计和自动一致性审计；未通过前，`result.md` 中 MBench 结果保持 `PENDING`。

## 2026-08-25：SMPLify 输入需求调查完成

- VERIFIED：查阅 ViMoGen 官方 README 和官方 `mbench/pose_quality.py`。官方流程先把 `(T,22,3)` 的关节 `.npy` 通过 SMPLify 逆运动学转换为 SMPL 参数，再由 `Body_Penetration` 读取 `vertices`、由 `Pose_Quality` 读取轴角 `pose`；因此缺失的不是普通 Python 包，而是一个独立的“关节到 SMPL 参数”预处理阶段。
- VERIFIED：官方 ViMoGen 仓库的基础 `requirements.txt` 没有包含 SMPLify、`mesh_intersection` 或 NRDF 权重；README 只说明需要运行 SMPLify，没有给出唯一的 SMPLify 版本、配置或下载地址。不能在未核对版本和输出格式前盲目下载第三方实现。
- VERIFIED：服务器已有并已核验 `SMPL_NEUTRAL.pkl`、MBench 数据、`mesh_intersection` CUDA 扩展和 NRDF 配置/权重；这些不需要用户再次下载，也不需要重新生成动作。
- PENDING：严格官方协议仍需确定并固定 SMPLify 实现、SMPL pose prior/VPoser（如所选实现需要）、拟合参数、坐标系和输出字段。输出至少必须可审计地包含 `pose`、`joints`、`vertices`，并与当前 22 关节输入逐帧对齐。
- DECISION GATE：在补齐并通过单动作预检（输入坐标、SMPL 重建关节误差、有限值、字段和维度）之前，不启动 2,700 个正式 SMPLify 任务，不下载未知第三方模型，不把直接由 276D 导出的 SMPL 结果冒充官方 SMPLify 结果。

## 2026-08-25：官方 SMPLify 自动预处理复核与数值故障根因更正

- CORRECTION：上一节把 SMPLify 判断为缺失的独立预处理程序，这一判断已被官方完整入口源码推翻，现标记为 SUPERSEDED。官方 `mbench/__init__.py` 会对 `Body_Penetration` 和 `Pose_Quality` 自动调用 `mbench/render.py`，将对应 `.npy` 关节轨迹执行仓库内置 SMPLify，并缓存包含 `pose`、`joints`、`vertices` 的 `.pt`；无需另找第三方 SMPLify 实现。
- VERIFIED：服务器已有官方要求的 `data/body_models/smpl/SMPL_NEUTRAL.pkl`，实际大小 `247186228` 字节，SHA256=`4924f235e63f7c5d5b690acedf736419c2edb846a2d69fc0956169615fa75688`；同时已核验 `J_regressor_extra.npy`、`gmm_08.pkl`、`neutral_smpl_mean_params.h5`、下采样索引、身体分段文件、NRDF 权重及 `mesh_intersection` CUDA 扩展。用户不需要重新下载 SMPL 模型。
- ROOT CAUSE VERIFIED：服务器 NumPy/OpenBLAS 默认自动选择的 CPU 内核错误计算姿态先验协方差的行列式和逆矩阵，导致 `prior.py` 出现非法平方根，随后 LBFGS 报 `IndexError: list index out of range`。设置 `OPENBLAS_CORETYPE=HASWELL` 后，全部协方差行列式为正，逆矩阵最大残差约 `3.55e-14`。
- VERIFIED：使用 `OPENBLAS_CORETYPE=HASWELL` 对正式样本 `150.npy` 的前两帧运行官方 `mbench.render.render(..., render_video=False)` 成功，SMPLify 用时 `60.95` 秒，生成可审计 `.pt`；`pose=(2,24,3)`、`joints=(2,22,3)`、`vertices=(2,6890,3)` 均为有限值。NRDF 姿态评分和 CUDA BVH 身体碰撞两条下游路径均通过真实产物冒烟测试。
- RISK RESOLVED：原补跑合并脚本错误假设原有运动质量动作编号与姿态质量动作编号完全一致；实际官方分别使用编号 `0–149` 与 `150–249`。已修复为按动作编号合并两个子集的并集，保留重叠动作的已有指标，并拒绝重复编号、缺失指标、非有限值。
- VERIFIED：补跑脚本已固定 `OPENBLAS_CORETYPE=HASWELL` 并记录数值环境；新增专项测试覆盖不相交子集、重叠子集、无效数值和重复编号。服务器补跑与三路实验专项测试合计 `12 passed in 1.78s`，随后服务器完整回归测试为 `147 passed in 23.27s`。
- VERIFIED：编号 `150` 的完整 `100` 帧动作官方 SMPLify 预检成功，拟合用时 `133.46` 秒，端到端 `136.05` 秒，产物约 `8.0 MiB`。真实产物 `pose=(100,24,3)`、`joints=(100,22,3)`、`vertices=(100,6890,3)` 均为有限值；官方 NRDF 姿态质量得分为 `1.8637345731258392`，CUDA BVH 已对全部 `100` 帧完成身体碰撞检查，两条下游指标链路合计约 `2.1` 秒。
- CAPACITY ESTIMATE：严格官方协议需要处理 `27×100=2700` 条拟合动作；按已测单样本 `133.46` 秒，串行拟合下限约 `100` 小时。当前服务器配额为 `25 vCPU`、`32 GB` 显存，单样本预检约占 `888 MiB` 显存和接近 `2` 个 CPU 核；若运行 `8–12` 路并行，理想下限约 `8–13` 小时，实际需额外计入显卡争用和样本差异。不能再承诺两三小时完成，也不能擅自减少官方优化迭代次数。
- PENDING：完整预检已证明无需用户下载任何模型，也无剩余已知依赖；大规模正式补跑和论文账本更新尚未启动。正式运行须在新版本结果目录使用受控并行、缓存断点续跑和明确资源监控；完成并通过统计审计前，`result.md` 的 MBench 结果继续为 `PENDING`。

## 2026-08-25：受资源保护的官方 MBench v3 补跑已启动

- VERIFIED：用户授权启动严格官方 SMPLify 补跑，要求控制 GPU、CPU 和显存占用。新增 `scripts/launch_official_mbench_repair_v3.sh`；增强 `scripts/run_official_mbench_repair_parallel.py` 和 `scripts/run_official_mbench_repair_worker.py`，不覆盖 v1 完整运动质量结果或已排除 v2 目录。
- VERIFIED：服务器实际资源为 `25 vCPU`、`90 GiB` 容器内存、`32607 MiB` 显存和约 `318 GiB` 剩余磁盘。约 `68 GiB` 是可回收文件缓存，因此内存保护使用 cgroup 有效工作集，而不是把缓存误判成进程耗尽内存。
- VERIFIED：正式调度配置为 `max_workers=10`、`initial_workers=4`、每轮最多增加 `2` 路、轮询间隔 `15` 秒。达到以下任一阈值即暂停增加任务：CPU 配额使用率 `85%`、GPU 计算利用率 `90%`、显存占用 `70%`、有效内存占用 `75%`、GPU 温度 `78°C`；剩余磁盘低于 `64 GiB` 同样停止扩容。
- VERIFIED：每个 worker 固定 `OPENBLAS_CORETYPE=HASWELL`、`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`、`NUMEXPR_NUM_THREADS=1`，并要求完整 `100` 条姿态评价动作及全部数值有效；发现失败时调度器停止扩容并中断仍在运行的同批任务。动作 `.pt` 缓存由官方入口生成，重新启动时可被官方程序复用。
- VERIFIED：新增资源阈值和样本完整性测试后，服务器专项测试 `15 passed in 1.71s`，完整回归 `150 passed in 22.29s`。
- IN PROGRESS：服务器北京时间 `2026-08-25 23:56` 正式启动调度主进程 PID=`60732`，输出根目录 `/root/autodl-tmp/vimogen_clean/results/phase5/mbench_publication_v1/official_motion_quality_v3/`，调度日志 `/root/autodl-tmp/vimogen_clean/diagnostics/phase5/mbench_publication_v1/official_repair_v3_scheduler.log`，PID 文件为同目录 `official_repair_v3_scheduler.pid`。
- VERIFIED：启动后已存在 `4` 条真实 evaluator 子进程、`4` 条 `RUNNING` 记录、`0` 条失败；实际显存约 `3.5/32 GiB`、GPU 温度约 `36°C`、有效工作内存约 `6.9/90 GiB`、CPU 配额使用率约 `13%`。GPU 计算利用率达到 `100%`，因此资源保护将 `gpu=100.0%` 标记为扩容阻断，并自动维持当前 `4` 路；这表示单卡计算已饱和，不是调度卡住。若后续 GPU 利用率低于阈值且其他资源仍安全，调度器将自动增加并发，最多 `10` 路。
- IN PROGRESS：结构化实时状态文件为 `/root/autodl-tmp/vimogen_clean/results/phase5/mbench_publication_v1/official_motion_quality_v3/scheduler_state.json`，包括 `active_workers`、`completed_jobs`、`failed_jobs`、`smplify_cache_count/2700`、资源快照和 `resource_blockers`；每个条件/种子/方法单独写入 `run_record.json` 和 `evaluate.log`。
- PENDING：全部 `27` 组评价和 `2700` 条官方 SMPLify 缓存完成后，调度器自动生成 `/root/autodl-tmp/vimogen_clean/diagnostics/phase5/mbench_publication_v1/official_motion_quality_summary_v3.json`。结果仍须经过正式审计和写入根目录 `result.md` 后才能作为论文主张；骨盆角度独立统计及原始通道诊断汇总修复仍待后续处理。

### Immediate continuation

1. 先检查 `official_motion_quality_v3/scheduler_state.json` 和 `official_repair_v3_scheduler.log`，核对主进程、实际活跃路数、缓存计数、失败记录和资源阻断原因。
2. GPU 若持续 `100%` 且显存/温度安全，保持受控调度，不为提高表面 CPU 占用强制增加任务。
3. 任一 worker 失败时，先检查对应 `evaluate.log` 和 `run_record.json`；保留现有 `.pt` 缓存，不覆盖 v1 或 v3 已完成结果。
4. 待 27 组完成后，核对自动生成的 `official_motion_quality_summary_v3.json`、150/100 提示词子集和配对统计，再更新正式论文账本 `result.md`。

## 2026-08-28：根前向多种子验证与 v1.2 严格约束实现

- VERIFIED：新增协议 `vimogen_relative_root_forward_v1_2_trunk_stabilized`，保持直接 `body_pose/root_rotation/root_translation` 为唯一姿态权威，`J/dJ/dR/dT` 全部由 FK 和相邻姿态派生；旧 v1/v1.1 代码路径与结果目录未覆盖。
- VERIFIED：v1.2 使用独立的俯仰、完整前向、水平朝向、躯干方向角度约束，不构造跨量纲加权总损失；允许编辑的直接通道仅为根旋转和 `spine1/2/3`，根位置、髋、腿、手臂、肩、颈部保持不变。
- VERIFIED：补齐最终候选的统一验收：四项角度误差分别非增且至少一项下降；合成后的总体标准化 276D 修正 RMS 必须不超过 `0.05`；脊柱候选失败时返回未编辑端点，不保留只改根的违规候选。服务器专项测试 `4 passed`，完整回归 `216 passed in 27.23s`。
- VERIFIED：服务器已运行初始参数 `residual_gain=1.5, max_step=8°, heading_gain=0.75, max_heading_step=2°, trunk_gain=0.75, sigma=[0.066,0.75]` 的 42 及三个留出种子矩阵；在尚未加入最终总预算的旧验收版本中，seed42 通过，留出种子分别因转弯样本的前向/水平 P95 或尾部跳变失败，不能宣称五种子通过。
- VERIFIED：为诊断残差传递，服务器额外测试 `residual_gain=2.5, heading_gain=1.0, max_heading_step=3°`。该组在部分失败样本上改善到前向 P95 `1.84°`，但 seed1057660199 的转弯样本仍有前向 P95 约 `2.59°`、尾部额外 SO(3) 跳变约 `2.98°`；增加 `sigma_max` 到 `0.92` 反而恶化到约 `7.84°`，因此不选用。
- VERIFIED：严格总预算代码已同步服务器；在 seed0 `+5°` 真实冒烟中，不可行合成候选被安全拒绝并返回基线（无超预算输出），一致性仍通过。这证明当前“总 RMS≤0.05”与四项非增约束尚未同时具备可行候选，严格 v1.2 尚未达标。
- VERIFIED：v1.2 运行配置将遗留的 `motion_weight` 置为 `0.0`，明确不启用 v1/v1.1 的角度加权运动损失；该字段仅保留在配置接口中用于协议隔离。
- PENDING：尚未完成严格代码版本下五种子 × 两样本 × 四剂量的 40 个组合，也未生成 v1.2 正式 MP4。此前未加最终总预算的旧运行保留为校准证据，不能作为 v1.2 最终结果。

### Immediate continuation

1. 先在严格验收代码上设计可行的预算分配/候选搜索，使最终总 276D RMS、四项角度约束和尾部约束能够同时通过；不得放宽门槛或用终端补角掩盖失败。
2. 只有 seed0/42 校准在严格代码下有可行候选后，才重跑三个留出种子，并按逐种子逐剂量硬门槛判定，不做跨种子动作平均。
3. 五种子全部通过后，再生成 seed0 和最差种子的侧视网格/骨架 MP4；在此之前不把基线或旧验收视频称为 v1.2 控制证据。

## 2026-08-26：v3 补跑夜间进度核验

- VERIFIED：服务器调度器仍为 `RUNNING`，主进程未退出，日志持续产生心跳记录；当前完成 `4/27` 组，待处理 `19` 组，失败 `0` 组。
- VERIFIED：官方 SMPLify `.pt` 缓存计数为 `657/2700`，约完成 `24.3%`；已完成组对应的评价记录已由 worker 写入 v3 目录，未覆盖 v1 或 v2。
- VERIFIED：最近资源快照为 GPU 利用率 `100%`、显存 `4594/32607 MiB`（约 `14.1%`）、GPU 温度 `51°C`、CPU 配额使用率约 `16.1%`、有效工作内存约 `10.0/90 GiB`、剩余磁盘约 `313 GiB`。调度器因 `gpu=100.0%` 持续阻止增加并发，当前稳定维持 `4` 路。
- SUPERSEDED：本节原记录的“实际缓存吞吐约 `300` 条/小时”是经过时间计算错误得到的数值，不得用于进度或完成时间估算；正确测量见下方“v3 性能瓶颈复核”。
- PENDING：仍未生成最终 `official_motion_quality_summary_v3.json`，未完成全量一致性审计，`result.md` 中 MBench 正式结果继续保持 `PENDING`。

## 2026-08-26：v3 已完成组的两个姿态指标核验

- VERIFIED：已完成的 `4/27` 组均为 `returncode=0`、`status=COMPLETED`，每组 `motion_count=250`；这是原有编号 `0–149` 的 150 条运动质量记录与新增编号 `150–249` 的 100 条姿态质量记录合并后的预期数量。
- VERIFIED：四组对应的 `evaluate.log` 均出现 `Evaluation completed successfully`，没有发现 `ERROR`、`Traceback` 或 `Skipped`；日志同时输出了 `Body_Penetration` 与 `Pose_Quality` 的均值和标准差。例如 `m0/seed_000/absolute_position` 为 Body Penetration=`1.3522`（标准差 `0.8262`）、Pose Quality=`2.2092`（标准差 `0.5198`）；同一条件下融合方法分别为 `1.3489` 和 `2.2058`。这些仅是已完成组的阶段性观察，不是全量统计结论。
- VERIFIED：worker 只有在官方补跑结果返回成功、补跑结果确实包含完整 100 条动作且两个维度均为有限数值后，才会写入 `COMPLETED` 并生成合并文件。因此目前可以确认两个指标评价链路已经打通并能稳定产出。
- PENDING：当前只有 4 组完成，尚不能据此判断三种表示方法的最终优劣；仍需等待 27 组全部完成、生成汇总文件并进行提示词级配对统计和自动审计后，才可写入 `result.md`。

## 2026-08-26：v3 性能瓶颈复核与累计漂移解释边界

- VERIFIED：依据调度器精确启动时间、状态时间和缓存计数重新计算，运行 `10.45` 小时生成 `672/2700` 条 SMPLify 缓存，实际吞吐为 `64.28` 条/小时；相对单动作完整预检 `136.05` 秒与四路并发对应的理论值 `105.84` 条/小时，并行效率约 `60.7%`。若运行状态不变，剩余约 `31.5` 小时；这是滚动估算，不是完成承诺。
- VERIFIED：首批四个已完成组各处理 `100` 条姿态动作，用时约 `6.06–6.09` 小时，即并发状态下约 `220` 秒/动作，比单进程完整预检的 `136.05` 秒慢约 `63%`；四路总加速仅约 `2.46` 倍，证明存在显著的同卡计算争用。
- ROOT CAUSE VERIFIED：连续 GPU 采样显示计算核心使用率为 `100%`，显存带宽仅 `6–7%`，显存约 `4.6/32.6 GiB`，温度约 `49–51°C`；CPU 配额使用率约 `16%`。因此瓶颈是单卡 SMPLify 优化的计算/内核调度争用，而不是 CPU、显存容量、显存带宽、温度或磁盘。仅因为 CPU 和显存空闲而提高到 `8–10` 路，不能推导出更高吞吐，反而可能继续增加单样本耗时。
- ROOT CAUSE VERIFIED：官方 `simplify_loc2rot.py` 固定 `num_smplify_iters=150`；`smplify.py` 中相机阶段和身体阶段均使用 `LBFGS(max_iter=150)`，外层又分别调用优化步骤 `10` 次和 `150` 次。该迭代结构是主要耗时来源。`render_video=False` 已生效，日志确认时间几乎全部消耗在 SMPLify，两个姿态指标共享同一 `.pt` 拟合缓存，并不存在重复跑两遍指标的问题。
- DECISION：当前 v3 继续按冻结的严格官方协议运行，不在中途改变迭代次数、初值、批处理方式或混用不同配置的结果。安全但收益有限的候选优化是跨动作复用 SMPL 模型、姿态先验和转换对象；可能产生数量级提速的候选优化是减少外层迭代、早停或批量拟合，但这些会改变数值协议，必须另建 `smplify_perf_v1` 小样本等价性实验，不能混入 v3。
- INTERPRETATION BOUNDARY：绝对位置方法的“相对直接位置锚点偏离”为零是定义导致的，不是准确性或平滑性证据，也不能用于给三种方法排序。融合相对绝对位置的优势应由来源隔离真实动作盲测恢复误差和独立的导数/自然性指标（抖动、脚底滑动、脚部悬空、加速度与急动度）证明；融合相对速度积分的优势由长期累计漂移证明。正式表格须把绝对位置的零标注为“按定义为参考值”，并与真实恢复准确性和平滑性分栏报告。

## 2026-08-26：绝对平均骨盆角引导方案 v1 已实施并等待真实动作冒烟

- DECISION/FROZEN：新协议名为 `vimogen_absolute_mean_pelvis_v1`。控制目标是有效帧模型空间骨盆俯仰角的绝对平均值；`+5°/+10°` 不是逐帧常量，也不是相对 M0 的增量。角度只从窗口 9、锚定权重 1.0 的直接根旋转—旋转速度积分 SO(3) 融合结果计算。
- VERIFIED：新增服务器代码 `sampling/absolute_mean_pelvis_guidance.py`，在每个启用采样步执行 `x0_hat = x_sigma - sigma * v`、反标准化、SO(3) 融合、绝对平均角损失、去均值曲线保持损失、整体动作保持损失和 RMS 上限 0.05 的更新；更新后再次统一表示并重算关节、根旋转、根平移全部速度。旧 M0/M1 默认路径不启用该模块。
- VERIFIED：`sampling/flow_sampler.py` 新增互斥的 `absolute_mean_guidance` 接口及显式 FP32 `g0/g1` 结果；`reconciled` 仍保留调用方历史输出 dtype。`train_eval_vimogen.py` 新增独立配置段、M0/G0/G1 审计张量和指导摘要，禁止与旧相对角 M1 或额外末端 reconciliation 同时启用。
- VERIFIED：G0 为扩散结束后统一 SO(3) 根旋转并重算全部速度的输出。G1 只在 `abs(target - mean(G0)) <= 1°` 时搜索并施加一个不超过 1° 的世界坐标 x 轴 SO(3) 共同修正；残差超过 1° 时保留 G0、记录失败，不强制掩盖。
- VERIFIED：新增 `evaluation/absolute_mean_pelvis.py`，输出每帧曲线、绝对平均误差、去均值曲线相关/均方根误差、波动标准差比、动作保持 RMS 和根旋转—旋转速度 SO(3) 残差；成功门固定为每目标中位误差 `<=2°`、至少 `90%` 单元 `<=2°`、根旋转残差 `<=1e-4°`、相关中位数 `>=0.90`、标准差比 `0.8–1.2`。
- VERIFIED：协议在任何新模型运行前冻结到 `/root/autodl-tmp/vimogen_clean/results/phase6/absolute_mean_pelvis_v1/protocol.json`，SHA256=`1a3f9799542932cd5b4970ae941635f8d2d96de46720b28b6a5f7fb7443257de`。开发清单 20 条 SHA256=`c45b1fa103cad9b5da2ebae0500ef04e61fb3e95dabcacfbbd06d13cc36fd692`；MBench 步行候选 65 条、主盲测 40 条、鲁棒性 450 条、正式视频 12 条的清单 SHA256 分别为 `d8cbeb366cf03cfd5e5435eaa7c39ac71f515a05691d068229824ba8524eb603`、`36826ffe83f12995305d5e89fdb6386b750a1c2f3a827228095a13aaf1dafe1c`、`167f4e131c724abab043121bf2e09f8c5cdebcddfbf497194f8cf44f46d5a17e`、`e13f6594464d197415e50b4da23966e79ab2d12d63e39a2247961bb451253cd1`。
- VERIFIED：40 条主盲测分层为直行 6、转弯 8、手臂伴随 8、启停 8、复杂步行 10；选择使用固定种子 `20260826`，没有读取新方法结果。既有 `frozen_v2_reviewer1` 20 条开发集只用于强度 `{0.5,1.0,2.0}` 与 shape 权重 `{0.05,0.1,0.2}` 的 seed0 选择；seed1/2 只验证。
- VERIFIED：专项测试为 `27 passed in 3.95s`，覆盖绝对语义、三项损失、sigma `[0.25,0.65]`、RMS 上限、G0/G1、FP32 权威边界、门槛、冻结清单、运行配置、旧采样兼容和 MP4 合成。一次隐藏 GPU 的全项目回归在测试收集阶段因项目原有 T5 模块调用 `torch.cuda.current_device()` 报 `No CUDA GPUs are available`，不是测试断言失败；为避免与 v3 争用，完整 GPU 可见回归待 v3 完成后执行。
- VERIFIED：布局/编码 MP4 冒烟位于 `/root/autodl-tmp/vimogen_clean/results/phase6/absolute_mean_pelvis_v1/videos/smoke/layout_only/triptych_layout_codec_smoke.mp4`；已核验 H.264、20fps、1920×1080、20 帧。该视频明确标记 `VERIFIED_LAYOUT_CODEC_ONLY` 和 `not_model_evidence=true`，不能作为真实动作或控制证据。
- IN PROGRESS：官方 MBench v3 仍正常运行且未被停止或覆盖；最近 phase-6 守护日志记录 v3 `completed=5`、SMPLify 缓存 `792/2700`、活跃 worker `3`。真实动作冒烟守护进程 PID=`91034`，日志 `/root/autodl-tmp/vimogen_clean/results/phase6/absolute_mean_pelvis_v1/logs/real_motion_smoke_after_v3.log`；它每 60 秒只读检查 v3，只有 v3 状态为 `COMPLETED` 后才运行 sample 94、seed0、绝对目标 +5/+10 的两个冒烟单元、离线指标和两段三路 MP4。v3 异常结束时守护任务直接失败，不进入开发集。
- VERIFIED：门槛化运行工具已就绪：`scripts/run_absolute_mean_pelvis_v1.py`、`scripts/evaluate_absolute_mean_pelvis_v1.py`、`scripts/render_absolute_mean_triptych.py`、`scripts/orchestrate_absolute_mean_pelvis_v1.py`。调度器会在 v3 仍为 `RUNNING` 时拒绝任何 phase-6 GPU 实验；开发 seed0 九点网格、seed1/2 只验证及正式 40/450 运行均拒绝覆盖不完整目录。
- PENDING：真实动作冒烟、20 条开发网格、G1 自然性门、40 条主盲测、450 条鲁棒性和 24 个正式视频均尚未完成；因此 `result.md` 未更新。正式结果必须在角度、表示残差、曲线保持、自然性和产物完整性全部核验后才可写入。

### Immediate continuation

1. 先只读检查 v3 `scheduler_state.json` 和 PID `91034`；v3 未完成时不得手动启动 phase-6 GPU 运行。
2. v3 完成后检查 `summaries/real_motion_smoke.json`、两个 smoke run 的 `per_frame_angles.csv`/`per_action_metrics.csv` 和两段真实动作 MP4；若守护任务失败，按日志定位并保留失败目录。
3. 只有真实动作冒烟通过，才运行 `scripts/orchestrate_absolute_mean_pelvis_v1.py select`；seed0 选择通过后再运行 `verify`，不能用 seed1/2 重新选参。
4. 开发验证通过后才运行 `formal`；正式阶段仍需接入 G0/G1 的官方 MBench 自然性配对统计和 G1 晋升门，再生成预先固定的 24 段视频及标注为事后选择的最佳/最差案例。
5. 所有正式核验完成前保持根目录 `result.md` 不变；失败结果不得删除。
