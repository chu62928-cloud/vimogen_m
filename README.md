# ViMoGen 骨盆姿态控制研究代码

本仓库保存基于 [MotrixLab/ViMoGen](https://github.com/MotrixLab/ViMoGen) 开展的骨盆姿态控制、276 维动作表示一致性、评价与可视化研究代码。

当前首个基线快照对应 `vimogen_absolute_mean_pelvis_v3_tail_safe`：它保留完整前向运动学重建和末端安全融合，用于下一版解剖骨盆前倾控制的可追溯起点。

## 存档原则

- 使用分层 Git 提交记录每一阶段的实现和验证变化。
- 不提交模型权重、SMPL/SMPL-X 受许可模型、数据集、实验视频或大体积结果。
- 不提交服务器地址、密码、令牌、私钥或本机连接脚本。
- `results/phase6/absolute_mean_pelvis_v3/` 仅选择性保存冻结协议与小型清单，不保存生成结果。

## 上游与许可

本仓库不是 ViMoGen 官方仓库。上游代码及其模型、数据和第三方依赖仍受各自条款约束；本仓库只记录本研究中的新增或修改内容，不对上游资产授予额外许可。
