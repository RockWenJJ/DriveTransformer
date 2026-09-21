# DTL-ViTB-Fuse3-NoMap-S16：训练交接

模型正式名称：**DTL-ViTB-Fuse3-NoMap-S16**，W&B 展示名称使用同名。

本次任务只准备代码、配置、启动与监控入口。正式训练尚未启动。正常输入 ax、ay，不做加速度消融。

## 1. 已确定的模型

六路 480×960 图像 → 冻结的 DINOv3 ViT-B/16 → 取第 4、8、12 层 Patch 特征 → 通道拼接为 2304×30×60 → 可训练的 1×1 卷积压到 768 通道 → GroupNorm、GELU → 3×3 卷积（stride=1）、GroupNorm → 768×30×60 → DT-Large 的 12 层、768 维主解码器 → 目标检测、运动预测、自车规划。

- 代码中的取层索引为 `out_indices=(3, 7, 11)`，从零计数。
- 每个取层点使用 DINO 原有、冻结的输出 LayerNorm；三个取层点共享该 LayerNorm。移除 CLS 和 4 个 Register tokens，只拼接同一空间位置的 Patch 特征。
- `img_neck=None`；融合模块位于 backbone 的 `adapter`。三个尺度相同的特征按通道拼接，输出仍只有一个尺度。
- Adapter 与 Head 的步长均为 16，每路 1800 个 token，六路 10800 个 token，每个 token 768 维。
- DT-Large 的 Agent 预解码器、12 层主解码器、768 维注意力、3072 维 FFN 与 extra_post_norm 配置保留。
- 移除 Map queries、Map 历史记忆、Map 专用网络与损失；保留 Agent/Ego 分支。
- DINO 加载现有预训练权重并完全冻结。融合模块和 DT 网络从头初始化，`load_from=None`；不加载旧整模权重或本次验证权重。
- 三个数据划分都使用 `ego_accel_mode='xy'`、时间间隔 0.1 秒。在线正常输入 ax、ay，关闭评估 dropout，采用已有 sin_heading 解码。

## 2. A100 上的实际位置

主机：`wenjj_alicloud_A100`

代码目录：

```text
/data5T/wenjj/Projects/DriveTransformer_worktrees/large_vitb3_nomap_s16_20260921/DriveTransformer
```

分支：`feat/large-vitb3-nomap-s16-20260921`，基于 `6ac3bf088fb43c840ac18ca085eca85293a8d6be`。修改在这个独立工作区中；请使用此目录，不要切到原始 DriveTransformer 目录启动。

配置：

```text
adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large_vitb3_nomap_s16_res480_10ep.py
```

Python：`/home/wenjj/miniconda3/envs/drivetransformer/bin/python`

数据、预训练权重和 MMCV 编译扩展已连接到服务器现有资源。验证材料放在 `preparation/`，其中的临时权重已加入 Git 忽略规则。正式输出为 `work_dirs/large_vitb3_nomap_s16_res480_10ep/`。

## 3. 正式训练默认参数

| 项目 | 配置 |
|---|---|
| 显卡 | 4 张 A100 80GB，默认 GPU 0、1、2、3 |
| 每卡 batch | 10，总 batch 40 |
| 更新次数 | 58,700 |
| 样本输入总量 | 2,348,000 次，约 10 轮；与此前总 batch 80、29,350 步一致 |
| 训练集 | 当前 950 个片段、234,769 帧 |
| 初始化 | DINO 预训练；其余模块从头训练 |
| 优化器 | AdamW，学习率 3e-5，weight decay 0.01 |
| 学习率计划 | 2000 步线性 warmup，随后余弦下降；沿用 DT-Large 学习率 |
| 梯度裁剪 | max_norm=35 |
| 随机种子 | 0 |
| 权重保存 | 每 2000 步及最后一步，最多保留 5 个 |
| 日志 | 每 50 步；文本、TensorBoard、W&B |
| 预期最终权重 | `iter_58700.pth`，并核对 `latest.pth` 指向 |

已验证的模型参数量约 448.11M，其中冻结 86.35M、可训练 361.76M。选择每卡 batch 10 是为了保留显存余量；不要直接沿用 Base 的每卡 batch 20。

## 4. 已完成的验证及边界

- 9 项 backbone/配置测试：实际取层内容、拼接位置、分块前向一致性、冻结梯度边界、旧单层配置兼容性，以及 Large/S16/正常加速度配置。
- 1 项保存恢复测试：权重文件名、记录的迭代号与 AdamW 更新次数一致。
- 2 项监控测试：查询不启动任务；进程消失和截断日志不会被误报为完成。
- 完整分辨率、完整 12 层、四卡每卡 batch 10 的短时训练验证：损失和梯度有限，Adapter 有更新，Map 参数与损失均不存在。初次三步验证的最大显存预留约 61.52 GiB。
- 严格加载验证权重并与 DINO 预训练逐张量比较，主干权重保持完全一致；正常 ax、ay 来自实际数据；完整分辨率推理、清空记忆后的重复推理以及连续帧推理通过。
- 最终保存恢复验证使用 `verified_fresh` 和 `verified_resume`；机器可读结果以 `preparation/readiness.json` 为准。

准备过程中修正了当前工作区 `mmcv/runner/iter_based_runner.py` 的断点迭代号：原实现保存 `iter_N.pth` 时内部记录 N−1，恢复会重复一轮编号；现在保存已完成的更新次数 N，并做实际四卡续训验证。

这些验证说明训练和恢复链路可用，不代表模型已收敛或驾驶效果已得到验证。短时验证改用短训练上限和文本日志，不能据此报告正式训练时间或效果。长期显存、损失趋势和 ETA 由正式启动后的日志确认。现有 runner 续训恢复模型、优化器与步数，数据流、随机状态和模型历史缓存不保证逐位重现。

## 5. 后续 Codex 启动与监控入口

以下命令仅供之后启动时使用，本次准备任务没有执行正式 `start`。

```bash
cd /data5T/wenjj/Projects/DriveTransformer_worktrees/large_vitb3_nomap_s16_20260921/DriveTransformer
/home/wenjj/miniconda3/envs/drivetransformer/bin/python tools/large_vitb3_run.py check
```

`check` 只读检查：准备签收结果、关键源码哈希、数据/权重路径、四卡资源与预定命令。还需由启动代理核实当前实验队列没有预约或排他任务；资源被占用时不要强行启动，也不要终止其他实验。

确认具备启动条件后，运行：

```bash
nohup /home/wenjj/miniconda3/envs/drivetransformer/bin/python -u tools/large_vitb3_run.py start > preparation/formal_supervisor.log 2>&1 &
```

训练使用四卡端口 35531。已有 W&B 登录与环境设置保持不变；可沿用已有 `WANDB_MODE`。监督进程会保存 `run_state.json`，记录进程身份、命令、开始/结束时间与退出码。不要只凭 GPU 占用判断任务状态。

查询：

```bash
/home/wenjj/miniconda3/envs/drivetransformer/bin/python tools/large_vitb3_run.py status
```

输出包含当前迭代、百分比、最近 loss/grad_norm、每步时间、预计剩余小时、日志更新时间、最新权重、训练进程状态。ETA 应在正式数据流稳定后采用；最初几步不作为耗时承诺。

中断后恢复：

```bash
/home/wenjj/miniconda3/envs/drivetransformer/bin/python tools/large_vitb3_run.py check --resume-from work_dirs/large_vitb3_nomap_s16_res480_10ep/latest.pth
nohup /home/wenjj/miniconda3/envs/drivetransformer/bin/python -u tools/large_vitb3_run.py start --resume-from work_dirs/large_vitb3_nomap_s16_res480_10ep/latest.pth > preparation/formal_resume_supervisor.log 2>&1 &
```

恢复必须显式指定正式训练目录里的权重；启动工具会检查迭代号、AdamW 状态及整模严格加载，禁止使用 `preparation/` 中的验证权重。首次启动不允许覆盖已有训练记录。

训练完成需同时满足：退出码 0、日志到 58,700 步、最终权重存在。训练完成后仍需另行评估；本入口不会自动启动 CARLA、闭环评测或下一个实验。

## 6. 可直接交给后续 Codex 的任务

请先读取本交接文件以及服务器 `preparation/readiness.json`，使用指定工作区和配置。检查当前 GPU 0–3、实验队列、数据和源码签收后，再启动这一项正式训练并监控。保持正常 ax、ay，不增加加速度消融，不更换 4/8/12 取层、12 层 DT-Large、768 维、NoMap、S16 或训练预算。遇到显存不足、非有限损失、无进展或异常退出，先保留日志和断点并报告原因，不自行缩小模型或改变数据、评测规则。报告训练进度、ETA、最新权重与进程状态；完成训练后明确说明评估尚未启动。
