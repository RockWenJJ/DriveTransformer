# DriveTransformer 项目上下文（长期实验记录，供 Claude Code 参考）

> 由本地 Claude 会话于 2026-08-27 依据记忆整理。涵盖机器环境、全部实验结果、路径约定与踩坑记录。

## 机器与环境

- 机器: `wenjj@120.25.201.195`（多用户共享，8×GPU 可见）。所有数据在 `/data5T/wenjj`；`~/Projects`、`~/miniconda3`、`~/carla` 均为指向它的符号链接。
- conda 环境（`/data5T/wenjj/miniconda3`）:
  - `drivetransformer`: torch 2.1.0+cu118。训练、闭环评测、开环评测统一用它（eval 脚本 `conda activate drivetransformer`）。
  - `orion`: torch 2.4.1+cu118（Orion 项目用）。
  - `b2d_zoo`: py3.8, 仅 CARLA client 依赖，**没有 torch 是正常的**。
- CARLA 0.9.15: `~/carla`，无 root 启动方式 `~/vulkan-local` + `carla-headless.sh`。
- Bench2Drive 数据 416G: `/data5T/wenjj/bench2drive`（本仓库 `data/bench2drive` 已链接）。
- nuScenes: 正在从百度网盘下载到 `/data5T/wenjj/nuscenes`（2026-08-27 启动，tmux 会话 `nuscenes_dl`，脚本 `/data5T/wenjj/tmp_scripts/nuscenes_baidu.sh`，幂等可断点续跑）。**注意：缺 `v1.0-trainval_meta.tgz`（标注元数据），网盘里还没有，用户会补转存。**
- 旧机器 8.147.110.55 已于 2026-08-27 核验后释放，不要引用它。

## 关键路径（本仓库）

- 训练 ckpt 在**仓库内部**（不是仓库根的 work_dirs）: `adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_{base,base_lowres,base_camdrop}/iter_100000.pth`，每个模型另有 iter_96000/99000 中间 ckpt。base/camdrop 各 5G，lowres 12G（含 7 个 ckpt）。
- 闭环评测结果: `/data5T/wenjj/Projects/Bench2Drive/eval_results/`（57 个目录：dt_base_merged、dt_lowres_dev94_merged、各分块 dt_*_A/B_f0..f3 等）。
- 闭环评测基础设施: `/data5T/wenjj/Projects/Bench2Drive/eval_scripts/`（`slot_worker.sh gpu rpcport tmport jobfile`；`run_eval_dt.sh config+ckpt+gpu+xml+run_name`）。
- 开环评测脚本: `/data5T/wenjj/tmp_scripts/ol_eval.py`（**仓库本身没有开环评测**，自建；用法 `--config --ckpt --out --limit`，结果在仓库 `ol_eval_results/`）。
- 未入 git 的配置: `adzoo/drivetransformer/configs/drivetransformer/drivetransformer_base_lowres.py` 与 `drivetransformer_base_camdrop.py`。

## 实验历史（2026-08，全部完成，GPU 空闲）

三个模型都是 base 结构、100k iters 训练：`base`（基准）、`base_lowres`（低分辨率图像增广，8/25 08:58 完成，final loss ~22.5）、`base_camdrop`（随机丢摄像头增广）。

**闭环（Bench2Drive）dev94 对比**（94 条路线；dt_base 需从其 220 条记录文件中按 dev94 route_id 过滤）：

| 模型 | DS | RC | 完成率 | 成功率 | 能力均值 |
|---|---|---|---|---|---|
| dt_base | 50.76 | 87.36 | 71.3% | 19.1% | 19.51 |
| dt_lowres | 49.67 | 78.97 | 53.2% | 17.0% | 16.28 |
| dt_camdrop（全量 220） | 43.49 | — | — | — | — |

lowres 能力分: OT 5.6 / MG 12.5 / EB 33.3 / GW 30.0 / TS 0.0（38 条 TS 全失败）；base: OT 22.2 / MG 9.4 / EB 33.3 / GW 30.0 / TS 2.6。

**开环（log-replay）val split 对比**（12806 样本 / 50 场景，L2 误差 @1s/2s/3s，米；碰撞率%）：

| 模型 | L2@1s | L2@2s | L2@3s | 平均 | 碰撞率 |
|---|---|---|---|---|---|
| dt_lowres | 0.61 | 1.75 | 3.22 | **1.86** | **1.64** |
| dt_base | 0.66 | 1.98 | 3.62 | 2.09 | 1.85 |
| dt_camdrop | 0.70 | 2.23 | 4.13 | 2.35 | 2.90 |

**核心结论**: ①开环/闭环发散——lowres 开环最好但闭环更差（完成率 -18pt、TS 全挂）；②camdrop 两头都最差；③两个增广都不如 base。早期全量参考: dt_large DS 63.23 / success 35.45%，dt_base 全量 DS 49.68 / RC 84.56 / success 20.91%（eval_results/dt_base_merged）。

## 用户训练偏好（务必遵守）

1. 永远用 `drivetransformer_base`，**不要用 large**（太慢，用户明确拒绝过）。
2. micro-batch 默认 20，OOM 时**不要问，自动**按 20→16→14→12 回退（调 `samples_per_gpu`）。
3. 默认 20 epochs；micro=20 双卡时 5000 iters/epoch，max_iters=100000。micro=20@A100 80G 约占 61G，~3.46s/iter。

## 开环评测技术要点（ol_eval.py）

- GT `ego_fut_trajs_fix_time` 是 ego 系下绝对坐标，30 点 @0.1s，索引 = 10t−1（1s→9, 2s→19, 3s→29）。
- agent 未来轨迹在 `gt_attr_labels[:, :12]`（6 点 @0.5s，agent 自车系，存储为 R@traj，R=rot(yaw+π)，逆变换用 R^T）；碰撞判据 = ego 中心距 agent 未来中心 <2.0m。
- 预测取 `out[0]["ego_fut_preds_fix_time"][0, 0]`，与 GT 同坐标系。
- 流式记忆重置: forward_test 在 scene_token 变化时重置；脚本注入 `f"{token}#w{window}"`，window 在 (token 变化 OR not prev_exists) 时递增。

## 已知坑（重要）

- `eval_results/dt_base_dev94/results.json` 实际包含全部 220 条记录——比较前必须按 dev94 的 route_id 集合过滤。
- CARLA 相邻 RPC 端口会冲突（bind: Address already in use → segfault + evaluator 挂死）: 端口间隔 ≥10（如 30750/60/70/80），启动错峰 45s。
- `pkill -f` 的模式若出现在自己的 ssh 命令行里会杀死自己的 shell → 用括号模式如 `"leaderboard_[e]valuator"`、`"CarlaUE4[.]sh"`。
- forward_test 对裸 bool/int 键（prev_exists、index）会在 unwrap 循环崩溃 → 输入按 `INPUT_KEYS` 白名单过滤。
- shell 陷阱: `cd X && A & B &` 会把整条链后台化，B 拿不到变量和 cwd。
- `drivetransformer_base_lowres[.]py` 这类 config 路径的 pgrep 模式会误匹配评测进程。
- Claude Code 安装: 官方 install.sh 对本机 IP 区域屏蔽，升级需从别的机器拷贝 `~/.local/share/claude-code/`（当前 2.1.232）。
