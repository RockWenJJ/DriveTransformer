DriveTransformer dev24 云端评测用法

2026-09-09。冻结的 dev24_v1 已接入 wenjj_alicloud_A100 的评测入口。默认评测全部 24 条；传入 `--shard 1` 至 `--shard 4` 时，每次评测对应的 6 条。四个分片互斥，合起来恰好是完整 dev24。

**入口与配置。** 以下两个入口执行同一套修复工作树代码：

```text
/data5T/wenjj/Projects/DriveTransformer/tools/evaluate_dev24.sh
/data5T/wenjj/Projects/Bench2Drive/eval_scripts/run_eval_dt_dev24.sh
```

入口固定使用 `/data5T/wenjj/Projects/DriveTransformer_worktrees/consistency_20260909/DriveTransformer` 中的 agent、控制器和模型代码。路线文件及冻结清单保存在该工作树的 `evaluation/routes/dev24_v1/`，同时在 Bench2Drive 的 `leaderboard/data/` 下提供同内容副本。

`/data5T/wenjj/Projects/DriveTransformer_worktrees/consistency_20260909/matrix_base/` 与 `matrix_dinob/` 中，C0–C5 共 12 份配置均已绑定 dev24。`tools/build_interface_matrix.py` 此后生成的配置也默认绑定这份名单。入口会核验 XML 哈希、路线数量与配置绑定；不一致时退出，避免误跑 dev94 或被改过的子集。

**先检查实际运行参数。** 在云服务器执行以下示例。`--gpu 0` 表示物理 GPU 编号，实际评测前应改为当时可用的 GPU；示例中的 `--dry-run` 只校验配置和输出运行参数，不加载模型、不启动 CARLA、不建立结果目录。

```bash
/data5T/wenjj/Projects/DriveTransformer/tools/evaluate_dev24.sh \
  --config /data5T/wenjj/Projects/DriveTransformer_worktrees/consistency_20260909/matrix_base/C1_eval_deterministic.py \
  --ckpt /data5T/wenjj/Projects/DriveTransformer/adzoo/drivetransformer/work_dirs/drivetransformer/drivetransformer_base_10ep/iter_24400.pth \
  --gpu 0 --run-name base_C1 --seed 0 --dry-run
```

确认资源可用后，去掉 `--dry-run` 才会开始真实闭环评测。DINO-B 使用 `matrix_dinob/` 中的对应配置和 `drivetransformer_base_dinov3_vitb16/iter_25000.pth` 权重。

**分片与结果。** 如需四份并行，分别加入 `--shard 1`、`--shard 2`、`--shard 3`、`--shard 4`，每份分配独立 GPU。默认服务端口按物理 GPU 编号区分，也可用 `--rpc-port` 和 `--tm-port` 显式指定。配对模型保持相同路线、seed 和分片方式；不要把同一模型的全量与分片结果重复计入。

结果目录为：

```text
/data5T/wenjj/Projects/Bench2Drive/eval_results/dev24_v1_<run-name>/seed<seed>_all/
/data5T/wenjj/Projects/Bench2Drive/eval_results/dev24_v1_<run-name>/seed<seed>_shard<N>/
```

其中 `results.json` 是原生评测结果，`eval_manifest.json` 记录路线、权重、解析后的配置、agent、控制器的哈希及 seed、入口命令。默认拒绝覆盖已有结果目录；有中断结果时可加 `--resume`，入口会先核验上述协议是否一致。

四分片结束后，用现有 `tools/merge_dev94_results.py` 的 `--routes` 指向完整 dev24 XML 合并，并核对唯一路线数为 24、无缺失或重复。该脚本按 XML 决定集合，文件名中的 dev94 不限定本次路线数量。dev24 用于初筛，入围配置仍需完整 dev94；原有 dev94 文件和入口保留供全量验证。

**本次验证。** 4 项路线测试通过，涵盖分片并集、XML 改动检测、过期配置拒绝和 dry-run 不启动评测。Base/DINO-B 的 12 份配置与全量/四分片组合，共 60 种路由检查通过；两个云端入口、后续配置生成器、带 seed 的评测器启动参数检查通过。已实际导入修复工作树的 agent 和控制器，并用 Bench2Drive 原生解析器解析主文件和四个分片，得到 24/6/6/6/6 条。尚未启动 CARLA 或加载权重评测，因此本次完成的是路线接入与启动前验证，未产生新的模型成绩。
