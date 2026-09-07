# 独立拆分验收

日期：2026-09-07。此次拆分从已完成优化的工作台提交
`786d554fa4215ae6e7be9481770408b3f3328748` 提取必要环境代码，建立新的独立 Git 历史。

## 精简与隔离

- 原工作台源码：103 个 Python 文件，21781 行。
- 独立环境：48 个 Python 源文件，约 7260 行；包含公共适配、核心、规则分组和完整工作台。
- 移除了研究训练器、冻结模型、检查点、实验账本、旧配置与历史输出；它们没有进入新 Git 历史。
- `.venv` 是从标准 Python 创建的干净虚拟环境，未启用 system-site-packages，未安装 Torch 或 Open-SCORE。
- 根目录 `make_env.py` 和可安装的 `had_env` 包提供同一工厂，图形依赖按需安装。

本机原研究目录 `E:\Code\Open_Score` 仍为旧研究 HEAD
`45b627627323a7b567a7331f2f4dcf76e07c8dfc`；源码、配置和运行任务未修改。
独立环境位于 `E:\Code\Open_Score_HAD_Workbench`，目录没有再次改名。

拆分前完整历史保存在本机
`E:\Code\.had-backups\workbench-before-standalone-20260907\workbench.bundle`。
旧子目录和文档也在该备份位置。新目录 `.git` 是独立版本库，不再引用原项目的 common Git directory。

## 物理与接口验证

`tests/data/physics_reference.json` 是从迁移前源码生成的独立原生物理基准：3 个场景、17 个真实物理步，
覆盖随机异构初态、uniform 初态和保护目标接触终局。保存原源码哈希、世界状态、观测、奖励和结果，
新测试不需要旧源码或旧 Git 历史。

规则分组迁移另做了 32 个完整回合逐步精确比对：
4v4 / 8v8 / 16v8 / 32v32 × rule / grand / static_rule / random × 种子 3 / 72。
状态、选择动作、奖励、done 和事件一致；4 个完整轨迹哈希保存在
`tests/data/grouping_extraction_validation.json`，供独立回归重算。

标准接口通过 PettingZoo 的 `parallel_api_test` 与 `parallel_seed_test`，同时覆盖离散和连续模式。
测试还检查动作整体校验、死亡最后转换、终局奖励不重复、采样截断不判胜、全局状态、
固定观测槽位、MPE 列表适配、渲染生命周期和独立随机性。

## 测试结果

干净环境完整回归：**109 passed，2 skipped，16.27 秒**。
两项跳过仅因为没有安装可选 Torch；相关 Torch 测试另外在备份中的隔离解释器执行，**9 项通过**。
包含不访问真实 GPU 的 CUDA 初始化边界检查：要求 GPU 上下文在策略适配器构造前就绪，
否则明确拒绝，避免首轮推理使用外部随机流。

Qt 离屏模式运行真实控件、事件循环和 spawn 工作进程，验证暂停、单物理步、宏边界、实时分支、
自动保存与退出；分支计算不会推进父会话。此处离屏只替换显示表面，不替换 HAD 物理或工作进程。

完整测试 XML 和机器可读安装验收保存在本机 `outputs/validation/`；输出不进入精简 Git 仓库。
源码包与 wheel 的来源标识都扫描实际安装文件；wheel 不把外层其他项目的 Git HEAD 当作本包提交。

## 可复验命令

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:SDL_VIDEODRIVER = 'dummy'
$env:SDL_AUDIODRIVER = 'dummy'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pip wheel . --no-deps -w dist
```

正常使用窗口前，移除上述 Qt/SDL 离屏环境变量。示例评估仅验证工具链，不是算法性能比较结论。
此前的物理热点提速和 Qt 显示优化被保留，但这次拆分不宣称新增训练吞吐或帧率收益。
