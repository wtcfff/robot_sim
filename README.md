# robot_sim



robot_sim 是基于 Isaac Gym 的机器人强化学习训练框架，用于在同一任务基准下，比较普通 PPO、HIM 和 DreamWaQ。项目将任务的物理模型、奖励、随机化、观测、动作和地形，与网络结构和优化算法分开组织，方便在同一个任务上切换训练方法，再把同一套方法扩展到其他任务。

当前内置 **Wheelleg 轮足机器人任务**，提供训练、PT（PyTorch 检查点）续训、固定场景评估、ONNX 导出、键盘控制、MuJoCo sim2sim 和地形预览工具。

> **方法范围**：`him` 实现速度监督与原型表示学习；`waq` 实现速度监督与变分重构，当前是 **不含 AdaBoot（自适应引导训练机制）的 CENet 变体**。CENet 是利用历史观测估计速度与环境隐变量的网络，当前策略使用潜变量均值。两者尚未完成原论文的逐项复现核验。当前仓库不提供已收敛策略或三种方法的性能排名。

快速导航：[项目结构](#项目结构) · [安装](#安装与运行环境) · [训练与续训](#训练与续训) · [地形](#地形选择与预览) · [导出与迁移](#模型导出与-sim2sim) · [添加任务](#如何添加新任务)

## 项目特点

- **同一任务，三种方法。** 更换 `--method` 即可切换训练方法，无需复制任务。奖励函数、随机化范围、地形、控制周期、基础观测和动作定义均来自同一份任务配置。
- **任务与方法分别扩展。** 仿真交互、历史和轨迹缓存放在 `simulation/`；策略、价值网络和编码器放在 `modules/`；PPO 与辅助损失更新放在 `algorithms/`。
- **实现与参数就近维护。** 每个任务的实现和配置放在同一目录。基类提供可复用行为与默认值，任务只覆盖自身差异，嵌套可变配置按实例独立复制。
- **地形实现共用，选择属于任务。** 一个地形程序提供九类地形，任务按固定编号选择类型与权重；训练和预览使用同一套生成代码。
- **完整保留 PT 训练产物。** 检查点同时保存网络规格、权重、优化器、任务配置和训练配置；ONNX 转换根据检查点重建模型，不重复手填任务维度。
- **显式记录迁移依据。** 导出时生成独立 YAML，列出关节映射、观测布局、控制参数、时间步和历史填充规则。MuJoCo 执行端读取 ONNX 与 YAML，无需创建 Isaac Gym 环境。

## 比较的内容与边界

PPO（Proximal Policy Optimization，近端策略优化）是三种方法共用的策略更新算法；HIM 和 WAQ 在此基础上增加各自的历史估计网络和辅助训练目标。

| 方法入口 | 策略可用信息 | 价值网络可用信息 | 辅助训练 |
|---|---|---|---|
| `ppo` | 当前基础观测 | 同一基础观测 | 无 |
| `him` | 当前基础观测、基础观测历史及其估计结果 | 任务提供的特权观测 | 速度监督、原型表示学习 |
| `waq` | 当前基础观测、基础观测历史及其估计结果 | 任务提供的特权观测 | 速度监督、下一帧观测重构、潜变量约束 |

普通 PPO 使用对称 actor-critic：策略网络和价值网络读取相同的可观测信息。“对称”不要求两者的隐藏层大小相同。特权观测是仿真器能够提供、部署时通常无法直接取得的状态或物理参数，仅用于历史方法的价值估计和辅助监督，不直接送入部署策略。

**三种方法的基础观测和动作定义相同，完整策略输入并不相同**：HIM/WAQ 额外接收历史，网络内部还会拼接估计速度和环境表示。比较结果同时反映信息利用方式、模型容量和训练目标的差异。

建议在相同任务配置、采样预算、评估场景和多个训练种子下比较。相同种子不保证不同网络消耗相同的随机数，也不保证 GPU 物理逐步完全确定，因此不能把一次训练视为统计结论。

## 项目结构

```text
.
├── robot_sim/                         # Python 包
│   ├── config.py                      # 共用训练默认值、网络参数与方法差异
│   ├── registry.py                    # 注册任务，按名称创建环境、网络和算法
│   ├── simulation/
│   │   ├── runner.py                  # 采样、训练调度、日志和 PT 检查点
│   │   ├── storage.py                 # 轨迹、历史缓冲和回报计算
│   │   ├── terrain.py                 # 九类共用地形的实现和编号解析
│   │   └── envs/
│   │       ├── base/
│   │       │   ├── base_env.py         # 仿真生命周期、公共缓冲和环境接口
│   │       │   ├── base_config.py      # 通用配置及独立配置实例
│   │       │   ├── legged_robot.py     # 迁移保留的足式机器人专用实现
│   │       │   └── legged_robot_config.py
│   │       └── wheelleg/
│   │           ├── wheelleg.py        # 观测、控制、奖励和随机化的具体实现
│   │           └── wheelleg_config.py # 本任务参数和地形选择
│   ├── modules/
│   │   ├── actor_critic.py             # 普通策略网络与价值网络
│   │   ├── him_actor_critic.py         # 组合历史估计器与策略/价值网络
│   │   ├── him_estimator.py            # HIM 速度与环境表示估计器
│   │   └── waq_estimator.py            # WAQ 速度估计与变分重构网络
│   ├── algorithms/
│   │   ├── ppo.py                     # 共用 PPO 更新
│   │   ├── him_ppo.py                 # HIM 辅助更新
│   │   └── waq_ppo.py                 # WAQ 辅助更新
│   └── utils/                         # 配置转换、种子与数学工具
├── scripts/
│   ├── train.py                       # 训练与续训入口
│   ├── evaluate.py                    # 固定场景评估与 JSON 输出
│   ├── export_onnx.py                  # PT 转 ONNX、数值校验、生成迁移 YAML
│   ├── keyboard_deploy.py              # Isaac Gym 中键盘控制，支持 PT/ONNX
│   ├── play.py                        # 复用播放循环，运行给定速度目标
│   ├── sim2sim.py                     # MuJoCo 中执行 ONNX
│   ├── view_terrain.py                # 使用共用生成器预览地形
│   └── sim2sim_configs/
│       └── wheelleg.yaml              # MuJoCo 目标模型与关节映射模板
├── resources/robots/wheelleg/
│   ├── meshes/                        # 机器人网格
│   └── urdf/                          # URDF、wheelleg.xml、scene.xml
├── tests/test_framework.py
├── docs/implementation.md             # 迁移差异与开发验证记录
├── setup.py
└── LICENSE
```

### 配置应该放在哪里

| 内容 | 维护位置 |
|---|---|
| 机器人资产、控制、指令、奖励、随机化、噪声、基础/特权观测维度 | 各任务的 `*_config.py` |
| 如何构建观测、施加控制、计算奖励与执行随机化 | 各任务的实现类，或适用的基类 |
| PPO 超参数、采样长度、网络层宽、历史长度、辅助损失参数 | `robot_sim/config.py` |
| 地形几何实现与固定编号 | `robot_sim/simulation/terrain.py` |
| 任务使用哪些地形、相对权重与难度规模 | 任务配置的 `terrain` 部分 |
| 目标仿真器 XML、刚体/关节/执行器名称、方向与偏置 | `scripts/sim2sim_configs/<task>.yaml` |
| 某个已训练模型实际使用的规格和完整迁移参数 | PT 检查点及其导出的 YAML |

`BaseEnv` 是环境基类，管理仿真器和公共接口；`BaseEnvCfg` 是配置基类，提供通用默认参数。任务类实现行为，任务配置类定义取值。`LeggedRobot` / `LeggedRobotCfg` 是面向足式机器人的进一步复用层，并不要求所有新任务继承它们。

## 安装与运行环境

训练采用 **NVIDIA Isaac Gym**，MuJoCo 用于 sim2sim。两者属于不同执行端；本仓库不包含 Isaac Gym 安装包。

### 环境准备

1. 安装适合目标机器的 NVIDIA 驱动、CUDA/PyTorch 和 [Isaac Gym](https://developer.nvidia.com/isaac-gym)，确认 Isaac Gym 示例可以运行。
2. 激活已经配置好的专用 Python 环境。下面以 `leggedgym` 为环境名，可替换为自己的名称。
3. 在仓库根目录安装源码包及所需工具依赖。

```bash
conda activate leggedgym

# 使用现有环境的 PyTorch / NumPy，以可编辑方式安装本项目。
python -m pip install -e . --no-deps
python -m pip install PyYAML tensorboard scipy

# ONNX 转换、验证与 MuJoCo 执行需要的可选依赖。
python -m pip install onnx onnxruntime mujoco
```

`--no-deps` 不会安装项目依赖，适用于已经配好 Isaac Gym、PyTorch 和 NumPy 的环境。新增工具依赖仍应与专用环境的 Python 版本兼容。Isaac Gym 进程必须先导入 `isaacgym`、再导入 `torch`，各仿真脚本已按此顺序组织。

下列命令均从仓库根目录运行；可编辑安装用于直接读取当前源码和仓库内机器人资源。

### 已验证的软件组合

| 组件 | 开发验证版本 |
|---|---|
| Python | 3.8.10 |
| Isaac Gym | Python 3.8 的 `gym_38` 绑定 |
| PyTorch | 2.4.1+cu121 |
| NumPy / SciPy | 1.23.5 / 1.10.1 |
| ONNX / ONNX Runtime | 1.15.0 / 1.19.2 |
| MuJoCo | 3.2.3 |

GPU 验证使用 NVIDIA RTX 3060 Laptop GPU（6 GB）。ONNX 数值校验和 MuJoCo 策略推理使用 ONNX Runtime 的 CPU 执行提供器。上表是已测试组合，不代表已经验证任意其他版本。

## 训练与续训

### 选择任务和方法

以下命令在相同 Wheelleg 配置下分别选择三种方法。可先选择其中一条运行：

```bash
python -m scripts.train --task wheelleg --method ppo --num-envs 128 --seed 1 \
  --iterations 1000 --headless --log-dir logs/wheelleg/ppo/run_01

python -m scripts.train --task wheelleg --method him --num-envs 128 --seed 1 \
  --iterations 1000 --headless --log-dir logs/wheelleg/him/run_01

python -m scripts.train --task wheelleg --method waq --num-envs 128 --seed 1 \
  --iterations 1000 --headless --log-dir logs/wheelleg/waq/run_01
```

`--num-envs` 指并行环境数量；`--iterations` 指本次执行的采样与更新轮数。每轮默认让各环境采样 48 个策略步，再进行 PPO 更新。示例中的 1000 轮用于演示完整流程，不是保证收敛的训练预算。

Wheelleg 配置保留 4096 个环境的默认值，实际数量应根据显存调整。建议先用较小数量确认链路，再扩大规模；目前短程训练验证覆盖 8 个环境，128 和 4096 并行规模尚未验证。有限显存机器不宜同时启动多个 Isaac Gym 训练进程。

常用参数：

| 参数 | 作用 |
|---|---|
| `--task` | 已注册的任务名，目前内置 `wheelleg` |
| `--method` | `ppo`、`him` 或 `waq` |
| `--num-envs` | 覆盖本次并行环境数量 |
| `--steps-per-env` | 每轮每个环境采样的策略步数 |
| `--iterations` | 本次执行的更新轮数，续训时为追加轮数 |
| `--seed` | 新训练使用的随机种子，续训读取保存值 |
| `--terrain-ids` | 本次启用的地形编号，覆盖任务选择 |
| `--device` | 仿真与训练设备，默认 `cuda:0` |
| `--headless` | 不创建图形窗口 |
| `--log-dir` | 指定实验输出目录 |
| `--resume` | 从本框架的 PT 检查点恢复训练 |

### 训练数据流

`train.py` 读取任务与方法配置，通过环境工厂创建仿真任务，再由 `Runner` 调度采样和更新：

1. 环境重置，返回基础观测和可选的特权观测。历史方法使用本次实际初始观测填充历史。
2. 策略根据当前基础观测，以及方法所需的历史，生成动作。
3. 环境执行多个物理步，计算奖励和下一状态，并标记终止或截断。
4. 轨迹缓存保存观测、动作、奖励、价值等数据；达到采样长度后计算回报和广义优势估计，即用于衡量动作优劣的训练信号。
5. PPO 更新策略和价值网络；HIM/WAQ 另外更新历史估计器。按保存间隔写入 PT，结束时也保存一次。

历史缓存按环境独立重置，避免不同回合的数据混合。环境会保留自动重置之前的最终观测，用于时间/边界截断的价值估计；跨回合的观测不会组成估计器的时序训练对。

### 日志与检查点

未指定 `--log-dir` 时，默认目录为 `logs/<task>/<method>/<时间>/`：

```text
logs/wheelleg/him/<时间>/
├── config.json                    # 本次任务、训练和模型规格
├── metrics.jsonl                  # 每轮奖励和损失
├── events.out.tfevents.*           # TensorBoard 事件
├── model_100.pt
└── ...
```

PT 包含模型规格、网络权重、优化器、任务/训练配置、部署快照和累计训练计数。可通过以下命令查看曲线：

```bash
tensorboard --logdir logs
```

### 从 PT 续训

```bash
python -m scripts.train --resume logs/wheelleg/him/run_01/model_1000.pt \
  --iterations 500 --headless --log-dir logs/wheelleg/him/run_01_resume
```

此处追加 500 轮，完成后累计到第 1500 轮。续训从检查点读取任务、方法和配置，并恢复网络、优化器及计数。当前实现要求环境配置和网络规格与检查点一致；不要在续训时切换方法、任务、环境数量或地形布局。

续训会重新创建并重置仿真环境，不保存物理现场和随机数状态，因此不保证后续轨迹与从未中断的训练逐步一致。保存配置中的 `iterations` 是默认预算，实际完成量应查看检查点的 `iteration`、`total_steps` 和日志。

## Wheelleg 任务

Wheelleg 是内置参考任务，框架中的 runner、PPO 和网络工厂可供其他任务复用。它的奖励实现、奖励权重、随机化范围、噪声、初始状态、指令和控制参数从 Wheel-Legged-Gym 参考任务迁移而来。

### 观测与动作

| 基础观测块 | 维度 |
|---|---:|
| 机体系角速度 | 3 |
| 机体系重力方向 | 3 |
| 指令 `[前向速度, 横向速度占位, 偏航速度, 高度]` | 4 |
| 六个腿关节相对默认姿态的位置 | 6 |
| 八个关节速度 | 8 |
| 上一次动作 | 8 |
| **合计** | **32** |

三种方法均输出 8 维动作：六个腿关节使用位置 PD 控制，根据目标角度与速度反馈计算力矩；两个轮关节使用速度比例控制。物理步长 2.5 ms，每 4 个物理步更新一次策略，策略周期为 10 ms。

普通 PPO 的策略与价值网络都读取 32 维基础观测。HIM/WAQ 默认使用 5 帧历史（160 维），估计 3 维速度和 16 维环境表示，拼接后策略多层感知机（MLP，由全连接层组成的网络）输入为 51 维；价值网络读取 259 维特权观测，其中包含 77 个地形高度采样点及动力学、接触、随机化真值等。

### 随机化与历史初始化

任务包含质量/质心、接触摩擦/恢复系数、关节摩擦、控制增益、电机强度、力矩噪声、动作/观测延迟、外力扰动和初态随机化。具体实现和取值分别位于 `wheelleg.py` 与 `wheelleg_config.py`，三种方法复用同一套任务设置。

历史默认按“最旧 → 最新”排列，每个策略步采样一次，包含当前帧。首次启动和每次重置采用相同规则：**重复本回合实际初始观测**。没有初态随机化时，它对应配置的默认姿态；启用初态随机化时，应使用随机化后的状态，不用写死的默认姿态代替。若增大历史采样间隔，只在相应采样步推入新帧。

## 地形选择与预览

`simulation/terrain.py` 集中实现地形，任务配置决定启用哪些类型。编号按生成器的固定类型顺序定义，不因某些类型被关闭而重新编号：

| 编号 | 类型名 | 地形 |
|---|---|---|
| 0 | `flat` | 平地 |
| 1 | `smooth_slope` | 平滑斜坡，含正负坡度 |
| 2 | `rough_slope` | 粗糙斜坡，含正负坡度 |
| 3 | `stairs_negative` | 负台阶：中心低，向外爬升 |
| 4 | `stairs_positive` | 正台阶：中心高，向外下降 |
| 5 | `discrete_obstacles` | 离散障碍物 |
| 6 | `stepping_stones` | 踏石 |
| 7 | `gap` | 环形沟隙 |
| 8 | `pit` | 中央方形坑 |

通用地形默认 `selected_ids = None`，表示全选。Wheelleg 显式使用 `[0, 1, 2, 5]`，保留参考任务的启用集合。可在任务配置中覆盖：

```python
class terrain(LeggedRobotCfg.terrain):
    selected_ids = [0, 1, 2, 5]       # None 表示九种全选
    weights = None                  # 所选类型等权
```

也可设置 `weights = {0: 1.0, 1: 2.0, 2: 1.0, 5: 1.0}`。键必须恰好对应所选编号，权重为正数。生成器保证每种类型至少一列，因此 `num_cols` 不能小于类型数，有限列数下实际比例会有离散误差。

公共默认尺寸迁自 Wheelleg：每块 8×8 m，10 行、18 列，水平分辨率 0.1 m，垂直分辨率 0.005 m。各任务可自行覆盖尺寸、材料和难度规模。

```bash
# 查看任务当前选择的地形。
python -m scripts.view_terrain --task wheelleg

# 使用同一生成器查看九类地形的第 0、9 级。
python -m scripts.view_terrain --task wheelleg \
  --terrain-ids 0 1 2 3 4 5 6 7 8 --levels 0 9

# 训练时覆盖地形选择。
python -m scripts.train --task wheelleg --method ppo --num-envs 128 \
  --terrain-ids 0 1 2 --headless
```

预览按所选类型重新排布，每种一列，不推进训练或物理步。`--levels` 使用任务原始难度行号，难度为 `行号 / 原始行数`。无窗口检查可追加 `--headless --frames 3`。

## 模型导出与 sim2sim

### PT 转 ONNX

以训练得到的 HIM 检查点为例：

```bash
python -m scripts.export_onnx --checkpoint logs/wheelleg/him/run_01/model_1000.pt
```

默认在该实验目录下生成：

```text
exported/
├── model_1000.onnx                 # 编码器（如有）和确定性策略
└── model_1000.yaml                 # 完整、可独立检查的迁移配置
```

脚本从 PT 中的模型规格重建网络并加载权重，执行 ONNX 格式检查，以及 batch=1、7 的 PyTorch/ONNX Runtime 数值对比。训练仍正常保存 PT，导出不会覆盖它。

| ONNX 张量 | 一般形状 | Wheelleg 示例 |
|---|---|---|
| `obs` | `[batch, obs_dim]` | `[batch, 32]` |
| `obs_history`，仅历史方法 | `[batch, history_length × obs_dim]` | `[batch, 160]` |
| `actions` | `[batch, action_dim]` | `[batch, 8]` |

批量维度可变，观测/动作维度由具体检查点决定。无需给每个任务另建一份导出维度配置。HIM/WAQ 的 ONNX 包含执行策略所需的历史编码器，不包含价值网络和仅用于训练的分支。

### YAML 从哪里来

导出时合并两类信息，另存为模型旁边的完整 YAML：

| 来源 | 提供的内容 |
|---|---|
| PT 的训练/部署快照 | 观测块与缩放、实际关节顺序、默认姿态、控制增益、动作缩放和限幅、时间步、历史规则 |
| `scripts/sim2sim_configs/<task>.yaml` | 目标 XML、基座名称、关节/执行器名称、方向和偏置 |

可用 `--template` 选择其他目标映射模板。任务未提供部署快照时，脚本只生成 ONNX；有部署快照但找不到目标模板时，会提示 YAML 生成失败。

完整 YAML 明确记录历史帧数、单帧维度、展开顺序、采样间隔，以及启动/重置填充规则。它是检查 sim2sim 对齐、后续扩展 sim2real 时的配置依据。MuJoCo 执行时不重新读取任务默认配置，也不加载 PT。

关节位置转换遵循 `q_policy = sign × q_target + offset`，速度与力矩按相应符号转换。观测缺项不能自动猜测；确有占位维度时，可在观测块中使用 `source: constant` 明确给出 `value`、`size` 和 `scale`。

### 在 MuJoCo 中运行

```bash
python -m scripts.sim2sim --config logs/wheelleg/him/run_01/exported/model_1000.yaml

# 不创建窗口，执行有限策略步。
python -m scripts.sim2sim --config logs/wheelleg/him/run_01/exported/model_1000.yaml \
  --headless --steps 1000
```

Wheelleg 使用 `resources/robots/wheelleg/urdf/scene.xml` 作为场景入口，它包含同目录下的 `wheelleg.xml`，并通过 `../meshes` 引用网格。XML 和网格随机器人资源一起提供。

`--policy` / `--xml` 可覆盖 YAML 的模型路径。ONNX 默认相对 YAML 所在目录解析，XML 相对项目根目录解析。迁移到其他机器时需保留所用资源，或显式调整路径。

当前 MuJoCo 运行器支持已实现的机器人本体观测块，以及逐关节位置、速度或力矩控制；目标执行器要求单位传动比。其他传感器、控制形式或传动机制需要扩展执行端。真实机器人通信尚未实现。

## 播放、键盘控制与评估

### Isaac Gym 播放

```bash
python -m scripts.play --checkpoint logs/wheelleg/him/run_01/model_1000.pt --speed 0.5

python -m scripts.keyboard_deploy --checkpoint logs/wheelleg/him/run_01/model_1000.pt

# 在相同 Isaac Gym 执行循环中改用 ONNX 推理。
python -m scripts.keyboard_deploy --checkpoint logs/wheelleg/him/run_01/model_1000.pt \
  --onnx logs/wheelleg/him/run_01/exported/model_1000.onnx
```

PT 用于恢复任务配置；指定 `--onnx` 时动作由 ONNX 推理产生。播放使用名义物理与控制参数，关闭训练随机化和观测噪声，在平地运行，不改动保存的训练配置。

| 按键 | 作用 |
|---|---|
| W / S | 前进 / 后退 |
| A / D | 左转 / 右转 |
| 空格 | 将运动目标置零 |
| R | 重置 |

Isaac Gym 按住键时更新目标，并按 `--command-ramp-time` 平滑改变指令。MuJoCo 原生窗口使用按键回调，当前为按键设置目标、空格置零。两种执行端均可使用 `--headless --steps <数量>` 检查无窗口运行。

### 固定场景评估

```bash
python -m scripts.evaluate --checkpoint logs/wheelleg/him/run_01/model_1000.pt \
  --steps 2000 --num-envs 16 --seed 123
```

评估根据检查点的任务设置和固定种子重建地形，关闭训练随机化/噪声，冻结地形与指令课程，使用任务提供的指令序列和指标。Wheelleg 每 5 s 切换静止、前进、转向、后退四个阶段；2000 个策略步覆盖一个 20 s 周期。

结果默认写入检查点旁的 `.evaluation.json`，包括速度/偏航/高度误差、平均奖励、回合结束和失败次数，以及训练配置与实际评估配置。可以用 `--output` 指定已有目录中的输出文件。

比较不同模型时应保持任务设置、评估种子、环境数量和评估时长一致。该脚本是固定协议下的检查工具，当前不包含自动多种子汇总或显著性检验，也不替代随机扰动鲁棒性评估。

## 如何添加新任务

### 1. 建立任务实现与配置

```text
robot_sim/simulation/envs/new_task/
├── __init__.py
├── new_task.py                     # NewTask 环境类
└── new_task_config.py              # NewTaskCfg 配置类

resources/robots/<robot>/           # 如需新机器人，放置其资产
```

独立任务可以继承 `BaseEnv` / `BaseEnvCfg`，自行实现物理、观测与控制；与现有足式实现相符的任务可继承 `LeggedRobot` / `LeggedRobotCfg`。选择基类前应检查其观测、关节索引、控制与运动学假设，不要将 Wheelleg 的实现直接视为任意机器人模板。

在任务配置中定义资产、指令、控制、奖励、随机化和地形选择，明确 `env.num_observations`、`env.num_actions` 与 `env.num_privileged_obs`。维度应由本任务实际布局确定，不沿用 Wheelleg 的 32/8/259。新任务的奖励和参数改动应限定在自身配置中，避免改动其他任务的基类默认值。

`BaseConfig` 会将嵌套配置实例化并深复制可变值。定义子配置时仍需明确是继承已有字段，还是完全替换一个配置分组；不要额外建立共享的全局可变参数对象。

### 2. 满足训练接口

当前 runner 采用项目自身的批量接口，并非 Gymnasium 的五返回值接口：

```python
obs, privileged_obs = env.reset()
obs, privileged_obs, rewards, dones, info, legacy_history = env.step(actions)
```

| 接口/数据 | 约定与作用 |
|---|---|
| `num_envs`、`num_obs`、`num_actions`、`num_privileged_obs` | 环境数量和各张量维度，供 runner 构建网络与缓冲 |
| `obs` | `[N, obs_dim]`，三种方法共用的基础观测 |
| `privileged_obs` | `[N, critic_dim]`，历史方法的价值输入；普通 PPO 不使用 |
| `actions` | `[N, action_dim]`，策略动作，由任务转换成实际控制量 |
| `rewards` / `rew_buf` | `[N]`，本策略步奖励；runner 同时用 `rew_buf` 记录奖励 |
| `dones` | `[N]`，当前步是否结束回合 |
| `info['final_obs']` / `info['final_privileged']` | 自动重置前的下一状态，不能替换成新回合初始状态 |
| `info['terminated']` / `info['truncated']` | `[N]`，区分失败终止和时间/边界截断 |
| `legacy_history` | 保留的迁移兼容返回项，当前 runner 不使用 |
| `close()` | 释放仿真与查看器资源 |

各张量应与所选训练设备一致。即使只训练普通 PPO，也需要满足 runner 对最终观测和截断标记的约定。

**当前 HIM/WAQ 估计器还要求**：环境的 `base_lin_vel` 提供 `[N,3]` 的真实机体系线速度标签，并提供特权价值输入。下一帧基础观测用于表示学习或重构。没有三维速度这一监督量的任务，需要先适配估计目标及辅助更新，不能仅注册任务就直接套用现有估计器。

### 3. 注册任务

在 `robot_sim/registry.py` 的 `_TASKS` 中加入惰性导入条目：

```python
'new_task': (
    'robot_sim.simulation.envs.new_task.new_task', 'NewTask',
    'robot_sim.simulation.envs.new_task.new_task_config', 'NewTaskCfg',
),
```

四个字段分别是环境模块、环境类名、配置模块、配置类名。测试或单独程序也可以调用 `register_task(name, env_class, config_class)` 临时注册；若要让各脚本启动后都能发现任务，应把注册接入共同导入入口。

实现接口并注册后，同一任务即可通过三种方法入口训练：

```bash
python -m scripts.train --task new_task --method ppo --headless
python -m scripts.train --task new_task --method him --headless
python -m scripts.train --task new_task --method waq --headless
```

无需复制 runner、轨迹缓存、PPO 或地形生成器。任务之间共用网络实现，实际输入输出维度由各自配置传给网络工厂。

### 4. 按需接入评估与部署

| 能力 | 新任务需提供的适配 |
|---|---|
| 关闭训练随机化进行播放/评估 | 配置类的 `disable_training_randomization()` |
| 外部指令控制 | 环境类的 `set_external_commands(command)`，与脚本的指令顺序对齐 |
| 固定场景评估 | `evaluation_commands(time_s)`、`evaluation_metrics()`，并在 `info` 中返回重置前的评估指标 |
| ONNX 对齐快照 | `deployment_config()`，记录实际观测块、关节顺序、名义控制与初始状态 |
| MuJoCo | `scripts/sim2sim_configs/new_task.yaml` 及对应 XML/网格；运行器需支持该任务的观测和控制 |
| 地形预览 | 与现有足式预览脚本的地形、资产创建接口兼容，或增加对应适配 |

目标映射模板只维护目标端差异，观测/动作维度与历史参数继续来自训练检查点。当前键盘和地形预览工具面向机器人运动任务，注册一个任意任务并不意味着它自动具备这些交互能力。

## 验证与当前状态

在已经配好 Isaac Gym、GPU 和部署依赖的环境中运行：

```bash
python -m unittest tests.test_framework -v
```

已有开发验证覆盖：

- 配置实例隔离、九类地形生成、网络信息路由与维度；
- HIM/WAQ 辅助更新、历史采样与独立重置、回合边界回报计算；
- GPU 环境重置/截断，以及 Isaac Gym 与 MuJoCo 的观测和控制转换；
- 三种方法的短程 GPU 更新、PT 保存、ONNX 数值校验和 MuJoCo 无窗口执行；
- PT 续训、固定评估 JSON 输出和 Isaac Gym 的 ONNX 播放。

参考任务源码比对测试仅在开发机的参考目录可用时执行，其他环境会跳过该项；其余检查仍需满足对应软件和 GPU 条件。详细迁移差异和开发记录见 [docs/implementation.md](docs/implementation.md)。

本地可能存在 `logs/validation/wheelleg/`，该目录已被 Git 忽略，不是仓库提供的预训练权重。其 `ppo_resume/model_2.pt` 是从 PPO 第 1 轮检查点恢复后追加 1 轮的续训检查产物：8 个环境、每轮各 4 步、累计 64 条状态转移，每轮将采样数据遍历 1 次（1 个 epoch）、分为 2 个小批次。它只验证恢复流程，不代表已学会运动，也不是第四种训练方法。

当前尚未完成完整训练收敛、三种方法的多种子性能比较、4096 并行规模、图形键盘交互以及真实机器人验证。短程检查通过仅说明相关代码链路可运行。

## 项目参考

- [Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [本末tita](https://github.com/DDTRobot/tita_rl)
- [逐际动力tron2](https://github.com/limxdynamics/tron2_rl_lab)
- [华南虎26步兵](https://github.com/scutrobotlab/wheeled-legged_RL)

## 许可证

代码许可说明见 [LICENSE](LICENSE)