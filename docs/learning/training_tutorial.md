# Microduck RL 训练教程：从零开始训练一个走路策略

本教程说明如何基于 mjlab 框架，从头定义并训练一个 Microduck 机器人的走路策略。

## 目录结构概览

```
mjlab (第三方框架，不修改)
├── envs/              # 环境基类、管理器
├── tasks/velocity/    # 标准速度任务模板
├── rl/                # PPO 算法
└── ...

mjlab_microduck (项目自定义层)
├── robot/
│   ├── microduck/              # MJCF 模型文件 (XML)
│   │   ├── robot_walk.xml      # 走路模型
│   │   └── robot_groundcontact.xml
│   └── microduck_constants.py  # 机器人配置
├── tasks/
│   ├── mdp.py                  # 奖励/观测/事件函数
│   ├── microduck_velocity_env_cfg.py  # 任务配置
│   └── __init__.py             # 任务注册
└── export.py                   # ONNX 导出
```

---

## 第一步：定义机器人模型（MJCF XML）

**文件位置**：`src/mjlab_microduck/robot/microduck/robot_walk.xml`

MJCF (MuJoCo XML) 定义机器人的物理结构：

```xml
<mujoco model="microduck">
  <compiler angle="radian" meshdir="meshes/" />
  
  <default>
    <joint damping="0.1" armature="0.001" />
    <geom contype="1" conaffinity="1" condim="3" friction="1.0 0.3 0.3" />
  </default>
  
  <worldbody>
    <body name="trunk_base" pos="0 0 0.12">
      <freejoint name="root" />
      <inertial pos="0 0 0" mass="0.8" diaginertia="0.001 0.001 0.001" />
      <geom name="trunk_collision" type="mesh" mesh="trunk" />
      
      <!-- 左腿 -->
      <body name="left_hip_yaw" pos="0 0.035 0">
        <joint name="left_hip_yaw" type="hinge" axis="0 0 1" range="-0.5 0.5" />
        <geom type="mesh" mesh="hip_yaw" />
        
        <body name="left_hip_roll" pos="0 0 0">
          <joint name="left_hip_roll" type="hinge" axis="1 0 0" range="-0.3 0.3" />
          <!-- ... 更多关节 ... -->
        </body>
      </body>
      
      <!-- 右腿（镜像） -->
      <!-- ... -->
      
      <!-- 头部 -->
      <body name="neck_pitch" pos="0.05 0 0.03">
        <joint name="neck_pitch" type="hinge" axis="0 1 0" range="-0.5 0.5" />
        <!-- ... -->
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <!-- 位置控制执行器 -->
    <position name="left_hip_yaw_act" joint="left_hip_yaw" kp="10" />
    <!-- ... 其他关节 ... -->
  </actuator>
</mujoco>
```

**关键点**：
- 14 个主动关节（12 腿 + 2 头颈）
- 关节命名约定：`left_hip_yaw`, `right_knee` 等
- 被动关节（如齿轮间隙）以 `passive_` 开头

---

## 第二步：定义机器人配置

**文件位置**：`src/mjlab_microduck/robot/microduck_constants.py`

```python
from pathlib import Path
import mujoco
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import DelayedActuatorCfg
from mjlab.utils.spec_config import CollisionCfg

# 1. 加载 XML 的函数
_ROBOT_DIR = Path(__file__).parent / "microduck"

def get_walk_spec() -> mujoco.MjSpec:
    """加载走路模型的 MJCF 规格"""
    return mujoco.MjSpec.from_file(str(_ROBOT_DIR / "robot_walk.xml"))

# 2. 初始姿态（HOME 位置）
HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        r".*hip_yaw.*": 0.0,
        r".*left_hip_roll.*": -0.1,
        r".*right_hip_roll.*": 0.1,
        r".*left_hip_pitch.*": -0.3,
        r".*right_hip_pitch.*": 0.3,
        r".*left_knee.*": 0.0,
        r".*right_knee.*": 0.0,
        r".*left_ankle.*": 0.3,
        r".*right_ankle.*": -0.3,
        r".*neck_pitch.*": 0.0,
        r".*head_pitch.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

# 3. 碰撞配置
FULL_COLLISION = CollisionCfg(
    geom_names_expr=[r".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, r".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# 4. 执行器配置（BAM 电压控制模型）
from mjlab_microduck.actuator import FrictionDRBamActuatorCfg

actuators = FrictionDRBamActuatorCfg(
    motor_name="xl330",           # 舵机型号
    model="m6",                   # 摩擦模型复杂度
    target_names_expr=(r"^(?!passive_).*",),  # 排除被动关节
    kp_fw=200.0,                  # 固件 P 增益
    vin_range=(6.5, 8.2),         # 电压范围
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=6.0,
    delay_min_lag=3,              # 指令延迟（仿真步）
    delay_max_lag=6,
)

# 5. 组装机器人配置
MICRODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,  # 关节限位软边界
    ),
)
```

**关键点**：
- `EntityCfg` 封装了机器人的所有物理属性
- `FrictionDRBamActuatorCfg` 使用 BAM 模型模拟真实舵机物理
- `HOME_FRAME` 定义初始姿态，影响训练收敛

---

## 第三步：定义 MDP 函数

**文件位置**：`src/mjlab_microduck/tasks/mdp.py`

MDP (Markov Decision Process) 函数定义奖励、观测、事件等。

### 3.1 奖励函数示例

```python
import torch
from mjlab.envs import ManagerBasedRlEnv

def feet_air_time_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.125,
    threshold_max: float = 0.300,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """鼓励左右脚腾空时间对称（步态对称性）
    
    Args:
        env: 环境实例
        sensor_name: 接触传感器名称
        threshold_min: 最小腾空时间阈值（秒）
        threshold_max: 最大腾空时间阈值（秒）
        command_threshold: 速度命令阈值（低于此值不奖励）
    
    Returns:
        奖励值，形状 (num_envs,)
    """
    # 从接触传感器获取腾空时间
    contact_sensor = env.sensors[sensor_name]
    air_time_left = contact_sensor.data.air_time[:, 0]   # 左脚
    air_time_right = contact_sensor.data.air_time[:, 1]  # 右脚
    
    # 计算对称性：左右脚腾空时间差越小越好
    symmetry = 1.0 - torch.abs(air_time_left - air_time_right)
    
    # 只在有速度命令时奖励
    cmd_norm = env.command_manager.get_command("twist").norm(dim=-1)
    active = cmd_norm > command_threshold
    
    # 裁剪到 [threshold_min, threshold_max] 范围
    reward = torch.clamp(symmetry, threshold_min, threshold_max)
    return torch.where(active, reward, torch.zeros_like(reward))


def upright_reward(
    env: ManagerBasedRlEnv,
    body_name: str = "trunk_base",
    std: float = 0.1,
) -> torch.Tensor:
    """鼓励机器人保持直立（躯干朝上）
    
    Returns:
        高斯奖励：exp(-tilt_angle² / std²)
    """
    # 获取躯干朝向
    body_idx = env.scene["robot"].body_names.index(body_name)
    quat = env.scene["robot"].data.body_quat[:, body_idx]  # (num_envs, 4)
    
    # 计算倾斜角度（z 轴与重力方向的夹角）
    # quat = [w, x, y, z]，提取 z 轴方向
    up_vector = torch.zeros_like(quat)
    up_vector[:, 3] = 1.0  # z 轴
    # 旋转 up_vector 由 quat
    tilted_up = quat_apply(quat, up_vector[:, :3])
    
    # 与重力方向 (0, 0, 1) 的点积 = cos(tilt_angle)
    cos_tilt = tilted_up[:, 2]
    tilt_angle = torch.acos(torch.clamp(cos_tilt, -1.0, 1.0))
    
    # 高斯奖励
    return torch.exp(-(tilt_angle ** 2) / (std ** 2))
```

### 3.2 观测函数示例

```python
def joint_pos_rel_home(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """关节位置相对于 HOME 姿态的偏差
    
    Returns:
        形状 (num_envs, num_joints)
    """
    robot = env.scene[asset_cfg.name]
    current_pos = robot.data.joint_pos[:, env.scene[asset_cfg.name].joint_ids]
    home_pos = robot.data.default_joint_pos[0]  # HOME 姿态
    
    return current_pos - home_pos


def base_orientation_ee_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """躯干朝向 + 末端接触状态（用于摔倒检测）
    
    Returns:
        形状 (num_envs, 3 + num_feet)
    """
    # 投影重力（3D）
    gravity = env.scene["robot"].data.projected_gravity[:, :3]
    
    # 脚底接触状态
    contact_sensor = env.sensors[sensor_name]
    contact = contact_sensor.data.force[:, :, 2] > 1.0  # z 方向力 > 1N
    
    return torch.cat([gravity, contact.float()], dim=-1)
```

### 3.3 事件函数示例

```python
def randomize_com(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    ranges: tuple[float, float],
) -> None:
    """随机化质心位置（域随机化）
    
    Args:
        env_ids: 要重置的环境 ID
        ranges: 随机化范围 (min, max) 单位：米
    """
    robot = env.scene[asset_cfg.name]
    
    # 为每个环境采样不同的质心偏移
    num_envs = len(env_ids)
    offset_x = torch.empty(num_envs).uniform_(*ranges)
    offset_y = torch.empty(num_envs).uniform_(*ranges)
    
    # 应用到躯干质心
    trunk_idx = robot.body_names.index("trunk_base")
    robot.data.body_ipos[env_ids, trunk_idx, 0] += offset_x
    robot.data.body_ipos[env_ids, trunk_idx, 1] += offset_y


def expand_bam_friction_fields(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """扩展 BAM 摩擦场（启动时调用一次）
    
    BAM 执行器每步写入 per-env 的 dof_frictionloss/dof_damping，
    这些字段必须是 per-env 扩展的，否则会别名到单个共享缓冲区。
    """
    env.sim.expand_model_fields(("dof_frictionloss", "dof_damping"))
```

---

## 第四步：定义任务配置

**文件位置**：`src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`

任务配置组装所有组件：观测、奖励、事件、课程等。

```python
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    ObservationTermCfg,
    RewardTermCfg,
    EventTermCfg,
    TerminationTermCfg,
    CurriculumTermCfg,
)
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp

def make_microduck_velocity_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """创建 Microduck 走路环境配置
    
    Args:
        play: 是否为播放模式（减少随机化）
        rough: 是否使用粗糙地形
    """
    # 1. 从 mjlab 标准模板开始
    cfg = make_velocity_env_cfg()
    
    # 2. 设置机器人
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    
    # 3. 配置观测
    cfg.observations["actor"]["joint_pos_rel_home"] = ObservationTermCfg(
        func=microduck_mdp.joint_pos_rel_home,
        params={"asset_cfg": SceneEntityCfg("robot")},
        noise=UniformNoiseCfg(add=0.01, scale=0.01),
    )
    
    # 4. 配置奖励
    cfg.rewards["feet_air_time"] = RewardTermCfg(
        func=microduck_mdp.feet_air_time_reward,
        weight=3.0,  # 正权重 = 鼓励
        params={
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.125,
            "threshold_max": 0.300,
            "command_threshold": 0.01,
        },
    )
    
    cfg.rewards["upright"] = RewardTermCfg(
        func=microduck_mdp.upright_reward,
        weight=2.0,
        params={"body_name": "trunk_base", "std": 0.1},
    )
    
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.action_rate_penalty,
        weight=-0.1,  # 负权重 = 惩罚
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # 5. 配置事件（域随机化）
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",  # 启动时调用一次
    )
    
    cfg.events["randomize_com"] = EventTermCfg(
        func=microduck_mdp.randomize_com,
        mode="reset",  # 每次重置时调用
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "ranges": (-0.003, 0.003),  # ±3mm
        },
    )
    
    # 6. 配置终止条件
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )
    
    # 7. 配置课程学习
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * 24, "weight": -0.2},
                {"step": 1000 * 24, "weight": -0.4},
                {"step": 1500 * 24, "weight": -1.0},
            ],
        },
    )
    
    # 8. 配置地形
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG
    
    return cfg


# RL 训练配置
MicroduckRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        gamma=0.99,
        lam=0.95,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="velocity",
    num_steps_per_env=24,
    max_iterations=50_000,
)
```

---

## 第五步：注册任务

**文件位置**：`src/mjlab_microduck/tasks/__init__.py`

```python
from mjlab.tasks.registry import register_mjlab_task
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)

# 注册平地走路任务
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

# 注册粗糙地形走路任务
register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)
```

---

## 第六步：训练

### 6.1 冒烟测试（必做）

```bash
# 5 次迭代，64 个环境，快速检查配置错误
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num_envs 64 \
    --agent.max_iterations 5
```

**检查项**：
- [ ] 没有 NaN 错误
- [ ] 观测维度正确（61D）
- [ ] 所有奖励项都有输出
- [ ] wandb 日志正常记录

### 6.2 正式训练

```bash
# 4096 个环境，训练 50000 次迭代
uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num_envs 4096
```

**训练过程监控**（wandb）：
- `Episode_Reward/Mean`：总奖励应上升
- `Episode_Length/Mean`：episode 长度应增加
- 每个奖励项的加权值：惩罚项应 ≤ 0
- `Policy/entropy`：熵应缓慢下降（不是崩溃）

### 6.3 训练参数说明

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `num_envs` | 并行环境数 | 4096 |
| `num_steps_per_env` | 每个环境每迭代步数 | 24 |
| `max_iterations` | 最大迭代次数 | 50000 |
| `learning_rate` | 学习率 | 1e-3 |
| `gamma` | 折扣因子 | 0.99 |
| `clip_param` | PPO 裁剪范围 | 0.2 |

---

## 第七步：导出 ONNX

```bash
# 导出策略为 ONNX 格式（烘焙 obs normalizer）
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
    --wandb-run-path <entity>/<project>/<run_id> \
    --output policy.onnx
```

**关键点**：
- ONNX 中烘焙了观测归一化器（均值/标准差）
- 部署时不需要再做归一化
- 输入：61D 观测
- 输出：14D 动作（关节位置）

---

## 第八步：部署到真实机器人

```bash
# 在机器人上加载策略
robotctl policy add policy.onnx --name walking

# 切换到走路策略
robotctl policy use walking
```

---

## 完整训练流程图

```
┌─────────────────────────────────────────────────────────────┐
│  1. 定义机器人模型 (robot_walk.xml)                          │
│     - MJCF 格式，定义关节、质量、碰撞体                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 定义机器人配置 (microduck_constants.py)                  │
│     - EntityCfg：初始姿态、执行器、碰撞配置                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 定义 MDP 函数 (mdp.py)                                  │
│     - 奖励函数：upright, feet_air_time, action_rate          │
│     - 观测函数：joint_pos_rel_home, base_orientation         │
│     - 事件函数：randomize_com, expand_bam_friction_fields    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  4. 定义任务配置 (microduck_velocity_env_cfg.py)             │
│     - 组装观测/奖励/事件/课程                                 │
│     - 配置 RL 算法参数（PPO）                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  5. 注册任务 (__init__.py)                                   │
│     - register_mjlab_task(task_id, env_cfg, rl_cfg)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  6. 训练 (uv run train)                                      │
│     - 冒烟测试：64 envs, 5 iters                              │
│     - 正式训练：4096 envs, 50000 iters                        │
│     - 监控 wandb 日志                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  7. 导出 ONNX (scripts/export.py)                            │
│     - 烘焙 obs normalizer                                    │
│     - 输出 policy.onnx                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  8. 部署到真实机器人 (robotctl policy add)                    │
│     - 加载 ONNX 策略                                         │
│     - 实时推理：观测 → 策略 → 动作                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 常见问题

### Q1: 训练时出现 NaN 怎么办？

**原因**：
- 接触力过大导致数值不稳定
- 关节超出限位
- 域随机化范围过大

**解决**：
- 增加 `nan_state` 终止条件（立即重置）
- 减小域随机化范围
- 软化地形接触（`solref` 参数）
- 增加 `nconmax`（最大接触数）

### Q2: 策略不收敛怎么办？

**检查**：
- 奖励函数符号是否正确（正=鼓励，负=惩罚）
- 奖励权重是否合理（主任务 > 正则化）
- 观测是否包含足够信息
- 课程学习是否太快

**解决**：
- 先简化任务（平地、无随机化）
- 逐步增加难度
- 检查 wandb 中每个奖励项的贡献

### Q3: Sim2Real 失败怎么办？

**可能原因**：
- 域随机化不够
- 执行器模型不准确（BAM 参数）
- 观测延迟不匹配

**解决**：
- 增加域随机化范围
- 用真实舵机做系统辨识（`bam.fit`）
- 增加指令延迟（`delay_min_lag`）

---

## 参考资源

- **mjlab 文档**：https://github.com/mujocolab/mjlab
- **BAM 执行器**：`.venv/lib/python3.12/site-packages/bam/`
- **PPO 算法**：`.venv/lib/python3.12/site-packages/rsl_rl/algorithms/ppo.py`
- **AGENTS.md**：项目约定和常见陷阱

---

**最后更新**：2026-09-24  
**适用版本**：mjlab 1.3.0, rsl_rl 5.0.1
