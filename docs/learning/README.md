# Microduck RL 学习指南

## 简介

Microduck RL 是 Microduck 双足机器人的强化学习训练环境集合。基于 mjlab（MuJoCo Warp）框架，使用 PPO 算法训练各种运动策略，最终导出为 ONNX 部署到真实机器人上。

这个仓库实现了完整的 sim2real 流程：从仿真训练到真实机器人部署。

## 学习路径

建议按以下顺序阅读文档：

### 1. [项目概览](00-项目概览.md)
- 项目定位和 sim2real 流程
- 技术栈（mjlab, MuJoCo Warp, PPO, ONNX）
- 训练环境列表
- 项目结构

### 2. [训练框架](01-训练框架.md)
- mjlab 框架概览
- MuJoCo Warp 并行仿真
- PPO 算法配置
- 训练命令和工作流

### 3. [环境设计](02-环境设计.md)
- Observation 空间（61维）
- Action 空间（14维）
- 任务配置系统
- 关节布局和命名约定

### 4. [执行器模型](03-执行器模型.md)
- BAM 执行器模型（电压控制）
- 摩擦模型（Coulomb/Stribeck/负载相关）
- Backlash 间隙模拟
- 执行器域随机化

### 5. [域随机化](04-域随机化.md)
- 域随机化的目的
- 各种随机化参数
- 非累积性原则
- 课程学习

### 6. [奖励设计](05-奖励设计.md)
- 奖励符号约定
- 常见陷阱和解决方案
- 各类任务的奖励模式
- 正则化策略

### 7. [策略导出部署](06-策略导出部署.md)
- ONNX 导出流程
- 观察归一化器烘焙
- 发布到 Hugging Face Hub
- 机器人端加载

### 8. [训练经验](07-训练经验.md)
- 奖励黑客防范
- 课程节奏控制
- Sim2real 差距
- 调试技巧

## 快速开始

### 环境要求
- CUDA GPU（训练通过 MuJoCo Warp 加速）
- uv 包管理器
- Python 3.10+

### 安装和训练

```bash
# 克隆项目
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# 训练行走策略（需要 GPU，4096 环境约 1-2 小时）
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# 可视化训练好的策略
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# 导出为 ONNX
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
```

### 关键代码位置

- **任务配置**: `src/mjlab_microduck/tasks/microduck_*_env_cfg.py`
- **MDP 函数**: `src/mjlab_microduck/tasks/mdp.py`（奖励、事件、观察）
- **执行器模型**: `src/mjlab_microduck/actuator/friction_dr_bam.py`
- **机器人模型**: `src/mjlab_microduck/robot/microduck/`
- **导出脚本**: `src/mjlab_microduck/export.py`
- **发布工具**: `src/mjlab_microduck/publish/`

## 核心概念

### 1. Sim2real 流程
```
MuJoCo 仿真 → PPO 训练 → ONNX 导出 → 机器人部署
```

### 2. 统一的 Observation 契约
所有策略共享 61 维观察空间：
- 48 维本体感知（陀螺仪、重力、关节状态、历史动作）
- 13 维控制指令（速度、头部姿态、身体姿态）

### 3. BAM 执行器模型
不是理想的 PD 控制器，而是模拟真实 XL330 舵机的电压控制律：
- 反电动势
- 库仑/斯特里贝克摩擦
- 负载相关摩擦
- 电压下降

### 4. 域随机化
通过随机化物理参数训练鲁棒策略：
- CoM 偏移
- 质量/惯量
- 关节摩擦
- IMU 安装误差
- 编码器偏置

## 相关资源

- **microduck**: https://github.com/pollen-robotics/microduck（机器人端运行时）
- **mjlab**: https://github.com/mujocolab/mjlab（训练框架）
- **BAM**: https://github.com/Rhoban/bam（执行器模型）
- **AGENTS.md**: 项目内部的 AI 编码代理指南

## 学习建议

1. **先理解整体流程**：从项目概览开始，了解 sim2real 全貌
2. **阅读 AGENTS.md**：这是项目经验的精华总结
3. **从行走任务开始**：`microduck_velocity_env_cfg.py` 是最完整的参考
4. **动手训练**：先跑通一个 smoke test，再尝试完整训练
5. **关注 sim2real 差距**：理解为什么需要 BAM 和域随机化

## 常见问题

### Q: 为什么不用理想的 PD 控制器？
A: 真实舵机是电压控制的，有摩擦、反电动势、电压下降等非线性。理想 PD 在仿真中表现完美，但部署到真实机器人会失败。

### Q: 为什么 observation 是 61 维？
A: 这是所有策略共享的统一接口，使得运行时可以热切换策略。未使用的维度用零填充，而不是删除。

### Q: 训练一个策略需要多久？
A: 简单的 episodic 任务约 1000 次迭代（4096 环境），复杂的步态和恢复任务需要 4000-6000 次迭代。

### Q: 没有 GPU 怎么办？
A: 可以使用 `--hf-jobs` 参数提交到 Hugging Face Jobs 训练。

## 下一步

开始阅读 [00-项目概览.md](00-项目概览.md)，了解 Microduck RL 的全貌。
