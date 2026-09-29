---
title: 'Week 1：机器人学习基础与 Diffusion Policy'
description: '从机器人学习基础出发，梳理 Diffusion Policy 在 Push-T 图像任务中的数据流、训练流程与动作执行机制。'
publishDate: '2026-09-29'
tags:
  - diffusion-policy
  - robot-learning
  - embodied-ai
heroImage:
  src: './DiffusionPolicymodel.png'
  alt: 'Diffusion Policy 模型示意图'
  color: '#4891B2'
language: 'Chinese'
draft: false
comment: true
---

> **本周目标**
>
> 建立一条完整的数据流认知，并理解 Diffusion Policy 为什么对动作序列做扩散与去噪。
>
> **阅读范围**
>
> 本文中的代码位置和张量形状默认针对 **Push-T Image + Diffusion U-Net** 配置。其他任务、配置或真机数据的 Action 维度可能不同。

---

## 一页总结

### 机器人策略的基本闭环

```text
Environment / Robot
        ↓
Observation = Image + Robot State
        ↓
Policy
        ↓
Action / Action Sequence
        ↓
env.step(action)
        ↓
New Observation, Reward, Done
```

### 本项目 Push-T 图像策略的数据流

```text
最近 2 帧观测
image:     [B, 2, 3, 96, 96]
agent_pos: [B, 2, 2]
        │
        ├─ image → Crop / Normalize → ResNet-18 → visual feature
        └─ agent_pos ───────────────────────────→ low-dim feature
                              │
                              └─ concatenate → observation condition
                                                   │
高斯噪声 Action Sequence [B, 16, 2] ──────────────┤
                                                   ↓
                                    Conditional 1D U-Net
                                    反复预测并去除噪声
                                                   ↓
action_pred: [B, 16, 2]  完整预测轨迹
                                                   ↓
从索引 To-1 开始截取 8 步
                                                   ↓
action:      [B, 8, 2]   本轮要执行的动作块
                                                   ↓
MultiStepWrapper 逐步拆开
                                                   ↓
PushTEnv.step(single_action: [2])
```

最重要的一句话：

> Diffusion Policy 不是给图像加噪，而是以图像和机器人状态作为条件，对 Action Sequence 加噪并学习去噪。

---

## 1. 机器人学习基础概念

### 1.1 Joint、DoF 与 End-Effector

- **Joint（关节）**：机器人能够转动或移动的连接部件。
- **DoF（自由度）**：描述机器人独立运动能力的数量。例如 6 自由度机械臂可以用 `q = [q1, ..., q6]` 表示关节状态。
- **End-Effector（末端执行器）**：真正与环境交互的末端部分，例如夹爪、吸盘、灵巧手或工具。

### 1.2 Joint Space 与 Cartesian Space

| 空间            | 描述对象           | 典型表示                | 直觉                             |
| --------------- | ------------------ | ----------------------- | -------------------------------- |
| Joint Space     | 每个关节的状态     | `q1, q2, ...`           | 机器人各关节现在是什么状态       |
| Cartesian Space | 末端在空间中的位姿 | `x, y, z + orientation` | 末端执行器现在位于哪里、朝向哪里 |

对应的两个基本问题：

```text
Joint State ──FK（正运动学）──> End-Effector Pose
Target Pose ──IK（逆运动学）──> Joint State
```

### 1.3 State、Observation 与 Action

**Robot State** 是机器人自身的状态，常见内容包括：

- joint position / velocity；
- gripper state；
- end-effector pose；
- 力、力矩或触觉信息。

**Observation** 是 Policy 实际能看到的信息，不一定等于环境的完整状态：

```text
Observation = Camera RGB + Robot State (+ 历史信息)
```

**Action** 是 Policy 发给环境或控制器的命令，可能是：

- 关节目标位置或速度；
- 末端绝对目标位姿；
- 末端位姿增量；
- 夹爪开合命令。

阅读任何机器人学习项目时，首先要确认：

1. Observation 里究竟有什么；
2. Action 表示什么物理量；
3. Action 是 absolute 还是 delta；
4. Policy 一次输出一步还是一段动作。

### 1.4 Absolute Action 与 Delta Action

- **Absolute Action**：直接预测目标状态，例如目标位置 `(x, y)`。
- **Delta Action**：预测相对当前状态的变化量，例如 `(Δx, Δy)`。

当前 Push-T 配置的 Action 是二维**绝对位置目标**：圆形 agent 下一步要去的 `(x, y)`。它不是速度，也不是 `(Δx, Δy)`。

---

## 2. Diffusion Policy 在解决什么问题

普通行为克隆可以直接学习：

```text
observation → action
```

但机器人示范数据有几个明显困难：

1. **多模态**：同一个状态可能有多种正确动作，例如从 T 形块左边或右边绕过去；
2. **时序相关**：连续动作必须保持一致，不能每一步在不同策略之间来回切换；
3. **高精度**：操作任务中很小的动作偏差也可能导致失败；
4. **长时规划与及时反应的冲突**：预测得太短容易短视，执行得太长又无法及时应对环境变化。

Diffusion Policy 把策略写成条件分布：

```text
p(Action Sequence | Observation History)
```

它从随机高斯噪声动作开始，在视觉与机器人状态的条件下反复去噪，最终生成一段有时序一致性的动作。

论文报告在 4 个机器人操作 benchmark、15 个任务上的平均提升为 **46.9%**。论文强调的三个关键设计是：

- receding-horizon action prediction；
- visual conditioning；
- time-series diffusion network（CNN 或 Transformer）。

---

## 3. 扩散模型基础：在 Action 上加噪和去噪

### 3.1 训练阶段

训练数据中有干净动作序列 $A^0$。训练时：

1. 随机采样一个扩散时间步 $k$；
2. 采样高斯噪声 $\epsilon$；
3. 按 noise schedule 给干净动作序列加噪，得到 $A^k$；
4. 模型接收 `观测条件 + 带噪动作 + 时间步 k`；
5. 模型预测噪声 $\hat{\epsilon}_\theta$；
6. 用 MSE 比较真实噪声和预测噪声。

核心目标可以记成：

$$
\mathcal{L}=\operatorname{MSE}\left(\epsilon,\epsilon_\theta(O,A^k,k)\right)
$$

当前代码的对应过程：

```text
clean action
    ↓ add_noise
noisy action
    ↓ ConditionalUnet1D(noisy_action, timestep, obs_condition)
predicted noise
    ↓ MSE(predicted noise, sampled noise)
loss.backward()
```

### 3.2 推理阶段

推理时没有真实动作可用：

1. 初始化一个 `[B, 16, 2]` 的高斯噪声轨迹；
2. 模型根据 Image、Robot State 和扩散时间步预测噪声；
3. scheduler 从 $A^k$ 更新到 $A^{k-1}$；
4. 重复多次，得到 $A^0$，即完整动作序列。

因此训练和推理的区别是：

| 阶段 | 起点                       | 模型学习/执行的任务            |
| ---- | -------------------------- | ------------------------------ |
| 训练 | 数据集中的真值 Action 加噪 | 预测被加入的噪声               |
| 推理 | 纯高斯噪声 Action          | 反复去噪，生成 Action Sequence |

---

## 4. 三个 Horizon：$T_o$、$T_p$、$T_a$

论文中的三个 horizon 是理解 Diffusion Policy 的关键。

| 名称                | 论文符号 | Push-T 配置 | 含义                                   |
| ------------------- | -------- | ----------: | -------------------------------------- |
| Observation Horizon | $T_o$    |           2 | 每次决策看最近多少帧观测               |
| Prediction Horizon  | $T_p$    |          16 | 每次生成多少步完整动作轨迹             |
| Action Horizon      | $T_a$    |           8 | 本轮实际执行多少步后重新观察、重新规划 |

执行方式是 receding horizon control：

```text
观察最近 2 帧
    ↓
预测未来 16 步
    ↓
只执行其中 8 步
    ↓
重新观察环境
    ↓
再次预测 16 步
```

为什么预测一段动作：

- 保持 temporal consistency，避免动作左右抖动；
- 能表达动作之间的时序关系，减少短视规划；
- 对示范中的 idle action 更稳健；
- 可以部分吸收图像处理、网络传输和推理带来的延迟。

为什么不把 16 步全部执行完：

- 环境可能在执行过程中发生变化；
- 执行块太长会降低闭环响应速度；
- $T_a$ 是动作一致性和环境响应性的折中。

论文消融实验中，8 步 action horizon 对多数测试任务是较好的折中，但它不是对所有任务都固定最优。

---

## 5. Push-T 项目中的完整数据流

### 5.1 数据配置定义了什么

`diffusion_policy/config/task/pusht_image.yaml` 定义：

```yaml
obs:
  image:
    shape: [3, 96, 96]
    type: rgb
  agent_pos:
    shape: [2]
    type: low_dim
action:
  shape: [2]
```

这表示单个时间步中：

- Image 是一张 `3 × 96 × 96` 的 RGB 图像；
- Robot State 只使用二维 `agent_pos = (x, y)`；
- Action 是二维目标位置 `(x, y)`。

注意：Push-T 环境内部其实还知道 T 形块的位置和角度，但 image policy 没有把完整块状态作为 low-dimensional state 输入；这些信息主要由图像提供。

### 5.2 Dataset 输出

`diffusion_policy/dataset/pusht_image_dataset.py` 从 replay buffer 读取：

```text
img, state, action
```

然后整理成一个长度为 16 的训练样本：

```python
{
    "obs": {
        "image": image,          # [T, 3, 96, 96]
        "agent_pos": agent_pos,  # [T, 2]
    },
    "action": action             # [T, 2]
}
```

其中：

- `image` 从 HWC 转成 CHW，并除以 255 映射到 `[0, 1]`；
- `agent_pos = state[:, :2]`，只取 agent 的二维位置；
- `T = horizon = 16`。

DataLoader 组成 batch 后：

```text
image:     [B, 16, 3, 96, 96]
agent_pos: [B, 16, 2]
action:    [B, 16, 2]
```

Policy 真正作为条件使用的是前 `n_obs_steps=2` 帧观测，而监督目标仍是完整 16 步 Action Sequence。

### 5.3 Image 在哪里进入模型

入口链路：

```text
batch['obs']['image']
    ↓
DiffusionUnetImagePolicy.compute_loss()
    ↓ 取前 2 帧并把 B、To 合并
[B*2, 3, 96, 96]
    ↓
MultiImageObsEncoder.forward()
    ↓ Crop + ImageNet Normalize + ResNet-18
visual feature
```

关键代码：

- `diffusion_policy/policy/diffusion_unet_image_policy.py:192-218`
- `diffusion_policy/model/vision/multi_image_obs_encoder.py:127-179`

### 5.4 Robot State 在哪里进入模型

在当前 Push-T image 配置中，Robot State 是 `agent_pos`：

```text
batch['obs']['agent_pos']
    ↓ normalize
[B*2, 2]
    ↓
MultiImageObsEncoder.forward()
    ↓ 不经过 ResNet，直接作为 low-dim feature
    ↓
与 visual feature 拼接
```

对应代码：

- Dataset 取值：`diffusion_policy/dataset/pusht_image_dataset.py:62-72`
- Encoder 拼接：`diffusion_policy/model/vision/multi_image_obs_encoder.py:167-179`

因此，`MultiImageObsEncoder` 这个名字容易让人误会：它不只编码图像，也会把 low-dimensional observation 拼接到视觉特征中。

### 5.5 Observation 如何条件化扩散网络

当前配置 `obs_as_global_cond: True`。前 2 帧的图像特征与 `agent_pos` 被编码后，reshape 成每个 batch 的一条 `global_cond`：

```text
Image feature + agent_pos
        ↓ concatenate
observation feature for each frame
        ↓ flatten To dimension
global_cond
        ↓
ConditionalUnet1D
```

扩散 U-Net 的主要输入仍然是带噪的 Action Sequence，`global_cond` 用来告诉去噪网络当前场景和机器人状态。

---

## 6. Action 的张量形状与 Policy 输出

### 6.1 Shape 总表

| 阶段                        | Shape        | 含义                      |
| --------------------------- | ------------ | ------------------------- |
| 单步 Action                 | `[2]`        | 一个绝对目标位置 `(x, y)` |
| Dataset 单样本              | `[16, 2]`    | 一个长度为 16 的动作窗口  |
| 训练 Batch                  | `[B, 16, 2]` | Batch 内的真值动作序列    |
| 带噪 Action                 | `[B, 16, 2]` | 扩散训练或采样中的轨迹    |
| 完整预测 `action_pred`      | `[B, 16, 2]` | 去噪生成的完整动作轨迹    |
| 本轮执行 `action`           | `[B, 8, 2]`  | 从完整轨迹截出的动作块    |
| 单个环境收到的 Action Chunk | `[8, 2]`     | MultiStepWrapper 的输入   |
| `PushTEnv.step()` 单次输入  | `[2]`        | 真正执行的一步目标位置    |

### 6.2 Policy 输出的数据是什么

`DiffusionUnetImagePolicy.predict_action()` 返回一个字典：

```python
{
    "action": action,           # [B, 8, 2]，交给环境执行
    "action_pred": action_pred  # [B, 16, 2]，完整预测轨迹
}
```

代码不是简单取 `action_pred[:, :8]`，而是：

```python
start = To - 1
end = start + n_action_steps
action = action_pred[:, start:end]
```

当前 $T_o=2$，所以实际截取的是完整预测序列的索引 `[1:9]`。

---

## 7. `env.step(action)` 在哪里

### 7.1 Runner 层：一次交给环境一个 Action Chunk

`diffusion_policy/env_runner/pusht_image_runner.py:197-209`

```text
policy.predict_action(obs_dict)
    ↓
action_dict['action']       # [B, 8, 2]
    ↓
env.step(action)
```

这里的 `env` 是向量化环境，所以第一维是并行环境数量。

### 7.2 MultiStepWrapper：把动作块拆成单步

`diffusion_policy/gym_util/multistep_wrapper.py:101-124`

```python
for act in action:          # action: [8, 2]
    observation, reward, done, info = super().step(act)
```

### 7.3 底层 Push-T 环境：真正执行一步

`diffusion_policy/env/pusht/pusht_env.py:109-138`

```python
def step(self, action):     # action: [2]
    ...
    return observation, reward, done, info
```

底层使用 PD 控制把圆形 agent 推向目标 `(x, y)`，物理仿真以 100 Hz 更新，而策略控制频率是 10 Hz。

---

## 8. 一个 Episode 在哪里开始和结束

### 8.1 Episode 开始

Runner 中：

```python
obs = env.reset()
policy.reset()
```

位置：`diffusion_policy/env_runner/pusht_image_runner.py:176-180`。

底层 `PushTEnv.reset()` 会：

1. 重建仿真空间；
2. 根据 seed 采样 agent 与 T 形块的初始状态；
3. 设置状态；
4. 返回第一帧 observation。

位置：`diffusion_policy/env/pusht/pusht_env.py:87-107`。

### 8.2 Episode 循环

```text
while not done:
    obs → policy.predict_action()
        → 8-step action chunk
        → env.step(action chunk)
        → new obs, reward, done, info
```

### 8.3 Episode 结束

有两种主要结束条件：

1. **任务成功**：T 形块与目标区域的覆盖率超过 `success_threshold = 0.95`；
2. **时间截断**：累计执行达到 `max_steps = 300`。

`MultiStepWrapper` 在动作块内部一旦发现 `done=True` 就停止执行剩余动作。Runner 对所有并行环境的 `done` 做汇总，全部结束后退出 rollout 循环。

---

## 9. 从训练脚本开始的完整训练流程

### 9.1 入口

```text
train.py
  ↓ Hydra 读取配置
train_diffusion_unet_image_workspace.yaml
  ↓ _target_
TrainDiffusionUnetImageWorkspace
  ↓
workspace.run()
```

`train.py` 本身只负责：加载配置、找到 Workspace 类、实例化并调用 `run()`。

### 9.2 初始化

Workspace 初始化时创建：

- `DiffusionUnetImagePolicy`；
- optimizer；
- 可选 EMA model；
- global step 和 epoch 状态。

### 9.3 Dataset 与 DataLoader

`run()` 中：

```text
Hydra instantiate dataset
    ↓
PushTImageDataset
    ↓
DataLoader
    ↓
normalizer = dataset.get_normalizer()
    ↓
policy.set_normalizer(normalizer)
```

### 9.4 每个 Batch 的训练

```text
batch 移到 GPU
    ↓
policy.compute_loss(batch)
    ├─ normalize image / agent_pos / action
    ├─ 取前 2 帧 observation
    ├─ 编码 image 与 agent_pos，得到 global condition
    ├─ 给完整 action sequence 加随机噪声
    ├─ ConditionalUnet1D 预测噪声
    └─ MSE loss
    ↓
loss.backward()
    ↓
optimizer.step()
    ↓
lr_scheduler.step()
    ↓
EMA update
```

### 9.5 每个 Epoch 后

按配置决定是否执行：

- rollout：在 Push-T 环境里真正闭环执行；
- validation：计算验证集 diffusion loss；
- train sample：生成动作并与真值动作计算 MSE；
- checkpoint：保存 last checkpoint 和 top-k checkpoint；
- logging：写入 WandB 和本地 JSON log。

注意：训练 loss 衡量的是噪声预测误差，不等价于任务成功率。真正评价策略需要 rollout。

---

## 10. 网络结构与实现选择

### 10.1 CNN-based Diffusion Policy

当前项目主要学习的配置使用：

- ResNet-18：把图像编码成视觉特征；
- GroupNorm：替换 BatchNorm，适合扩散模型与 EMA；
- Spatial Softmax：论文中用于保留空间位置信息；
- Conditional 1D U-Net：沿动作时间维做卷积去噪；
- FiLM / global conditioning：把 observation feature 注入去噪网络。

论文建议新任务优先尝试 CNN 版本，因为它通常更容易训练、需要的调参更少。

### 10.2 Transformer-based Diffusion Policy

Transformer 版本通过 cross-attention 使用 observation embedding，并对 Action token 使用 causal attention。它可能更适合快速变化、高频动作或速度控制，但训练对超参数更敏感。

---

## 11. Normalization 为什么重要

论文特别强调 Action normalization：

- 每个 Action 维度通常分别按 min/max 缩放到 `[-1, 1]`；
- DDPM 每一步会把预测裁剪在 `[-1, 1]`，因此普通的零均值、单位方差标准化可能让部分动作范围不可达；
- 接近常数的维度只做零均值平移，避免数值问题；
- rotation representation 需要单独处理。

当前 Push-T 代码还会：

- 把图像转为 `[0, 1]`；
- 再使用 ImageNet mean/std 归一化图像；
- 对 `agent_pos` 和 Action 使用 Dataset 统计量建立 `LinearNormalizer`。

推理结束后，Action 必须 unnormalize，才能回到环境坐标系。

---

## 12. 论文主要结论、优点与限制

### 12.1 主要优点

1. **多模态动作分布**：能学习多种可行动作，并在一次 rollout 中稳定选择其中一种；
2. **高维输出**：能够直接生成整段 Action Sequence；
3. **时序一致性**：动作不容易在不同模式之间跳变；
4. **训练稳定**：噪声预测目标避免了隐式策略中困难的归一化常数估计；
5. **闭环重规划**：执行一部分动作后重新观察，能应对扰动；
6. **适合 position control**：论文实验中与 receding-horizon action sequence 有较好配合。

### 12.2 实验层面的认识

- 论文覆盖仿真与真机、单臂与双臂、刚体与流体任务；
- Image Policy 通常更适合较短但大于 1 的 observation horizon，论文认为 2 是多数任务的好折中；
- Push-T 的常用配置是 $T_o=2$、$T_a=8$、$T_p=16$；
- 真机实验用 DDIM 减少推理去噪步数，以满足实时控制；
- 视觉编码器端到端训练或低学习率微调通常优于直接冻结预训练视觉编码器。

### 12.3 限制

- 仍然继承 Behavior Cloning 的问题：示范不足或分布外状态会导致性能下降；
- 多轮去噪带来比普通回归策略更高的推理开销和延迟；
- 对需要极高控制频率的任务仍有挑战；
- Action Horizon 太长会降低对环境变化的响应速度；
- 模型效果仍依赖数据质量、动作表示、控制器与机器人硬件能力。

---

## 13. 建议按什么顺序阅读代码

针对 Push-T image + Diffusion U-Net，建议按这条数据流阅读，而不是按目录顺序阅读：

1. `diffusion_policy/config/train_diffusion_unet_image_workspace.yaml`  
   先看 horizon、模型、scheduler、batch size 和训练参数。
2. `diffusion_policy/config/task/pusht_image.yaml`  
   确认 Observation/Action 的物理含义与单步张量形状。
3. `train.py`  
   看 Hydra 如何找到 Workspace。
4. `diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`  
   看 Dataset、DataLoader、训练循环、验证、rollout 和 checkpoint。
5. `diffusion_policy/dataset/pusht_image_dataset.py`  
   看原始数据如何变成 `image + agent_pos + action`。
6. `diffusion_policy/policy/diffusion_unet_image_policy.py`  
   重点看 `compute_loss()`、`predict_action()` 和 `conditional_sample()`。
7. `diffusion_policy/model/vision/multi_image_obs_encoder.py`  
   找到 Image 和 Robot State 真正进入模型的位置。
8. `diffusion_policy/model/diffusion/conditional_unet1d.py`  
   看带噪 Action 如何在 observation condition 下被去噪。
9. `diffusion_policy/env_runner/pusht_image_runner.py`  
   看 Policy 输出如何进入闭环 rollout。
10. `diffusion_policy/gym_util/multistep_wrapper.py`  
    看 `[8, 2]` 的 Action Chunk 如何拆成 8 个 `[2]`。
11. `diffusion_policy/env/pusht/pusht_image_env.py` 与 `pusht_env.py`  
    看 observation 的生成、单步动作执行、reward、done、reset。

---

## 14. 核心问题速查

### Image 在哪里进入模型？

从 `batch['obs']['image']` 进入 `DiffusionUnetImagePolicy.compute_loss()` 或 `predict_action()`，再进入 `MultiImageObsEncoder.forward()`，经过图像预处理和 ResNet-18，最后成为扩散 U-Net 的条件。

### Robot State 在哪里进入模型？

当前任务的 Robot State 是 `agent_pos`。它从 `batch['obs']['agent_pos']` 进入 `MultiImageObsEncoder.forward()`，不经过 ResNet，直接与视觉特征拼接。

### Action 的张量形状是多少？

单步 `[2]`，完整训练/预测序列 `[B, 16, 2]`，每轮交给环境执行的是 `[B, 8, 2]`，底层 `PushTEnv.step()` 每次最终收到 `[2]`。

### Policy 输出的数据是什么？

一个字典：`action` 是本轮执行的 8 步动作块，`action_pred` 是完整 16 步预测轨迹。

### `env.step(action)` 在哪里？

Runner 在 `pusht_image_runner.py` 调用向量环境的 `env.step(action)`；`MultiStepWrapper` 拆分动作块；最终落到 `pusht_env.py` 的单步 `step()`。

### 一个 Episode 在哪里开始和结束？

在 Runner 中由 `env.reset()` 开始；成功覆盖率超过 95% 或达到 300 步上限时结束。

---

## 15. 后续学习问题

- `ConditionalUnet1D` 内部每一层如何使用 diffusion timestep embedding 和 global condition？
- 为什么当前代码截取 `action_pred[:, To-1:To-1+Ta]`？预测窗口的时间对齐方式是什么？
- Dataset 的 `pad_before` 和 `pad_after` 如何处理 Episode 边界？
- Action normalization 的 min/max 来自哪些 Episode，是否会发生数据泄漏？
- EMA model 为什么常常比最后一步训练模型更适合评估？
- rollout success rate 与 validation diffusion loss 为什么可能不一致？
- 从仿真 Push-T 迁移到真机时，Action Space、控制频率、安全约束需要怎样修改？

## 16. 学习检查清单

- [ ] 能区分 State、Observation 和 Action；
- [ ] 能区分 Joint Space 与 Cartesian Space；
- [ ] 能区分 Absolute Action 与 Delta Action；
- [ ] 知道 Diffusion Policy 扩散的是 Action，不是 Image；
- [ ] 能解释 $T_o=2$、$T_p=16$、$T_a=8$；
- [ ] 能说出 Image 和 `agent_pos` 在哪里进入模型；
- [ ] 能说出 `action` 与 `action_pred` 的区别；
- [ ] 能从 Runner 找到真正的 `PushTEnv.step([2])`；
- [ ] 能描述一个 Episode 的开始、循环与结束；
- [ ] 能按照 Dataset → Policy → Loss → Rollout 的顺序继续阅读项目。

---

## 参考资料

- 《Diffusion Policy: Visuomotor Policy Learning via Action Diffusion》
- 《具身 VLA 方向 9 月学习》中的 Week 1 学习目标与问题清单
- 当前项目中 Push-T image + Diffusion U-Net 的配置、Dataset、Policy、Runner 与 Environment 实现
