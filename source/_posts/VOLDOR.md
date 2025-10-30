---
title: VOLDOR
date: 2025-10-24 15:10:02
tags:
---
本文章为对于 **VOLDOR** 算法的梳理与总结，该SLAM定位算法共产出两篇论文产出，包括 《VOLDOR: Visual Odometry from Log-logistic Dense Optical flow Residuals》 和 《VOLDOR+SLAM:For the times when feature-based or direct methods are not good enough》

## 简介与说明

**VOLDOR** 算法 本质上是一个依赖于光流和先验概率的SLAM定位算法。

对于常规的视觉定位算法，在实践都已经取得一定的适用性，但是这些方法本身仍然相当受限于图像之间的视角差大小、图像几何特征以及运动模糊和遮挡等问题。同时，使用的最优化方法一般为最小二乘法或高斯分布假设，这个实际上依赖于前后帧之间的小位移（否则过大的初始差值，可能直接导致优化失败）；尽管也有依赖光流的定位方法，但是仍然服从几何一致性原则，在最优化过程的目标是保证多视图之间的几何一致对应。

但是 **VOLDOR** 算法则是完全以来光流和先验概率知识来进行定位估计，抛弃了几何一致性原则，保证在由于高速运动导致运动畸变的情况下仍然能正常运行。该框架中通过广义EM算法融合稠密光流序列，同步估计了相机运动、像素深度及运动轨迹置信度。

尽管在本论文出现时，已经出现了诸多的基于深度学习的视觉里程计框架，能同步评估深度、光流和相机运动等信息，但是由于其本身解释性较弱（在本人看来本质上是缺乏 Data Driving）导致效果仍然劣于最优方法。

## 问题说明

光流一般用于描述帧之间像素的运动关系，实际上也可以间接视为刚性流与描述物体运动的非受限流的组合。在该视觉里程计方案中，通过输入一系列光流，推断出具有时间一致性的场景结构（深度图）、相机运动以及各光流估计点对应的 `刚性概率`。论文中基于先验的自适应对数逻辑分布残差模型，在估计的刚性流与输入光流之间的端点误差（EPE）上构建了监督框架与系统框架。

**几何符号约定**： 输入一组外部计算的（观测）密集光流场序列 $X = \{X_t | t = 1,...,t_N\}$ ，其中 $X_t$ 表示从图像 $I_{t-1}$ 到 $I_t$ 的光流场，而 $X_t^j = (u_t^j, v_t^j)^T$ 表示时刻 $t$ 像素点 $j$ 的光流向量。目标是推断相机位姿 $T = \{T_t \mid t = 1, \cdots, t_N\}$，其中  $T_t \in \text{SE}(3)$  表示从  $t-1$  时刻到  $t$ 时刻的相对运动。
为建立观测光流 $\mathcal{X}$ 与位姿 $\mathcal{T}$ 的关联模型，我们引入两类隐变量:

  1. 定义于初始帧 $I_0$ 的深度场 $\theta$（ $\theta_j$ 表示像素 $j$ 的深度值）；
  2. 时序刚性概率图集 $W={W_t|t=1,...,t_N}$（ $W_t^j$ 表示 $t$ 时刻像素 $j$ 的刚性概率）。

基于深度图 $θ$ 和刚性图 $W$ ，通过对 $θ$ 关联的点云施加刚体变换 $T$（以 $W$ 为条件），可获得刚性光流 $\xi_t(\theta_j)$ 。设初始位姿 $T_0$ 为单位矩阵，定义 $\pi_t(\theta_j)$ 为使用给定的相机位姿 $\mathbf{T}$ 将深度 $\theta_j$ 关联的三维点投影到时刻 $t$ 相机图像平面的像素坐标：
$$
\pi_t(\theta_j) = \mathbf{K}\left(\prod_{i=0}^t \mathbf{T}_i\right)\theta_j \mathbf{K}^{-1}[x_j \ y_j \ 1]^\top
$$

其中，$\mathbf{K}$为相机内参矩阵，$x_j$、$y_j$为像素$j$的图像坐标。因此，刚性光流可定义为：
$$
\xi_t(\theta_j) = \pi_t(\theta_j) - \pi_{t-1}(\theta_j)
$$

**混合似然模型**： 论文中基于连续刚性概率$W_{j,t}$，建立了观测光流与刚性光流残差的联合概率模型 $W_t^j$。

$$
P(X_{t}^{\pi_{t-1}(\theta_j)} | \theta_j, T_t, W^j_t; T_1 \cdots T_{t-1})=
\begin{cases}
\rho\left(\xi_t(\theta_j) \parallel X_{t}^{\pi_{t-1}(\theta_j)}\right) & \text{若 } W^j_t = 1 \\
\mu\left(X_{t}^{\pi_{t-1}(\theta_j)}\right) & \text{若 } W^j_t = 0
\end{cases}
$$

其中概率密度函数 $ρ(·||·)$ 表示在观测光流 $X^{t}_{\pi_{t-1}(\theta_j)}$ 条件下产生刚性光流 $\xi_t(\theta_j)$ 的概率，$μ(·)$ 则是随 $X^{t}_{\pi_{t-1}(\theta_j)}$ 变化的均匀分布函数。在建模 $X_t$ 的概率时，虽然投 影过程依赖于先前的相机位姿 $T_1,\cdots,T_{t-1}$，但我们在条件概率中仅显式标注当前帧位姿 $T_t$ ——这是因为这些历史位姿被视为固定常量，且$X_t$本身并不包含与它们相关的信息。此外，若将所有历史位姿与 $X_t$ 进行联合建模，不仅会导致参数估计出现偏差，还会显著增加计算复杂度。在下文中，我们将式简记为 $P(X^{t}_{\pi_{t-1}(\theta_j)} | \theta_j, T_t, W^j_t)$。

此时，该视觉里程计问题可以建模为最大似然估计问题：

$$
\arg\max_{\theta,T,W} P(X | \theta, T, W) = \arg\max_{\theta,T,W} \prod_{t} \prod_{j} P(X_{t}^{\pi_{t-1}(\theta_j)} | \theta_j, T_t, W^j_t)
$$
但是此外还需要进一步促进确保密集隐藏变量 $\theta$ 与 $W$ 之间的空间一致性。

## Fisk 残差模型

经过测试，Fisk 残差模型更能解决实际过程中光流残差最优化的问题，实际上，这和该模型的极端值捕捉有关。
在该SLAM方法中，残差被定义为两个光流向量之间端点误差。
给定 $v_{ob} = X ^{\pi_{t-1}(\theta_j)}_t$ , 将 $v_{rig}=\xi_t(\theta_j)$ 用于来匹配真实值概率，建模为：
$$
\rho(v_{rig} \| v_{ob}) = F\left( \| v_{rig} - v_{ob} \|_2^2; A(v_{ob}), B(v_{ob}) \right)
$$
其中，$Fisk$ 分布的概率密度函数（PDF）定义为：
$$
F(x; \alpha, \beta) = \frac{(\beta/\alpha)(x/\alpha)^{\beta-1}}{\left(1 + (x/\alpha)^\beta\right)^2}
$$
Fisk 分布的参数通过以下拟合函数确定：
$$
A(v_{ob}) = a_1 e^{a_2 \| v_{ob} \|_2}
$$
$$
B(v_{ob}) = b_1 \| v_{ob} \|_2 + b_2
$$
其中，$a_1, a_2$ 和 $b_1, b_2$ 为依赖于光流估计方法的可学习参数。

额外的，对于离群值似然函数 $ \mu(\cdot) $，为了利用观测光流提供的先验信息，论文进一步将均匀分布的密度设为观测光流向量的函数$ \mu(\cdot) $。

$$
\mu(v_{ob}) = F(λ²||vob||₂²; A(vob), B(vob))
$$

其中 $λ$ 是调节密度的超参数，同时也是选择内点的严格度标准。λ的数值解释是当离群值与内点不可区分( $W^j_t = 0.5$ )时的光流端点误差百分比(EPE)。因此，不同大小的光流可以在公平的度量标准下进行比较，并被选择为内点。

## 整体迭代推理框架

### 深度和刚性更新

**广义期待最大化（GEM）**。本文在假设相机位姿 $T$ 已知且固定的前提下，随着时间序列推断出了深度 $\theta$ 以及刚性概率 $W$。我们通过期待最大化近似的估计真实后验概率 $P(X | θ, W; T)$。此处将固定参数 $T$ 隐去，采用受限分布族 $q(\theta, W)$，此处 $q(\theta, W)= \prod_{j} q(\theta_{j}) \prod_{t} q(W_{t})$ 。为了进一步简化计算， $q(\theta^j)$ 被限制为克罗内克δ函数族 $q(\theta_j) = \delta(\theta_j = \theta_j^*)$，其中 $\theta_j^*$ 是待估计的参数。此外，$q(W_t)$ 继承了刚性映射 $W_t$ 所定义的平滑性。在这一步中，基于$W_t^j$ 的估计概率密度函数来求解 $\theta_j$ 的最优值。接下来将说明为此任务选择的估计器。

**最大似然估计（MLE）** 该问题的标准 MLE 定义为：
$$
\theta_j^{\text{MLE}} = \arg \max_{\theta_j^*} \sum_t q(W_t^j) \log P\left(X_t^{\pi_t^{-1}(\theta_j)} \mid \theta_j = \theta_j^*, W_t^j\right) \quad (9)
$$
其中，$q(W_t^j)$ 是 E 步给出的估计分布密度。
然而，我们通过实验发现，MLE 准则对初始化的不准确性过于敏感。具体而言，我们仅使用第一帧光流初始化深度图，并利用其深度值逐帧引导后续相机位姿。因此，若初始值存在噪声或误差，使用 MLE 进行优化会对刚性概率 $W$ 施加高选择压力，导致算法倾向于保留少量高精度初始值，而舍弃其余观测。
由于我们的图像批次分析是序列化进行的，这种选择性会显著减少可用于估计后续相机位姿的有效观测数据，从而影响整体优化效果。

**最大内点估计（MIE）** 为降低初始化和序列更新带来的偏差，我们将 MLE 准则松弛为以下 MIE 准则：

$
\theta_j^{\text{MIE}} = \arg \max_{\theta_j^*} \frac{\sum_t q(W_t^j = 1) \log P\left(X_t^{\pi_t^{-1}(\theta_j)} \mid \theta_j = \theta_j^*, W_t^j = 1\right)}{\sum_{W_t^j} P\left(X_t^{\pi_t^{-1}(\theta_j)} \mid \theta_j = \theta_j^*, W_t^j\right)} \quad (10)
$
该准则通过最大化刚性（内点选择）映射 $W$ 来确定最优深度。

**基于采样-传播的优化** 本文通过通过采样-传播策略优化 $\theta_j^{\text{MIE}}$：
随机采样深度 $\theta_j^{\text{smp}}$ 与前一深度 $\theta_j^{\text{prev}}$ 及邻域传播值 $\theta_{j-1}^{\text{nbr}}$ 进行比较。
从三者中选择最优估计更新 $\theta_j^{\text{MIE}}$。
更新后的 $\theta_j^{\text{MIE}}$ 将传播至相邻像素 $j+1$。

**刚性映射的更新** 本文采用行列分割策略，将 2D 图像分解为多条 1D 隐马尔可夫链，并在刚性映射上施加成对平滑项：
$$
P(W_t^j \mid W_t^{j-1}) = \begin{pmatrix}
\gamma & 1 - \gamma \\
1 - \gamma & \gamma
\end{pmatrix} \quad (11)
$$
其中，$\gamma$ 为转移概率，用于约束相邻刚性值的相似性。
在 E 步中，根据 $\theta$ 更新刚性映射 $W$。如式 (11) 定义的平滑性所示，我们采用前向-后向算法推断隐马尔可夫链中的 $W$：
$$
q(W_t^j) = \frac{1}{A} m_f(W_t^j) m_b(W_t^j) \quad (12)
$$
其中：

$A$ 为归一化因子。
$m_f(W_t^j)$ 和 $m_b(W_t^j)$ 分别为 $W_t^j$ 的前向与后向消息，通过递归计算得到：

$$
m_f(W_t^j) = P_{t,j}^{\text{ems}} \sum_{W_t^{j-1}} m_f(W_t^{j-1}) P(W_t^j \mid W_t^{j-1}) \quad (13)
$$
$$
m_b(W_t^j) = \sum_{W_t^{j+1}} m_b(W_t^{j+1}) P_{t,j+1}^{\text{ems}} P(W_t^{j+1} \mid W_t^j) \quad (14)
$$
此处 $P_{t,j}^{\text{ems}}$ 为式 (2) 中的发射概率：
$$
P_{t,j}^{\text{ems}} = P\left(X_t^{\pi_t^{-1}(\theta_j)} \mid \theta_j, W_t^j; T\right)
$$

## 位姿更新

在固定深度图θ和刚性映射W的条件下，本文中通过光流链 $X$ 更新相机位姿。具体流程如下：

1. 基于稠密PnP的位姿估计
3D-2D对应关系构建：利用深度图提取3D点，结合光流计算其在相邻帧的2D投影，形成稠密PnP（Perspective-n-Point）问题[99]。
相对运动估计：以t−1时刻相机坐标系为参考系，通过3D-2D匹配求解相对位姿变换。
鲁棒性保障：采用蒙特卡洛采样和P3P（最小化3点求解）策略，避免初始化依赖并抑制外点影响。该方法对视觉里程计系统的冷启动（§5.3）至关重要，即使刚性映射无先验信息（全1初始化）仍能有效估计位姿。
2. 最大后验概率（MAP）建模
位姿优化问题可表述为：
$$
T^* = \arg\max_T P(T \mid X; \theta, W) \quad (15)
$$
由于直接计算后验分布$P(T \mid X; \theta, W)$需对T积分（边缘化$P(X)$），我们采用蒙特卡洛近似：

分组采样：对深度图每个位置$θ^{j1}$，随机采样两个不同位置{j2,j3}，构成三元组Θ_g = {θ^{j1}, θ^{j2}, θ^{j3}}，并关联刚性值W_g^t = {W_{j1}^t, W_{j2}^t, W_{j3}^t}。
后验近似：

$$
P(T \mid X; \theta, W) \approx \prod_t \left[ \frac{1}{S} \sum_{g=1}^S P\left(T_t \mid X_t^{\pi_{t-1}(\Theta_g)}; \Theta_g, W_g^t \right) \right] \quad (16)
$$
其中S为总组数。每组通过P3P算法[24,48,55,42]高效求解：
$$
\hat{T}_g^t = \phi\left( \left( \prod_{i=0}^{t-1} T_i \right) \Theta_g K^{-1}[x_g \, y_g \, 1]^\top, \, \pi_{t-1}(\Theta_g) + X_t^{\pi_{t-1}(\Theta_g)} \right) \quad (17)
$$

$\phi(\cdot,\cdot)$为P3P求解器（采用AP3P[42]）。
输入1：t−1时刻3D点（由历史位姿链变换得到）。
输入2：t时刻2D对应点（通过光流位移计算）。

3. 变分分布与后验融合
高斯近似：用正态分布$q(T_g^t) \sim \mathcal{N}(\hat{T}_g^t, \Sigma)$近似真实后验（Σ为固定协方差矩阵）(18)。
加权融合：根据刚性值权重$\|W_g^t\| = \prod_{W_i \in W_g^t} W_i$抑制外点，近似全局后验：

$$
P(T \mid X; \theta, W) \approx \prod_t \left[ \frac{1}{\sum_g \|W_g^t\|} \sum_g \|W_g^t\| q(T_g^t) \right] \quad (19)
$$

4. 李代数空间MeanShift优化问题转化：后验分布的众数求解等价于在高斯核（协方差Σ）下应用MeanShift[10]。
李代数处理：将位姿$\hat{T}_g^t \in SE(3)$通过对数映射转换为6维向量$p = \log(\hat{T}_g^t) \in \mathfrak{se}(3)$，在向量空间执行MeanShift，确保优化兼容流形结构。

## 算法流程

如表1所示，算法输入为稠密光流序列 $\mathbf{X} = \{\mathbf{X}_t \mid t = 1, \cdots, t_N \}$，输出为每帧相机位姿 $\mathbf{T} = \{\mathbf{T}_t \mid t = 1, \cdots, t_N \}$ 以及首帧深度图 $\theta$。通常每批次处理4-8个光流场。
初始化阶段：

将所有权重矩阵 $\mathbf{W}$ 初始化为全1矩阵
首帧位姿 $\mathbf{T}_1$ 通过以下方式获取：

基于光流场 $\mathbf{X}_1$ 通过最小中值平方估计器[68]计算本质矩阵
若存在先前估计结果（如连续帧批次重叠），则复用历史位姿


通过 $\mathbf{T}_1$ 与 $\mathbf{X}_1$ 的双视图三角化计算初始深度图 $\theta$

优化循环（通常3-5次迭代收敛）：
$$
\begin{equation}
\min_{\mathbf{T}, \theta, \mathbf{W}} \sum_t |\mathbf{W}_t \odot (\mathbf{X}_t - \pi(\mathbf{T}_t \theta))|_F^2
\end{equation}
$$
其中 $\odot$ 表示哈达玛积，$\pi(\cdot)$ 为投影函数。
关键说明：

在更新相机位姿前，不对刚性映射 $\mathbf{W}_t$ 做平滑处理，以保留观测数据中可能的高频噪声细节（见公式(11)的平滑性约束）

![Algorithm Integration](/img/VOLDOR/algorithm_integration.png)