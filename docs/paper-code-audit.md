# 论文—代码一致性审计

## 审计范围与判定规则

本报告以 `docs/paper.md` 为算法原理、数学定义、变量语义和理论结构的最高优先级依据，审计 `src/`、`configs/`、`scripts/`、`tests/` 及基线实验链路。状态含义如下：

- **Exact**：公式、变量语义和实现直接一致。
- **Equivalent**：写法不同但数学等价。
- **Approximation**：论文允许或工程实现需要的近似，适用条件已明确。
- **Missing**：论文有定义，审计前没有实现。
- **Conflict**：审计前实现与论文定义冲突。
- **Unsupported assumption**：审计前或当前工程参数缺少论文依据。

本次没有通过移动障碍、修改路径、降低噪声或放宽风险阈值来改善实验结果。论文公式修正后出现的不可行状态被保留并显式报告。

## 公式与代码映射

| 论文公式/概念 | 对应代码 | 当前实现 | 审计前状态 | 本次修改 | 近似条件或待确认事项 |
|---|---|---|---|---|---|
| 式(2)：`s_m,g_m,r_m,t_k,tau_H` | `network/timestamp_buffer.py::CommandEnvelope, TimestampBuffer`; `network/delay_channel.py` | 显式计算 `r-g`、`g-s`、`t-s`，并验证因果顺序 | Exact，但分解量没有 API/测试 | 是 | 仿真驾驶端收到状态后立即生成命令，即 `g_m` 没有额外人工响应延迟；这是未校准的工程假设 |
| `tau_H` 与逐障碍 `tau^S_{j,k}` | `network/timestamp_buffer.py`; `safety/robust_cbf_qp.py::RelativeStateObservation`; `runtime.py::StepInput` | 两种年龄由不同对象持有，安全接口逐障碍接收 `age` | Conflict：安全层使用单一标量 `tau_s` | 是 | 仿真场景目前给各障碍相同的传感器年龄，但运行时和测试允许彼此不同 |
| 四类轨迹：人工意图、自主、共享执行、障碍 | `risk/prediction_tube.py::PredictionRiskTube`; `runtime.py::_build_prediction_problem` | 四种对象使用独立字段和类型；共享执行轨迹由混合控制经动力学/线性化传播 | Missing/Conflict：没有结构化风险管，未来人工路径被当作 oracle | 是 | 已校准 SCAND 模型存在时传播非零 `Sigma_H`；模型或历史不足时才显式标记因果常值保持、`Sigma_H=0` |
| 式(3)：差速车 Euler 离散 | `dynamics/diff_drive.py::step, rollout, jacobians` | 直接实现论文 Euler 更新 | Conflict：原实现为精确圆弧积分 | 是 | Euler 离散误差应由独立执行误差界覆盖，当前数值尚未物理标定 |
| 式(4)：控制层混合后传播 | `risk/prediction_tube.py::blend_controls, trajectory_positions`; `dynamics/diff_drive.py::rollout_gaussian_controls`; `runtime.py` | `u=u_H+alpha(u_A-u_H)` 后通过车辆模型传播；控制预测的联合协方差经局部 Jacobian 传播为位置 `Sigma_H` | Exact/Approximation | 是，接入校准预测 | 均值使用非线性 Euler 传播，协方差使用均值轨迹处一阶传播；未进行位置轨迹逐点插值 |
| 式(5)：置信椭圆 | `types.py::GaussianTrajectory`; `risk/prediction_tube.py::tube_radius` | 保存均值/协方差并提供外接圆半径工具 | Approximation | 否 | 在线上层约束直接使用方向投影标准差，不显式构造椭圆对象；数学上与式(10)所需投影一致 |
| 式(6)：预测风险管 `T_k` | `risk/prediction_tube.py::PredictionRiskTube`; `runtime.py::_build_prediction_problem` | 保存 `E_H,p_A,E_E,E_O,C,U` 及三类安全裕量 | Missing | 是 | 当前对象是每次固定线性化参考处的风险管快照，不表示整段闭环联合置信区域 |
| 式(7)–(8)：`C_i,U_i` | `risk/conflict.py::trajectory_conflict_and_uncertainty` | 按 `D_p` 归一化，分别计算均值分歧和协方差迹 | Missing | 是 | 仿真使用 `D_p=diag(1 m,1 m)`；正式实验应由作者确认物理归一化尺度 |
| 式(9)：`Sigma_Z` 与交叉协方差 | `risk/chance_constraint.py::relative_position_covariance` | 支持完整交叉项；缺省不允许静默设为零 | Conflict：直接相加，隐式假设独立 | 是 | 确定性仿真显式声明执行与障碍预测误差独立；真实数据需验证或改用保守标准差和界 |
| 式(10)：半空间机会裕量 | `risk/chance_constraint.py::chance_margin`; `runtime.py::_build_prediction_problem` | 固定单位法向，距离中加入 `kappa*sqrt(n^T Sigma n)` | Exact/Equivalent | 是，补显式法向 | 标准差而非方差进入裕量；固定法向是论文指定的局部凸化条件 |
| 式(13)：共享输入关于 `alpha` 仿射 | `risk/prediction_tube.py::blend_controls` | 向量化实现同一公式 | Equivalent | 否 | 无额外残差控制 `r_i` |
| 式(14)–(17)：局部线性化、`S` 与 `g` | `authority/linearization.py::affine_state_prediction`; `authority/sensitivity.py::project_position_rows`; `runtime.py::_build_prediction_problem` | 使用论文 Jacobian 凝聚为 `x=offset+S alpha`，再投影 `g=n^T S` | Conflict：使用非线性有限差分，且每次固定全自主参考 | 是 | 成功求解后移位上一可行 `alpha` 作为下一参考；局部问题无解时按论文允许的计算预算追加一次端点重线性化，不放宽硬约束 |
| 线性化启动参考 | `runtime.py::SharedControlRuntime` | 首周期无上一可行解时使用零自主权序列，之后移位上一可行解 | Unsupported assumption | 是 | 论文没有规定首次初始化；选择零序列与第一级最小自主权目标一致，且不改变任何硬约束 |
| 式(18)：控制权可行域 | `authority/constraints.py::build_authority_constraints`; `authority/lexicographic_qp.py` | 支持机会、输入、输入变化率、控制权变化率、轮速、走廊、状态/控制可信域等硬约束；仿真当前启用机会约束和输入上下界 | Missing/Conflict | 是 | 论文/配置未给实际变化率、轮速、走廊和可信域数值，因此这些项没有在仿真中臆造启用，仍属重要 Missing |
| 式(19)：次级目标 | `authority/lexicographic_qp.py::secondary_objective_matrices` | 实现控制权平滑、相对人工输入修改、任务轨迹项及极小唯一性正则项 | Conflict/Approximation | 是 | 删除无论文依据的线速度差目标；任务项以自主局部计划作 `p_task` 的能力已实现，但开发/验证选择 `w_P=0`；意图轨迹保持项仍未启用，因为论文没有给出 `w_H,W_i,D_p` 的实验数值 |
| 式(20)：第一级最低累计自主权 | `authority/lexicographic_qp.py::budget_weights, LexicographicAuthority.solve` | 归一化 `Delta_i/T_H`，零 Hessian 的严格 LP | Conflict：曾加入小二次正则且预算量纲不一致 | 是 | OSQP 以零 Hessian 求解 LP，与线性规划数学等价；出口额外复核全部硬约束残差 |
| 式(21)：第二级预算 | `LexicographicAuthority.solve` | 独立第二次求解并硬约束 `I_alpha<=A*+delta_alpha` | Conflict：旧预算与 `delta_alpha` 单位不一致 | 是 | 在不使用正式 seed 0–49 的开发/验证中选择 `delta_alpha=0.01`；它是工程标定值，不是论文初稿给定值 |
| 上层不可行处理 | `LexicographicAuthority.solve`; `runtime.py::SharedControlRuntime.step` | 显式保存 stage 状态和不可行率；不增加松弛；把标记为不可行的名义命令交给安全层 | Conflict：曾静默返回 `alpha=0` 作为普通解 | 是 | 当前平台后备仍以 `alpha=0` 生成名义命令；论文没有唯一指定上层不可行后的名义命令，需作者确认 |
| 式(23)–(24)：源时刻相对状态与年龄传播 | `safety/robust_cbf_qp.py::RelativeStateObservation.mean_at`; `safety/observations.py` | 保存源时刻相对位置/速度并传播到当前时刻 | Conflict：旧接口直接使用当前位置并另加年龄半径 | 是 | 2-D 仿真没有真实传感器 FIFO，使用常速度反向构造源时刻估计再正向传播；仅适用于当前恒速障碍仿真 |
| 式(25)：协方差传播 | `RelativeStateObservation.covariance_at` | 使用 `P_pp+tau(P_pv+P_vp)+tau^2 P_vv`，并验证联合协方差半正定 | Missing | 是 | 仿真协方差为无交叉项的各向同性模型，真实系统需估计四个块 |
| 式(26)–(27)：误差集合与支撑函数 | `RobustCBFFilter::_deterministic_radius, _directional_support, _radial_bound` | 分开计算确定性球、方向支撑和外接球半径 | Missing/Conflict | 是 | `b_clk,b_model,b_act` 及加速度界来自配置但未做物理标定；下一步使用同一源数据的 `tau+T_s` 传播 |
| 式(28)–(29)：前视点及包围半径 | `RobustCBFFilter.filter`; `observations_from_current_estimates` | `G(psi)` 同时包含 `v,omega`，半径增加前视距离 | Conflict：旧实现用质心且忽略 `omega` | 是 | 圆形车体和障碍外包络是保守几何近似 |
| 式(32)：采样间相对移动 | `RobustCBFFilter.filter` | `bar_R=R_phys+bar_V*T_s` | Missing | 是 | 恒速仿真中障碍速度范数就是上界；真实障碍最大速度及最大命令保持间隔需 Windows/实车标定 |
| 式(33)–(36)：`H+`,`H-` 与离散鲁棒约束 | `RobustCBFFilter.filter`; `evaluation/metrics.py` | 计算并记录 `H-`，使用 `H+` 构造逐障碍硬仿射约束 | Conflict：旧连续式近似且含松弛 | 是 | 若初始 `H-<0` 或 QP 不可行，论文前向不变性前提不成立；代码如实报告，不伪造证书 |
| 式(37)：最终安全过滤 QP | `RobustCBFFilter.filter` | 两变量严格凸 QP，目标仅为相对 `u_N` 的最小修改，无松弛 | Conflict 后修正 | 是 | 无解时执行停止后备并标为 infeasible；停止对移动障碍不自动构成安全保证 |
| `Sigma_H` 不得进入安全过滤器 | `RobustCBFFilter.filter` 接口及测试 | 安全接口只接收执行侧相对观测和误差协方差 | Conflict 风险已消除 | 是 | 有接口反射测试防止回归 |
| 理论概率保证 | `predictors/calibration.py`; 测试与审计报告 | 仅提供校准工具，不宣称默认参数具有真实覆盖率 | Unsupported assumption 风险 | 是，收紧表述 | 未使用独立数据校准前，只能报告经验结果，不能声称式(12)/(40)概率保证 |
| SCAND 人工控制先验与 `Sigma_H` | `scripts/run_e0.py`; `scripts/tune_e0.py`; `predictors/human_ar.py`; `dynamics/diff_drive.py::rollout_gaussian_controls`; `runtime.py` | 按完整 run 划分，A 驾驶员训练/验证、B 驾驶员测试；训练 run 内交叉验证选择阶数，验证集校准后将联合控制协方差传播为位置协方差 | Missing | 是 | 当前为 13-run Jackal 子集；AR 未使用环境或局部目标特征，跨驾驶员 95% 经验覆盖率为 93.47%，不能声称达到严格 95% 保证 |
| THÖR 障碍预测与真实轨迹回放 | `scripts/run_e0.py`; `predictors/obstacle_cv.py`; `sim2d/scenarios.py::thor_crossing_obstacle` | 按完整 recording 划分；训练/验证拟合常速度协方差；held-out 轨迹仅经刚体变换后回放 | Missing | 是 | 常速度单峰高斯仅是短时工程近似；无外部处理数据时场景显式记录 `synthetic-fallback` |
| WSL/Windows 共用算法入口 | `runtime.py::SharedControlRuntime.step, StepInput, StepOutput`; `sim2d/simulation.py` | 仿真与将来的 Webots 控制器使用同一无平台依赖的算法调用；输入显式分开上层障碍估计和执行侧安全观测 | Missing | 是 | Windows 仍需实现 Webots 传感、规划、手柄和执行适配器；不能复用 WSL 的 Linux `.venv` |

## 基线与实验公平性

五种方法共享同一 `World`、Euler 车辆模型、人工命令生成策略、自主局部控制器、障碍轨迹、网络配置、随机 seed 和末级安全过滤器。相同 seed 的独立运行产生相同的网络年龄样本前缀，已有回归测试覆盖。由于这是闭环比较，不同方法形成不同车辆状态，因此实际生成的状态反馈式人工/自主输入数值可以随闭环状态不同；共享的是生成策略和参数，而不是强制重放同一数值序列。

- `delay_agreement` 只替换控制权分配规则。
- `single_step` 只保留第一个预测步的机会约束，同时保留与完整方法相同的非风险硬约束。
- `weighted_sum` 使用完整未来可行域和相同次级目标/归一化，只把两级优化替换为单级加权和。
- `human_filter` 令上层自主权为零，但保留相同末级安全过滤器。
- `ours` 使用完整多步可行域、两级预算和相同末级过滤器。

安全隔离实验从同一份 `u_N` 日志重放 `none/standard/robust` 三种执行层，因此不会因重新规划名义命令破坏控制变量。当前场景坐标、障碍位置、网络噪声和风险阈值不是论文初稿给出的实验参数；其中部分在论文提供前曾为 smoke 可运行性调整。本次审计没有继续修改这些数值，它们不能作为论文参数已有依据的证据。

## 当前工程近似

1. 人工意图默认加载 13-run SCAND 子集训练/验证得到的透明 AR 模型；其控制均值通过非线性车辆模型传播，联合控制协方差在均值轨迹处一阶传播为位置 `Sigma_H`。前 `order` 个因果历史样本尚未积累或产物缺失时，显式退回常值保持及零位置协方差。
2. 障碍预测采用 THÖR 训练/验证得到的短时单峰常速度高斯模型；它只适用于当前短时动态障碍场景，不表达多模态行人意图。crossing 的真值来自 held-out THÖR track，仅刚体变换其坐标。
3. `Sigma_E` 是固定各向同性位置协方差上界，没有随 `alpha` 变化；这符合“求解前冻结协方差”的凸化条件，但数值尚未由执行实验校准。
4. 仿真中 `Sigma_E` 与 `Sigma_O` 的独立性被显式配置为真；真实传感/定位系统若存在共同误差源，必须提供交叉协方差或采用保守标准差和界。
5. 上层先在移位的上一可行解处做局部凸化；若局部问题无解，最多在相反控制权端点追加一次重线性化。该固定计算预算符合论文允许的实现范围，不声称原始非线性问题的全局最优。
6. 安全仿真适配器从当前精确仿真状态按恒速模型反推源时刻相对状态，以检验年龄传播公式；Windows 传感器接入必须改为真实源时间戳观测。
7. 末级误差集合把下一步视为同一源观测继续老化到 `tau^S+T_s`；论文没有给出传感器在下一周期刷新时的唯一误差集合参数化。

## 尚未实现或需作者确认

1. 式(18)所需的真实输入变化率、控制权变化率、轮速上限、凸可行驶走廊、状态/控制可信域数值均未在论文或配置中给出。构造器已实现，但正式仿真不能在没有物理依据时自行填数。
2. 式(19)排版中平滑项与 `w_u` 项之间、意图项与 `w_P` 项之间缺少可见加号。上下文表明应为求和，但排版应由作者确认。
3. 论文没有规定首次求解没有上一可行轨迹时的线性化参考。当前采用零自主权启动，并在局部无解时允许一次端点重线性化；二者均是显式工程初始化策略。
4. 论文要求上层不可行时交给安全层或平台后备流程，但没有指定此时 `u_N`。当前使用人工候选 (`alpha=0`) 并显式标记不可行，不把它当作优化解。
5. `E^+_{j,k}` 如何结合下一次传感器刷新、时钟误差和执行误差没有唯一数值模型。当前采用同一源观测老化 `T_s` 的保守实现。
6. 障碍最大速度、最大命令保持间隔、执行器变化率和全部不确定性界缺少实测标定。`cbf_gamma=0.8` 满足论文 `0<gamma<=1`，但数值本身并非论文实验给定。
7. 论文实验章节没有给出完整数值设置或结果，无法判断当前四个 smoke 场景是否对应论文正式实验，也不能以当前指标反向验证算法定义。
8. 当前没有真实传感器日志或平台认证后备控制，因此式(40)的有限时域概率保证前提尚未建立。
9. SCAND Jackal 当前只下载并审计了 13 个选定 run，并非完整 SCAND；这足以形成 A 到 B 的跨驾驶员 E0 测试，但不能冒充对完整数据集的验证。

## 验证记录

本次审计提交前的验证结果如下。结果用于检查实现一致性和失败可见性，不用于宣称论文方法优于基线。

- 数据核验：13 个 SCAND Jackal bag 均可读取并转换；13 个 CSV 一一对应 8 个 A run 和 5 个 B run，共 44,452 行。时间戳严格递增，状态/控制无 NaN/Inf，且 `|v|<=2.0 m/s`、`|omega|<=1.4 rad/s`。13 个 THÖR 3-D TSV 的官方 MD5 均通过，转换得到 1,861,429 个观测及 1,116 个连续 track segment。
- E0 SCAND：seed 17，在 6 个 A 训练 run 内按未校准 NLL 交叉验证选择 AR(3)，另用 2 个 A run 校准，全部 5 个 B run 仅作测试；1,825 个训练窗口、575 个校准样例、6,359 个测试样例。校准尺度 1.4018；测试 `ADE=0.3765`、`FDE=0.4938`（二者混合控制量纲，仅作辅助）、`MAE_v=0.3206 m/s`、`MAE_omega=0.1244 rad/s`、`NLL=0.7301`、90/95/99% 经验覆盖率为 90.88/93.47/96.14%。95% 覆盖未达到名义值，不使用测试数据二次校准。
- E0 THÖR：在训练 recording 内选择 0.3 s 历史窗口，随后按完整 recording 进行 7/3/3 训练/校准/测试划分；19,460/7,058/7,479 个样例。测试 `ADE=0.1715 m`、`FDE=0.3498 m`、`NLL=-0.1719`、90/95/99% 经验覆盖率为 93.43/96.11/98.54%。
- 全量测试：`77 passed`，语句覆盖率 `86%`。新增测试覆盖因果重采样、延迟回调中按 `r_m` 采样、单调平台时钟、首回调已有命令、20 Hz 上层节拍、无 run/recording 泄漏、无测试集校准、任务轨迹项、两级预算、重线性化诊断、模型序列化、联合控制协方差到 `Sigma_H` 的传播、THÖR 刚体回放及平台运行时/安全接口。
- 静态与格式：Ruff check 通过，64 个 Python 文件格式检查通过，`git diff --check` 通过。
- 构建与导入：`compileall`、核心包导入及 `pip check` 通过。
- 配对 smoke：四个场景、五种方法、相同 seed 17、每次 3 s，共 20 次运行。真实 THÖR crossing 中 `single_step` 与 `human_filter` 碰撞；其余 18 次未碰撞。Ours 在该单次 crossing 的最小净空为 0.033 m；单 seed 不能支持方法优越性的统计结论。
- 上层状态：1,200 次共享控制更新中固定规则基线 480 次未运行 QP；其余记录为 253 次两级 `solved/solved`、1 次 `solved/solved inaccurate`、92 次单级 `solved`、347 次 `primal infeasible`、27 次 `maximum iterations reached`。失败显式进入后备，没有松弛或伪可行解。
- 末级状态：3,000 个执行步中 2,333 次 `solved`、1 次 `solved inaccurate`、546 次 `primal infeasible`、120 次 `maximum iterations reached`。无解时执行停止后备，但停止不被记录为安全证书。
- 安全隔离：相同 `u_N` 的 3 s `network_anomaly` 重放中，无过滤发生碰撞；standard 和 `tau^S=0/0.05/0.1/0.2 s` robust 均未碰撞，对应不可行率 43.3/52.0/52.0/52.7/54.7%。对真实 THÖR crossing 的相同人类名义命令，none、standard 和全部 robust 年龄均碰撞，过滤器不可行率约 35.3–38.7%；这符合论文“持续可行”是安全结论前提，不能把停车后备宣称为无条件保证。
- 数值/物理扫描：全部核心状态、候选控制、名义/过滤控制、`alpha`、年龄、距离、`H-`、预测裕量和求解时间无 NaN/Inf；`alpha` 及最终控制满足配置边界；同一场景五种方法的 `tau_H` 样本序列一致。

高不可行比例说明当前场景、固定法向局部近似、标量控制权混合和未标定参数组合经常不满足论文的持续可行性假设。由于论文实验参数尚为空，本次没有通过放宽约束或移动场景元素来降低该比例。
