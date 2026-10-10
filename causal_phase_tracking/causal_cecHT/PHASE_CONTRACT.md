# causal cecHT Phase Contract v1

初始批准日期：2026-08-06  
250 Hz 重基线批准日期：2026-08-07  
机器合同：`contracts/causal_cecht_phase_contract.json`  
状态：步骤一至步骤四 core、步骤六 pseudo-trigger、步骤七 100->250 因果 adapter 与离线 reference/replay 已实现；当前有效 core 为 exact 250 Hz-only，host wrapper 与正式 EDF 评价尚未实现。

## 1. 权威范围

本目录采用独立的 scoped machine contract。它对 causal cecHT 的流式估计、pseudo-trigger 和对应 simulation evaluation 具有权威性；项目级 `realtime_deployment/contracts/phase_tracking.json` 保留跨方法和 host 集成约束。两份合同的共享字段必须由测试保持一致；发生冲突时测试失败并要求项目所有者决定，禁止静默选择。

本合同不能削弱项目级 shadow-only、fail-closed 或无真实刺激约束。

## 2. 输入与相位

- 输入固定为 250 Hz，即每个样本 4 ms。非250 Hz输入 fail closed；本阶段不在 core 内重采样。
- 信号固定为 `Fpz-Cz`；Sleep-EDF 通道标签为 `EEG Fpz-Cz`。
- 极性固定为 `Fpz_minus_Cz`，公共单位固定为 uV。
- phase 使用 radian，规范化到 `[-pi, pi)`；`+pi` 规范成 `-pi`。
- `0 rad` 表示 Fpz-Cz 正峰，正方向表示 analytic phase 随时间增加。

正峰、analytic phase 和 endpoint 语义来自 Hilbert/ECHT 定义。[R1-R3] 圆周规范化和统计应按 circular statistics 处理，不能直接用普通线性均值替代。[R6]

## 3. Endpoint 和更新时间

主实现只允许每个 250 Hz 输入样本更新一次：`interval_samples=1`、`interval_s=0.004`。当前主窗口为 8 秒 = 2000 点、exact `n_fft=2000`；secondary fast 回归为 2001 点并在冻结环境解析到 2016 点。窗口取代关系见 8.2。不保留其他 update-grid sensitivity 分支。

cecHT estimate 的时间位置是 trailing window 的最后一个输入样本。未来从上游 ECHT 读取解析输出时，只能把 `z[-1]` 当作当前 endpoint；上游论文、仿真和 EEG loop 均以最新端点为目标。[R1-R3]

步骤二已固定上游 commit、许可证和 `z[-1]`；步骤四 steady-state使用构造期导出的 endpoint kernel，不在热路径调用 pinned FFT/transform。

## 4. Error sampling

连续 estimator error：

```text
e_est(t) = wrap(phi_estimate(t) - phi_reference(t))
```

它在每个 valid、ready 的 250 Hz endpoint 采样；正值表示 estimate 领先 reference。上游 cecHT EEG evaluation 使用 estimate-minus-reference 的方向。[R1-R2]

事件 trigger error：

```text
e_trigger(k) = wrap(phi_reference(t_event,k) - theta_target(k))
```

它只在 pseudo-event 采样；正值表示 reference 在事件时已经越过 target。必须分别在 `estimated_crossing_time` 和 `decision_time` 评价。睡眠闭环评价需要把连续 estimator 精度与 target-hit/event timing 分开报告。[R4-R5]

## 5. Target 和 crossing

- 默认 `theta_target=0 rad`，运行时可配置。
- evaluation grid 为八个等间隔圆周位置：`0, pi/4, pi/2, 3pi/4, -pi, -3pi/4, -pi/2, -pi/4`。
- target 在声明的 sample/block boundary 原子更新。
- 更新 target 时清空 crossing comparator、取消 pending event，不重放旧 crossing。
- 只允许正向 crossing；必须正确处理 `-pi/pi` wrap。
- 不记录反向 crossing 诊断事件。
- 初始 estimator 和 pseudo-trigger 均不使用 amplitude eligibility。

target-grid、target-hit circular error 和 event timing 的评价角度参考 Ferster 等的实时睡眠慢波算法 benchmark。[R5]

### 5.1 步骤六已实现内容（2026-09-10）

- `causal_phase_tracking/causal_cecHT/pseudo_trigger.py` 实现独立 `CausalCecHTTrigger`，只消费 valid/ready 逐样本相位。
- default target `0 rad` 可运行时更新；更新清 comparator 并取消 pending。
- positive-only crossing，相邻 endpoint 线性插值；event 保存 `crossing_time_s` 与 `decision_time_s`，二者均不是 acoustic onset。
- `refractory_s=0.5` 仅为开发期占位，留待真实刺激阶段调参；无 amplitude gate；所有补偿为 0。

### 5.2 步骤七已实现内容（2026-09-10）

- `causal_simulation_evaluation/rate_adapter.py`：strict causal rational polyphase FIR，100->250，kernel 500 Hz/40 Hz/33 taps，group delay 8 输出样本（0.032 s），样本不内部移位。
- `edf_replay.py`：EDF `EEG Fpz-Cz` loader/QA 与确定性 chunked replay。
- `offline_reference.py`：非因果 5/2 reference resample + `0.5-1.25 Hz` Butterworth `sosfiltfilt` + Hilbert；禁止进入在线路径。
- `evaluation_pipeline.py`：adapter -> core -> trigger 整链执行；`artifacts/smoke_metrics.json` 为 development smoke 结果，不是性能或泛化结论。

## 6. 时间与补偿

每个 estimate 必须携带 endpoint sample index 和 acquisition timestamp。pseudo-event 必须区分：

- `estimated_crossing_time`：两个相邻有效 endpoint 之间线性插值出的 crossing；
- `decision_time`：当前 endpoint callback 得知 crossing 的时刻。

两者都不是 acoustic onset。当前 estimator offset、scheduler prediction horizon 和 acoustic output delay 全部为 0。真实调度/声学补偿只有在以后测得相应延迟后才能启用；Santostasi 等明确区分算法目标相位、系统延迟和离线 reference 评价。[R4]

## 7. Invalid 和安全边界

not-ready 或 invalid 时，phase、amplitude、frequency 必须为 null，不能沿用旧值，也不能产生 pseudo-event。segment change、gap、乱序、重复、NaN/Inf、采样率/单位/通道/极性变化及显式 reset 都属于 fail-closed 输入。

本阶段只允许 timestamped pseudo-event。禁止声卡、硬件刺激 API 和人体刺激；`decision_time` 禁止标记为 `acoustic_onset`。

## 8. 项目所有者批准的 D1-D10

| 决策 | 已批准内容 |
|---|---|
| D1 | B：独立 scoped machine contract，并与项目级合同同步检查 |
| D2 | 默认 target 0 rad，evaluation 使用八点 grid |
| D3 | estimator error 为 estimate-reference；trigger error 为 reference-target |
| D4 | 公共输入单位 uV |
| D5（2026-08-06历史） | 原批准每样本100 Hz更新；已由2026-08-07的250 Hz重基线取代 |
| D6 | 只接受正向 crossing，不记录反向 crossing 诊断事件 |
| D7 | 初始 estimator 和 pseudo-trigger 均不设 amplitude gate |
| D8 | Markdown + JSON + contract tests |
| D9 | 步骤一允许修改现有项目级 `phase_tracking.json` |
| D10 | 不自动 commit；每步验收后再由项目所有者决定 |

### 8.1 2026-08-07 采样率重基线

- 当前有效输入和update grid均为exact 250 Hz，每样本更新。
- 6秒窗口的物理时长保持不变，因此主profile从600点改为1500点，exact `n_fft`同步改为1500。
- `0.5-1.25 Hz`、order 2 Butterworth、fixed `f0=0.875 Hz`、calibration、endpoint、相位和fault/reset语义不变。
- 原exact 100 Hz core边界被取代，但旧100 Hz manifest/golden保留为历史证据；当前回归使用文件名含`250hz`的新manifest/golden。
- 当前不实现 resampler。后续若输入不是 250 Hz，必须在 EDF replay 之前建立并验证严格因果、状态持久且 delay-accounted 的 rate adapter。

### 8.2 2026-09-15 窗口与 f0 重基线

- 当前主窗口改为 8 秒 = 2000 点，exact `n_fft=2000`；secondary fast 回归为 2001 点，冻结环境解析到 2016 点。
- fixed `f0` 由 0.875 Hz 改为 0.8 Hz，calibration 保持开启。
- 采样率、相位约定、target 语义、endpoint、fault/reset 与 update grid（每 250 Hz 样本更新一次）均不变。
- 取代关系：本节的窗口与 `f0` 取代 8.1 中的同名条目；8.1 的其余内容仍然有效。
- 依据：canonical 评估基线（8 s、`f0=0.8 Hz`、pooled MAE 1.041248 rad）与已部署的单文件插件均使用该 profile；机器合同见 `contracts/causal_cecht_phase_contract.json` 的 `window_rebaseline`。
- 待迁移：`cecht_synthetic_goldens.json`、`cecht_synthetic_cases.json` 以及 `test_endpoint_kernel.py`、`test_cecht_synthetic_math.py`、`run_smoke_evaluation.py` 仍基于 6 秒 / 1500 点 profile，尚未按 8 秒 profile 重新冻结。

## 9. References

**[R1]** Osmers E, Kolossa D. *Optimal Calibration of the Endpoint-corrected Hilbert Transform*. arXiv:2601.13962v2 (2026). https://arxiv.org/abs/2601.13962

**[R2]** `eikeosmers/cecHT`, pinned source planned for step 2: `e079c3115f202a4272cf64143bf33ddddc247778`. https://github.com/eikeosmers/cecHT/tree/e079c3115f202a4272cf64143bf33ddddc247778

**[R3]** MEEGkit ECHT implementation, BSD-3-Clause. https://github.com/nbara/python-meegkit/blob/master/meegkit/phase.py

**[R4]** Santostasi G et al. *Phase-locked loop for precisely timed acoustic stimulation during sleep*. Journal of Neuroscience Methods 259, 101-114 (2016). https://doi.org/10.1016/j.jneumeth.2015.11.007

**[R5]** Ferster ML et al. *Benchmarking Real-Time Algorithms for In-Phase Auditory Stimulation of Low Amplitude Slow Waves With Wearable EEG Devices During Sleep*. IEEE TBME 69(9), 2916-2925 (2022). https://doi.org/10.1109/TBME.2022.3157468

**[R6]** Berens P. *CircStat: A MATLAB Toolbox for Circular Statistics*. Journal of Statistical Software 31(10) (2009). https://doi.org/10.18637/jss.v031.i10
