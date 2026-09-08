# Hunting Code Defects: A-16 Init Gate & km_prior Resync

The learner mastered the engineering and statistical root cause of the early mortality collapse bug: how PyTorch's default `nn.Linear` initialization ($b=0$) sets output hazard to $h \approx \sigma(0) = 0.5$, causing survival probability and continuous discounting factor $\gamma_j = S(\Delta t_j)$ to collapse to $\approx 0.09$, crippling temporal difference bootstrapping. The learner also verified the fix using empirical Kaplan-Meier marginal hazards (`km_prior`) and the load-bearing `copy.deepcopy` target head re-synchronization (`target_head.load_state_dict(head.state_dict())`).

## Evidence
- `notebooks/05_init_bias_resync.ipynb` was created, executed, and baked with visual comparison of default premature death plunge vs `km_prior` population survival recovery, along with the deepcopy target resync demonstration.
- `lessons/0005-hunting-code-defects-a16-init-gate.html` was published and opened in the browser.

## Implications
- Directly grounds the real research history of the A-16 Init Gate audit (`research/continuous-survival-td/deviation_log.md` and `experiments/a16_init_gate.py`).
- Prepares the author to preempt reviewer attacks regarding neural network initialization stability and baseline parity.
