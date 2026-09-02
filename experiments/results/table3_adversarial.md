# Table 3: Adversarial Falsification Matrix & Kill Criteria Report

| Stress Test ID | Hostile Threat Interrogated | Kill Threshold (Falsifier) | Observed Metric | Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **EXP-04 (NC-B)** | Visit count confounding | Permuted retains > 50% gain | 95.9% gain retained (C=0.806) | **FAILED (Kill Criterion Fired)** |
| **EXP-03 (NC-A1/A2)** | Component bundling | Arm A1 or A2 retains > 50% gain | A1=93.9%, A2=91.8% retained | **FAILED** |
| **EXP-05 (NC-A3)** | Naive clamped division substitute | Clamped division matches stability & C-index | Clamped C=0.738, GradNorm=0.03 | **PASSED (Hypothesis Upheld)** |
| **EXP-06 (NC-C)** | Bootstrapping horizon drift across sampling | Count-geometric matches duration-geometric | Duration-geom C=0.809 vs Count-geom C=0.806 | **PASSED (Hypothesis Upheld)** |
| **EXP-08 (Diffusion)** | Projection variance blowup O(sqrt(n)) | Variance <= delta_s^2 / 6 (1.0417) | Measured avg step var = 0.2961 | **PASSED (Hypothesis Upheld)** |
