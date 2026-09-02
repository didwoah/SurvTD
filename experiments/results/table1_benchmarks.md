# Table 1: Multi-Cohort Dynamic Survival Performance (EXP-01 & EXP-02)

| Method | MIMIC-IV Sepsis-3 (C / AUC / IBS) | NASA C-MAPSS 50% (C / AUC / IBS) | PBC Trial (C / AUC / IBS) | Tumor Growth ODE (C / AUC / IBS) |
| :--- | :---: | :---: | :---: | :---: |
| **Person-Period (1h)** | 0.263 / 0.500 / 0.783 | 0.100 / 0.167 / 0.865 | 0.593 / 0.538 / 0.409 | 0.409 / 0.278 / 0.338 |
| **Dynamic-DeepHit** | 0.605 / 0.500 / 0.778 | 0.800 / 0.667 / 0.856 | 0.881 / 0.969 / 0.166 | 0.652 / 0.667 / 0.578 |
| **DeepTCSR (Clamped)** | 0.632 / 0.500 / 0.792 | 0.900 / 0.833 / 0.864 | 0.803 / 0.839 / 0.284 | 0.432 / 0.222 / 0.419 |
| **SurvTD (Ours)** | 0.566 / 0.500 / 0.808 | 0.500 / 0.500 / 0.862 | 0.789 / 0.837 / 0.307 | 0.439 / 0.472 / 0.399 |
