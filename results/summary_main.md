# E1 主表（自动汇总；口径优先级 full > pn分层 > 旧limit，来源列可审计）

| run | set | n | viol_f1 | v_mae | state_acc | format | phys_resid | ICS(T3) | 来源 |
|---|---|---|---|---|---|---|---|---|---|
| chronos | id | 999 | 0.330 | 0.0742 | 0.682 | — | — | — | chronos_id |
| chronos | ood_mv | 1000 | 0.221 | 0.0730 | 0.666 | — | — | — | chronos_ood_mv |
| chronos | ood_tx | 999 | 0.185 | 0.0705 | 0.885 | — | — | — | chronos_ood_tx |
| chronos | noise | 999 | 0.242 | 0.0727 | 0.496 | — | — | — | chronos_noise |
| deepseek_fs | id | 300 | 0.350 | 0.0508 | 0.167 | 1.000 | 0.0478 | 0.000 | deepseek_fs_id |
| deepseek_fs | ood_mv | 200 | 0.362 | 0.0644 | 0.175 | 1.000 | 0.0618 | 0.000 | deepseek_fs_ood_mv_pn100 |
| deepseek_fs | noise | 300 | 0.413 | 0.0715 | 0.130 | 0.987 | 0.0646 | 0.000 | deepseek_fs_noise_pn100 |
| deepseek_fs | holdout | 461 | 0.236 | 0.0976 | 0.150 | 1.000 | 0.0834 | 0.000 | deepseek_fs_holdout |
| deepseek_zs | id | 300 | 0.359 | 0.2253 | 0.000 | 0.803 | 0.0870 | 0.000 | deepseek_zs_id |
| deepseek_zs | ood_mv | 200 | 0.360 | 0.3218 | 0.005 | 0.710 | 0.3483 | 0.000 | deepseek_zs_ood_mv_pn100 |
| deepseek_zs | noise | 300 | 0.411 | 0.2128 | 0.000 | 0.810 | 0.1058 | 0.000 | deepseek_zs_noise_pn100 |
| deepseek_zs | holdout | 461 | 0.236 | 0.1550 | 0.000 | 0.866 | 0.1100 | 0.000 | deepseek_zs_holdout |
| diag | ood_tx | 20 | 0.000 | 0.5159 | 0.100 | 0.500 | 0.0299 | 0.420 | diag_ood_tx |
| dpo_v2 | id | 300 | 0.353 | 1.0000 | 0.000 | 0.000 | — | 0.000 | dpo_v2_id_pn |
| dpo_v3 | id | 300 | 0.353 | 1.0000 | 0.000 | 0.000 | — | 0.000 | dpo_v3_id_pn |
| dpo_v3fix | id | 300 | 0.574 | 0.0983 | 0.623 | 0.890 | 0.0314 | 0.543 | dpo_v3fix_id_pn |
| dsse[oracle] | id | 999 | 0.784 | 0.0388 | — | — | — | — | dsse_id |
| dsse[oracle] | ood_mv | 1000 | 0.576 | 0.0298 | — | — | — | — | dsse_ood_mv |
| dsse[oracle] | ood_tx | 999 | 0.062 | 0.0399 | — | — | — | — | dsse_ood_tx |
| dsse[oracle] | noise | 999 | 0.737 | 0.0474 | — | — | — | — | dsse_noise |
| dsse[oracle] | holdout | 461 | 0.711 | 0.1217 | — | — | — | — | dsse_holdout |
| dsse[oracle] | abstain | 150 | 0.769 | 0.0273 | — | — | — | — | dsse_abstain |
| dsse[profile] | id | 999 | 0.826 | 0.0269 | — | — | — | — | dsse_id |
| dsse[profile] | ood_mv | 1000 | 0.634 | 0.0187 | — | — | — | — | dsse_ood_mv |
| dsse[profile] | ood_tx | 999 | 0.400 | 0.0238 | — | — | — | — | dsse_ood_tx |
| dsse[profile] | noise | 999 | 0.718 | 0.0387 | — | — | — | — | dsse_noise |
| dsse[profile] | holdout | 461 | 0.701 | 0.0926 | — | — | — | — | dsse_holdout |
| dsse[profile] | abstain | 150 | 0.759 | 0.0271 | — | — | — | — | dsse_abstain |
| dsse[profile_lnr] | id | 999 | 0.822 | 0.0267 | — | — | — | — | dsse_id |
| dsse[profile_lnr] | ood_mv | 1000 | 0.704 | 0.0173 | — | — | — | — | dsse_ood_mv |
| dsse[profile_lnr] | ood_tx | 999 | 0.311 | 0.0237 | — | — | — | — | dsse_ood_tx |
| dsse[profile_lnr] | noise | 999 | 0.631 | 0.0432 | — | — | — | — | dsse_noise |
| dsse[profile_lnr] | holdout | 461 | 0.704 | 0.0936 | — | — | — | — | dsse_holdout |
| dsse[profile_lnr] | abstain | 150 | 0.793 | 0.0265 | — | — | — | — | dsse_abstain |
| grpo_ps60 | id | 300 | 0.353 | 0.3400 | 0.220 | 0.567 | 0.0438 | 0.338 | grpo_ps60_id_pn |
| grpo_tm60 | id | 300 | 0.353 | 0.3857 | 0.310 | 0.547 | — | 0.341 | grpo_tm60_id_pn |
| qwen7b_zeroshot | id | 300 | 0.313 | 1.0000 | 0.000 | 0.000 | — | 0.000 | qwen7b_zeroshot_id |
| sft15_a0 | id | 300 | 0.529 | 0.1044 | 0.527 | 0.907 | 0.0282 | 0.565 | sft15_a0_id_pn |
| sft15_a0 | holdout | 461 | 0.409 | 0.2565 | 0.419 | 0.794 | 0.0349 | 0.435 | sft15_a0_holdout |
| sft15_a1 | id | 300 | 0.353 | 1.0000 | 0.873 | 0.000 | — | 0.679 | sft15_a1_id_pn |
| sft15_a3 | id | 300 | 0.353 | 0.4541 | 0.000 | 0.213 | 0.0380 | 0.000 | sft15_a3_id_pn |
| sft15_nocurr | id | 300 | 0.471 | 0.1809 | 0.490 | 0.810 | 0.0361 | 0.466 | sft15_nocurr_id_pn |
| sft15_nocurr | holdout | 461 | 0.393 | 0.2917 | 0.362 | 0.679 | 0.0351 | 0.390 | sft15_nocurr_holdout |
| sft15_topo_volt | id | 300 | 0.529 | 0.2204 | 0.470 | 0.000 | 0.0411 | 0.426 | sft15_topo_volt_id_pn |
| sft15_volt_only | id | 300 | 0.577 | 0.3642 | 0.403 | 0.000 | 0.0464 | 0.402 | sft15_volt_only_id_pn |
| sft30 | id | 300 | 0.636 | 0.1836 | 0.637 | 0.820 | 0.0467 | 0.549 | sft30_id_pn |
| sft30 | holdout | 461 | 0.549 | 0.2404 | 0.601 | 0.727 | 0.0966 | 0.420 | sft30_holdout |
| sft_a0_base | id | 300 | 0.361 | 0.5826 | 0.167 | 0.370 | 0.0456 | 0.224 | sft_a0_base_id_pn |
| sft_a0_base | holdout | 461 | 0.239 | 0.7522 | 0.091 | 0.206 | 0.0535 | 0.142 | sft_a0_base_holdout |
| sft_a1_nochain | id | 300 | 0.353 | 1.0000 | 0.653 | 0.000 | — | 0.538 | sft_a1_nochain_id_pn |
| sft_a2_shuffled | id | 300 | 0.337 | 0.4611 | 0.197 | 0.363 | 0.0402 | 0.200 | sft_a2_shuffled_id_pn |
| sft_a3_cheat | id | 300 | 0.353 | 0.9835 | 0.000 | 0.260 | 1.4982 | 0.000 | sft_a3_cheat_id_pn |
| smoke | id | 30 | 0.233 | 1.0000 | 0.000 | 0.000 | — | 0.000 | smoke_id |
| solver[full] | id | 999 | 1.000 | 0.0002 | — | — | — | — | solver_id |
| solver[full] | ood_mv | 1000 | 1.000 | 0.0002 | — | — | — | — | solver_ood_mv |
| solver[full] | ood_tx | 999 | 0.936 | 0.0013 | — | — | — | — | solver_ood_tx |
| solver[full] | noise | 999 | 1.000 | 0.0002 | — | — | — | — | solver_noise |
| solver[full] | holdout | 461 | 1.000 | 0.0002 | — | — | — | — | solver_holdout |
| solver[full] | abstain | 150 | 1.000 | 0.0002 | — | — | — | — | solver_abstain |
| solver[masked] | id | 999 | 0.615 | 0.0512 | — | — | — | — | solver_id |
| solver[masked] | ood_mv | 1000 | 0.401 | 0.0322 | — | — | — | — | solver_ood_mv |
| solver[masked] | ood_tx | 999 | 0.499 | 0.0179 | — | — | — | — | solver_ood_tx |
| solver[masked] | noise | 999 | 0.617 | 0.0506 | — | — | — | — | solver_noise |
| solver[masked] | holdout | 461 | 0.655 | 0.0450 | — | — | — | — | solver_holdout |
| solver[masked] | abstain | 150 | 0.574 | 0.0610 | — | — | — | — | solver_abstain |
| v1 | id | 300 | 0.571 | 0.1143 | 0.600 | 0.867 | 0.0318 | 0.528 | v1_id_pn |
| v1 | ood_mv | 200 | 0.360 | 0.7523 | 0.080 | 0.240 | 0.0201 | 0.072 | v1_ood_mv_pn |
| v1 | ood_tx | 300 | 0.210 | 0.7871 | 0.030 | 0.207 | 0.0327 | 0.167 | v1_ood_tx_pn |
| v1 | noise | 300 | 0.580 | 0.1225 | 0.427 | 0.843 | 0.0381 | 0.419 | v1_noise_pn |
| v1 | holdout | 461 | 0.484 | 0.1770 | 0.518 | 0.811 | 0.0346 | 0.451 | v1_holdout |
| v1 | abstain | 150 | 0.569 | 0.1287 | 0.033 | 0.900 | 0.0390 | — | v1_abstain |
| v2 | id | 999 | 0.661 | 0.2244 | 0.759 | 0.870 | 0.1769 | 0.434 | v2_id_full |
| v2 | ood_mv | 200 | 0.386 | 0.2061 | 0.530 | 0.665 | 0.0209 | 0.475 | v2_ood_mv_pn |
| v2 | ood_tx | 300 | 0.193 | 0.4407 | 0.447 | 0.410 | 0.0286 | 0.473 | v2_ood_tx_pn |
| v2 | noise | 999 | 0.657 | 0.3058 | 0.549 | 0.836 | 0.2631 | 0.397 | v2_noise_full |
| v2 | holdout | 461 | 0.585 | 0.2738 | 0.614 | 0.761 | 0.1610 | 0.428 | v2_holdout |
| v2 | abstain | 150 | 0.736 | 1.7208 | 0.033 | 0.953 | 1.7204 | — | v2_abstain |
| v3 | id | 300 | 0.601 | 0.3616 | 0.367 | 0.530 | 0.0366 | 0.319 | v3_id_pn |
| v3 | holdout | 461 | 0.502 | 0.3056 | 0.275 | 0.390 | 0.0432 | 0.207 | v3_holdout |
| v3 | abstain | 150 | 0.665 | 0.1417 | 0.480 | 0.833 | 0.0318 | — | v3_abstain |
| v3b | id | 300 | 0.688 | 0.1590 | 0.813 | 0.903 | 0.0909 | 0.596 | v3b_id_pn |
| v3b | noise | 300 | 0.705 | 1.1014 | 0.533 | 0.863 | 1.1134 | 0.545 | v3b_noise_pn |
| v3b | holdout | 461 | 0.558 | 0.2994 | 0.633 | 0.770 | 0.1352 | 0.472 | v3b_holdout |
| v3b | abstain | 150 | 0.751 | 2.5615 | 0.033 | 0.947 | 2.5821 | — | v3b_abstain |

## 退化与衰减

| run | viol_f1 保持率 noise/id | ICS 衰减比 holdout/id |
|---|---|---|
| chronos | 0.733 | — |
| deepseek_fs | 1.181 | — |
| deepseek_zs | 1.144 | — |
| diag | — | — |
| dpo_v2 | — | — |
| dpo_v3 | — | — |
| dpo_v3fix | — | — |
| dsse[oracle] | 0.940 | — |
| dsse[profile] | 0.868 | — |
| dsse[profile_lnr] | 0.768 | — |
| grpo_ps60 | — | — |
| grpo_tm60 | — | — |
| qwen7b_zeroshot | — | — |
| sft15_a0 | — | 0.769 |
| sft15_a1 | — | — |
| sft15_a3 | — | — |
| sft15_nocurr | — | 0.838 |
| sft15_topo_volt | — | — |
| sft15_volt_only | — | — |
| sft30 | — | 0.766 |
| sft_a0_base | — | 0.636 |
| sft_a1_nochain | — | — |
| sft_a2_shuffled | — | — |
| sft_a3_cheat | — | — |
| smoke | — | — |
| solver[full] | 1.000 | — |
| solver[masked] | 1.003 | — |
| v1 | 1.015 | 0.854 |
| v2 | 0.994 | 0.987 |
| v3 | — | 0.650 |
| v3b | 1.025 | 0.792 |
