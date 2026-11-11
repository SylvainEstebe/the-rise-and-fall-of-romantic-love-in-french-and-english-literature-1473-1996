# French books: length sensitivity

The eighteenth-century decline-threshold date survives length control across all three flexible splines. For df 4–6, adjusted median dates range from 1736 to 1754, compared with 1744–1762 without adjustment. The 1650→1800 and 1800→1950 contrasts remain negative, with 95% intervals entirely below zero in all four adjusted models. Across the adjusted splines, 93.1–97.7% of all posterior draws place the 0.25-point threshold crossing in 1700–1799; 0.2–1.9% never cross within the observed range. These results concern the specified 0.25-point threshold after each draw’s global maximum, not the exact year when the curve first begins to slope downward.

All four adjusted models were refitted to the same 1,879 records using four chains × 4,000 retained draws, 2,000 tuning iterations, and seed 0. The covariate is centred log₂ total segment count; predictions hold it at zero (geometric mean 19.93 segments). The metadata field `n_chapters` counts roughly 15,000-character text segments, not literary chapters. Author deviations are fixed at zero. Year and spline bases follow the original analysis.

| Model | Quantity | Unadjusted | Length adjusted |
| --- | --- | --- | --- |
| Quadratic | 1650→1800 | -0.43 [-0.60, -0.27] | -0.48 [-0.65, -0.31] |
| Quadratic | 1800→1950 | -1.54 [-1.84, -1.23] | -1.47 [-1.77, -1.16] |
| Quadratic | Maximum year; boundary % | 1667 [1619, 1694]; boundary 0.0% | 1652 [1586, 1687]; boundary 0.2% |
| Quadratic | 0.25-point decline year; never % | 1768 [1738, 1789]; never 0.0% | 1760 [1718, 1784]; never 0.0% |
| Spline df=4 | 1650→1800 | -0.71 [-1.02, -0.39] | -0.79 [-1.11, -0.47] |
| Spline df=4 | 1800→1950 | -1.08 [-1.46, -0.68] | -1.00 [-1.39, -0.60] |
| Spline df=4 | Maximum year; boundary % | 1675 [1638, 1713]; boundary 1.1% | 1660 [1624, 1704]; boundary 0.9% |
| Spline df=4 | 0.25-point decline year; never % | 1752 [1719, 1775]; never 1.0% | 1741 [1703, 1771]; never 0.7% |
| Spline df=5 | 1650→1800 | -0.78 [-1.11, -0.45] | -0.85 [-1.18, -0.51] |
| Spline df=5 | 1800→1950 | -1.04 [-1.45, -0.62] | -0.98 [-1.40, -0.54] |
| Spline df=5 | Maximum year; boundary % | 1680 [1629, 1708]; boundary 2.1% | 1665 [1612, 1703]; boundary 2.0% |
| Spline df=5 | 0.25-point decline year; never % | 1744 [1712, 1766]; never 2.0% | 1736 [1690, 1761]; never 1.9% |
| Spline df=6 | 1650→1800 | -0.67 [-1.05, -0.28] | -0.74 [-1.13, -0.35] |
| Spline df=6 | 1800→1950 | -0.96 [-1.39, -0.53] | -0.91 [-1.34, -0.47] |
| Spline df=6 | Maximum year; boundary % | 1694 [1600, 1739]; boundary 0.3% | 1676 [1587, 1735]; boundary 0.4% |
| Spline df=6 | 0.25-point decline year; never % | 1762 [1723, 1784]; never 0.2% | 1754 [1672, 1782]; never 0.2% |

**Note.** Contrasts are later minus earlier expected scores; brackets give 95% equal-tailed intervals. Maximum dates refer to the global maximum within 1500–1996; boundary percentages combine the two endpoints. Decline-date intervals condition on crossing; never denotes no crossing within 1500–1996. The 0.25-point threshold date is not the first negative slope.

## Diagnostics

| model | needed_refit | divergences | max_rhat | min_ess_bulk | min_ess_tail |
| --- | --- | --- | --- | --- | --- |
| quadratic | False | 0 | 1.0024 | 1578.00 | 2929.35 |
| spline_4 | False | 0 | 1.0019 | 2219.36 | 4933.98 |
| spline_5 | False | 0 | 1.0021 | 2252.14 | 4085.89 |
| spline_6 | False | 0 | 1.0022 | 2135.14 | 3852.37 |

## Probability of a 1700–1799 decline-threshold date

| model | adjustment | p_decline_1700_1799_all_draws | p_decline_1700_1799_given_crossing | p_never_crossing |
| --- | --- | --- | --- | --- |
| quadratic | Unadjusted | 1.00 | 1.00 | 0.00 |
| quadratic | Length adjusted | 0.99 | 0.99 | 0.00 |
| spline_4 | Unadjusted | 0.99 | 1.00 | 0.01 |
| spline_4 | Length adjusted | 0.98 | 0.98 | 0.01 |
| spline_5 | Unadjusted | 0.97 | 0.99 | 0.02 |
| spline_5 | Length adjusted | 0.93 | 0.95 | 0.02 |
| spline_6 | Unadjusted | 0.99 | 0.99 | 0.00 |
| spline_6 | Length adjusted | 0.95 | 0.95 | 0.00 |
