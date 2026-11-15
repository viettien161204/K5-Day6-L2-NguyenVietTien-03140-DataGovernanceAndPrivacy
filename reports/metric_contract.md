# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T01 · Coverage Planner |
| Primary metric (tên đúng như evaluator in ra) | Weighted Coverage@K |
| Tăng tốt hay giảm tốt? | higher |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | Target cell; weight-adjusted |
| Công thức + numerator / denominator | Σ weight of target cells covered / Σ weight of all target cells |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | Candidate `clip_id` joins verified tags; all non-wildcard dimensions must match |
| Candidate pool cố định | `fixtures/t01_coverage/candidates.csv` (120 clips); separate holdout pool: `holdout_candidates.csv` (25 clips) |
| Review budget B / K | K=30 for main pool; K=6 for the separate holdout check; review queue=3 |
| Seed | 42 |
| Threshold policy (chọn trên dev như thế nào) | No learned threshold; planner uses weighted marginal coverage and a predeclared 0.5 match probability per unknown constrained dimension; unknown dimensions and candidate matches are treated as independent. GO gate: estimated weighted coverage ≥0.8 and every hard target confidently covered; HOLD if the gate is unmet but candidates exist; REWORK if any target has no compatible candidate. Freeze before official evaluation; assumptions are not calibrated probabilities. |
| Eval split (không dùng để tune) | Main: instructor `gt/verified_tags.csv`; holdout check: instructor `gt/holdout_verified_tags.csv` with K=6. Never read either in the pipeline. |
| Internal self-check set (không dùng GT) | `fixtures/t01_coverage/candidates.csv` (120 candidates) + `targets.csv` (15 target cells), seed 42. Check schema/provenance, K and unique IDs, deterministic repeat, wildcard/unknown behavior, C13 uncertainty, C15 no-candidate handling, review overrides/audit in unit tests, and K=10/20/30 planning curve. These are consistency checks, not label accuracy. |
| Freeze evidence | `targets.csv` SHA-256: `B3D193DC1B272BE96447F2BBA1D9B0C9CB5677F02149FDE06E85D2F822E94566`; keep weights and params unchanged after evaluation starts. |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | Missed weighted target cells and hard-target misses; report each target cell and the capture plan |
| Slices sẽ báo cáo (≥2) | Hard vs non-hard targets; per-target-cell coverage; budgets K=10/20/30 as exploratory analysis |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | Fixed finite target set (15 cells): report per-cell outcomes and weighted numerator/denominator; do not claim a population CI. Report holdout separately (25 candidates, K=6). |
| Ngày giờ chốt + người chốt | Draft prepared 2026-11-14; owner/reviewer: Nguyễn Viết Tiến - 03140. Three selected clips were approved; C13 escalated for metadata verification and C15 for new capture. C13 coverage remains uncertain until independently verified. |

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện.

Official metric artifacts are intentionally not fabricated: `src.evaluate` currently exits 2 because the instructor-only `gt/verified_tags.csv` overlay is absent. Generate `reports/metrics_baseline.json`, `reports/metrics_poc.json`, and the comparison table only after the instructor supplies the overlay.
