# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Điền đầy đủ, cụ thể, không viết chung chung. Mọi số liệu phải trace tới artifact do script sinh ra;
> official scores/table chỉ được điền từ `reports/metrics_*.json` và `reports/table1_baseline_vs_poc.md`
> sau khi instructor GT có mặt. Planning proxy/runtime lấy từ `out/poc/` và không thay official score.
> 3–5 trang (hoặc README + report.md tương đương).

> **Trạng thái:** Bài cá nhân; PoC đã chạy và qua kiểm tra metadata synthetic. Chưa có instructor overlay/ground truth,
> nên coverage estimate dưới đây là planning proxy, không phải điểm chấm chính thức. Ba clip đã được reviewer duyệt;
> C15 được escalate để quay mới và C13 được escalate để xác minh metadata. Commit evidence và official evaluation vẫn cần hoàn tất trước khi nộp.

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | IND-03140 (bài cá nhân; xác nhận nếu giảng viên yêu cầu mã nhóm khác) |
| Challenge | T01 · Coverage Planner |
| Thành viên + vai trò | Nguyễn Viết Tiến (03140) · cá nhân: owner, implementer và reviewer; implementation được AI hỗ trợ. |
| Git URL | https://github.com/viettien161204/K5-Day6-L2-NguyenVietTien-03140-DataGovernanceAndPrivacy |
| Tag `demo-180m` (commit) | Chưa tạo |
| Tag `final-24h` (commit) | Chưa tạo |
| Deadline (giờ VN) | Nhóm bổ sung |

## 01 · Problem statement

- Ai bị ảnh hưởng / ai chịu hậu quả khi hệ thống sai? Nhóm thu thập dữ liệu và nhóm huấn luyện/đánh giá mô hình; thiếu cảnh quan trọng có thể làm hiệu năng kém trên điều kiện đó.
- Điểm nghẽn cụ thể: ngân sách thu thập clip hữu hạn, metadata có thể sai hoặc `unknown`, và một số target cell không có candidate trong pool.
- Đầu vào → đầu ra: 120 candidate clips + 15 target cells có trọng số → 30 clip được chọn, coverage theo cell và capture plan.
- Tiêu chí GO / HOLD / REWORK: GO nếu estimated weighted coverage ≥0.8 và mọi hard target cell được phủ chắc chắn; HOLD nếu chưa đạt nhưng pool còn candidate phù hợp; REWORK nếu một target không có candidate tương thích. Đây là gate dựa trên metadata, không phải metric chấm chính thức.
- Giả định của nhóm: mỗi chiều tag `unknown` bị ràng buộc được tính với xác suất khớp 0.5; các chiều và candidate được xem độc lập khi gộp coverage. Giả định này chưa được hiệu chuẩn bằng nhãn và cần domain owner xác nhận.

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| Main candidate pool | 120 clips; 15 target cells | `fixtures/t01_coverage/` | CC0 theo README của lab | Yes |
| Holdout candidate pool | 25 clips; K=6 | `fixtures/t01_coverage/holdout_candidates.csv` | CC0 theo README của lab | Yes |
| Official labels | Không có trong student checkout | Instructor overlay: `gt/verified_tags.csv`, `gt/holdout_verified_tags.csv` | Chỉ instructor dùng để chấm | Synthetic labels |

- Schema: `clip_id, route, weather, light, object, risk, cost_min, duration_s`; target schema gồm `cell_id`, 4 chiều, `weight`, `is_hard`; `*` là wildcard.
- Seed: 42 · main và holdout là hai pool riêng; holdout không dùng để tune.
- `manifest.csv` verified (sha256): có — `python -m tools.verify_manifests`; 14/14 manifest pass. `targets.csv` SHA-256: `B3D193DC1B272BE96447F2BBA1D9B0C9CB5677F02149FDE06E85D2F822E94566`.
- Không dùng dữ liệu thật; không tự tạo hay đọc ground truth.

## 03 · Research & design

- References và ngày truy cập được liệt kê ở Mục 11; không dùng tham khảo ngoài để tuyên bố hiệu quả.
- Data flow: manifest/hash check → greedy weighted marginal coverage → contact sheet/review queue → human approve/reject/edit → pick + per-cell coverage + capture plan → GO/HOLD decision.
- Thành phần offline: Python/pandas/numpy; không dùng model pretrained, API ngoài hoặc dữ liệu thật.
- Trade-offs: greedy tối ưu marginal gain theo trọng số; cost và risk chỉ dùng tie-break, không thay mục tiêu coverage. `unknown` bị giảm trọng số kỳ vọng; giả định xác suất này có thể sai nếu lỗi tag có tương quan.

## 04 · Baseline vs PoC

- Baseline: seeded random selection (`src.examples.random_baseline`), seed 42, K=30; không dùng thông tin target để tối ưu chọn clip.
- PoC: greedy weighted set-cover theo marginal gain, seed 42, K=30; với tag `unknown`, mỗi chiều bị ràng buộc được tính là xác suất khớp 0.5 nên coverage là kỳ vọng, không phải match đã xác minh. Chi phí và risk chỉ là tie-break khi marginal gain bằng nhau; thuật toán hiện không tối đa hóa gain-per-cost.
- Đối chứng Greedy Weighted Set-Cover không phạt `unknown`: đây là ablation tự nhiên để tách lợi ích của greedy khỏi chính sách bất định; nó cần chạy cùng pool/K với `unknown` được xử lý như tag đã biết hoặc theo một quy tắc không giảm trọng số. Repo hiện chưa sinh kết quả riêng cho ablation này, nên báo cáo không gán số đo cho nó và không gọi proxy PoC là bằng chứng thắng Greedy không phạt.
- Cùng budget / cùng pool / cùng seed: xác nhận cho main candidate pool; holdout pipeline K=6 đã chạy nhưng chưa có GT để chấm.

**Bảng 1 · Planning proxy, không phải official evaluation.** Weighted coverage ước lượng dùng cùng metadata đầu vào và giả định unknown=0.5; ground-truth metric chưa chạy.

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Official Weighted Coverage@K | Chưa evaluate | Chưa evaluate | Cần instructor GT; higher is better |
| Metadata-estimated weighted coverage (proxy) | 0.3750 | 0.8646 | `out/poc/planning_metrics.json`; không thay thế official score |
| Hard target cells covered confidently | 1/4 | 4/4 | Proxy theo observed tags; N=4 hard cells |
| Selection cost (minutes) | 232 | 136 | `planning_metrics.json`; candidate `cost_min` sum |
| Runtime / 100 input clips | 0.0643 s | 5.6848 s | `out/baseline/run_info.json`, `out/poc/run_info.json`; Windows CPU run |

Diễn giải: proxy hiện tại so sánh Random với PoC trên cùng pool và K; không có kết quả official vì thiếu `gt/verified_tags.csv`. Greedy không phạt `unknown` chưa được chạy như một ablation riêng. Không tạo hay điền tay `metrics_baseline.json`, `metrics_poc.json` hoặc bảng official: `src.evaluate` dừng với exit code 2 khi instructor GT vắng mặt.

### Planning proxy theo ngân sách (không phải official evaluation)

| K | Estimated weighted coverage | Chi phí chọn (phút) | Δ coverage so với budget trước | Diễn giải |
|---:|---:|---:|---:|---|
| 10 | 0.7396 | 58 | — | Chưa đạt gate proxy 0.8 |
| 20 | 0.8646 | 98 | +0.1250 | Vượt gate proxy |
| 30 | 0.8646 | 136 | +0.0000 | Thêm 38 phút nhưng không tăng coverage proxy |

Nguồn: `out/poc/budget_curve.csv`; coverage dùng metadata và giả định unknown=0.5. Không diễn giải như Weighted Coverage@K chính thức.

## 05 · Evaluation protocol

- Primary: `Σ weight(verified-covered target cells) / Σ weight(all target cells)`, đơn vị target cell, K=30; không có TP/FP/FN.
- Threshold / budget B / K: K=30; gate proxy 0.8 và unknown-match probability 0.5 khai báo trong config trước official eval.
- Target set là finite 15 cells; báo numerator/denominator có trọng số và từng cell, không tuyên bố CI quần thể. Holdout K=6 báo riêng.
- Lệnh evaluate chính xác: `python -m src.evaluate --track t01 --pred out/poc/pick.csv --out reports/metrics_poc.json`. Main chưa chấm được vì student checkout không có `gt/`; holdout pipeline đã chạy nhưng holdout score cũng cần instructor GT.

## 06 · Result slices

| Slice | N | Baseline | PoC | Ghi chú |
|---|---|---|---|---|
| Exact-tag covered cells | 15 targets | Không đo ở baseline | 13 | `out/poc/coverage_details.csv` |
| Uncertain-tag target | 1 target (C13) | Không đo ở baseline | Estimated coverage 0.5 | Chỉ có candidate gợi ý `clip_093`; cần xác minh metadata |
| No-candidate target | 1 target (C15) | Không đo ở baseline | 0 | Không có candidate tương thích; đưa vào capture plan |
| Hard targets covered confidently | 4 hard cells | 1/4 | 4/4 | Planning proxy, không phải verified labels |

### Ba failure mode bắt buộc và shot list

| Failure mode | Bằng chứng / root cause | Hành động |
|---|---|---|
| Target không thể phủ bằng pool hiện tại | C15 (`alley/rain/dusk/pedestrian`) có 0 candidate tương thích. | Quay clip mới; chưa thể ước lượng phút quay từ pool hiện tại. |
| Chọn trùng cảnh / dư thừa | `out/poc/planning_metrics.json` ghi 5 clip trùng thêm trên 4 tổ hợp route/weather/light/object; một tổ hợp được chọn 3 lần. Greedy nhắm coverage target, không có ràng buộc diversity riêng. | Trước khi thu thêm, cân nhắc giới hạn số clip mỗi tổ hợp hoặc thêm diversity objective; đánh giá lại trên dev. |
| Metadata sai nhưng không gắn `unknown` | Pipeline xem tag cụ thể là khớp chắc chắn; vì instructor GT không có trong checkout, không thể xác định clip cụ thể nào bị gán sai hoặc đo false coverage. Đây là failure risk, chưa phải lỗi đã xác minh. | Reviewer kiểm tra tag trước khi xác nhận coverage; khi có overlay, phân tích mismatch theo từng chiều. |

`out/poc/shot_list.csv` có 3 đề xuất: C13 xác minh metadata của `clip_093` (candidate metadata ghi 12 phút; 2.4 phút/weight), C15 quay clip mới (chi phí chưa biết), và C09 quay clip dự phòng vì hard target này hiện chỉ có một exact candidate. Chi phí trong pool chỉ là metadata candidate, không phải báo giá quay mới; chi phí của clip mới để NA thay vì tự ước lượng.

- Latency / compute: PoC 6.8217 s cho 120 clips (5.6848 s/100 clips); baseline 0.0772 s; holdout 1.5628 s cho 25 clips; Python 3.11.9, Windows 10, Intel64 Family 6 Model 141 Stepping 1.

## 07 · Human & decision

- Review queue: `out/poc/review_queue.csv` có 3 clip ưu tiên; `review_queue_reviewed.csv` bổ sung target-gap rows C13/C15. Contact sheet offline tại `out/poc/contact_sheet.html`.
- Override/audit: reviewer Nguyễn Viết Tiến - 03140 xác nhận approve `clip_115`/C10, `clip_108`/C09 và `clip_098`/C11, không đổi clip; C13 được escalate với override `verify_metadata`, C15 được escalate với override `capture_new_clip`. `out/poc/audit_log.jsonl` ghi 5 sự kiện tương ứng. C13 vẫn chỉ là coverage chưa xác minh cho tới khi metadata của `clip_093` được kiểm tra độc lập. Với approve, `override_value` để trống vì không có override; note ghi rõ giữ clip.
- Decision policy: sau review run vẫn trả REWORK / `CANDIDATE_POOL_GAP`; estimated coverage=0.8646, không còn hard-target gap nhưng C15 không có candidate trong pool. Owner trong config là Nguyễn Viết Tiến - 03140. Escalation được audit nhưng không tự sinh candidate hay đổi coverage.
- HOLD khi coverage dưới 0.8 hoặc còn hard target chưa phủ chắc chắn nhưng pool còn candidate; REWORK khi pool không có candidate cho một target. Đây không phải quyết định pháp lý hay bằng chứng GT.

## 08 · Reproduce & tests

```bash
make setup          # hoặc: python -m venv .venv && pip install -r requirements.txt
python -m src.main --track t01 --config configs/lab.yaml --output out/poc --run-name poc --offline --seed 42
python -m src.main --track t01 --config configs/lab.yaml --pipeline src.examples.random_baseline --output out/baseline --run-name baseline --offline --seed 42
python -m src.evaluate --track t01 --pred out/poc/pick.csv --out reports/metrics_poc.json
pytest -q
```

- Output của `pytest -q`: pass; các test cần instructor GT bị skip vì overlay không có trong student checkout. `tools.preflight`: 0 fail, 2 cảnh báo tùy chọn; manifests 14/14 pass.
- `src.evaluate` cho baseline và PoC đã được thử; cả hai exit 2 vì thiếu instructor `gt/verified_tags.csv`. Do đó official metrics JSON và comparison table chưa được sinh; không thay bằng proxy hay số liệu gõ tay.
- Máy / CPU / OS / Python: Intel64 Family 6 Model 141 Stepping 1 · Windows 10 · Python 3.11.9. Run record base git sha: `f0983d1`; thay đổi hiện tại chưa commit.

## 09 · Limitations & ethics

- Rủi ro pháp lý / PII / license: fixture synthetic, CC0 theo lab; không dùng PII thật. Không đưa dữ liệu thật lên repo/cloud.
- Điều chưa biết; false alarms; nếu scale 10× thì sao: xác suất `unknown=0.5` chưa hiệu chuẩn; metadata tag sai nhưng không ghi `unknown` có thể tạo false coverage. 10× candidates làm tăng số lượt greedy; cần benchmark lại.
- Không đưa ra kết luận tuân thủ GDPR hay luật Việt Nam; nếu triển khai ngoài lab cần domain/data owner xác nhận mục tiêu và quyền thu thập dữ liệu.
- Kế hoạch cải tiến: chạy official GT/holdout, đối chiếu greedy xử lý unknown với random, đo review effectiveness và hiệu chuẩn uncertainty chỉ trên dev.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (cần nhóm bổ sung và đối chiếu với `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Nguyễn Viết Tiến (03140) | `viettien161204 <nguyenviettien161204@gmail.com>` | Cá nhân / owner; implementation có AI hỗ trợ | Thay đổi T01 hiện chưa commit; cần kiểm tra và commit dưới author cá nhân | |

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa (không chép nguyên văn) |
|---|---|---|
| https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html | 2026-11-14 | Mô tả BDD100K và lý do đa dạng điều kiện đường/thời tiết quan trọng khi xây dữ liệu lái xe; fixture của lab vẫn là synthetic. |
| https://sites.research.google/datacardsplaybook/ | 2026-11-14 | Data Cards khuyến nghị ghi provenance, ý định sử dụng, sampling, annotation và validation; dùng làm khung cho phần provenance/limitations, không phải bằng chứng score. |

---

**Bảng 3 · Checklist chốt khi nộp**

- [ ] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [ ] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [ ] Có baseline/PoC official metrics (đang chờ instructor GT), ≥3 failure modes được phân tích, reviewer thật và audit evidence.
- [ ] Có license/source provenance, data splits, seed, offline environment, GT/eval N.
- [ ] Mỗi thành viên có ≥1 commit thực chất khớp với `git shortlog -sne --all` và được nêu trong bảng trên.
