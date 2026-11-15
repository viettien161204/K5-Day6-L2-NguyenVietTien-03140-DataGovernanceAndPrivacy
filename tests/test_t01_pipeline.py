import json

import numpy as np
import pandas as pd

from src import main as runner
from src.pipeline import _match_probability, _shot_list


def _run(output):
    return runner.main([
        "--track", "t01",
        "--output", str(output),
        "--offline",
        "--seed", "42",
    ])


def test_t01_pipeline_writes_reproducible_plan_and_capture_gap(tmp_path):
    output = tmp_path / "run"
    assert _run(output) == 0

    pick = pd.read_csv(output / "pick.csv")
    candidates = pd.read_csv("fixtures/t01_coverage/candidates.csv")
    assert len(pick) == 30
    assert pick.clip_id.is_unique
    assert set(pick.clip_id) <= set(candidates.clip_id)
    assert pick["rank"].tolist() == list(range(1, 31))

    coverage = pd.read_csv(output / "coverage.csv")
    c15 = coverage.set_index("cell_id").loc["C15"]
    assert c15.status == "no_candidate_in_pool"
    assert set(coverage.cell_id) == {"C13", "C15"}
    coverage_details = pd.read_csv(output / "coverage_details.csv")
    assert len(coverage_details) == 15
    rebuilt_shot_list = _shot_list(coverage_details, candidates)
    assert set(rebuilt_shot_list.cell_id) == {"C09", "C13", "C15"}
    assert "C15" in set(pd.read_csv(output / "capture_plan.csv").cell_id)
    shot_list = pd.read_csv(output / "shot_list.csv")
    assert set(shot_list.cell_id) == {"C09", "C13", "C15"}
    assert shot_list.priority_rank.tolist() == [1, 2, 3]
    assert shot_list.loc[shot_list.cell_id == "C15", "action"].item() == "capture_new_clip"
    assert shot_list.loc[shot_list.cell_id == "C09", "action"].item() == "capture_redundant_backup"
    budget_curve = pd.read_csv(output / "budget_curve.csv")
    assert budget_curve.k.tolist() == [10, 20, 30]
    assert budget_curve.estimated_weighted_coverage.is_monotonic_increasing
    decision = json.loads((output / "decision.json").read_text())
    assert decision["status"] == "REWORK"
    assert decision["reason_code"] == "CANDIDATE_POOL_GAP"
    assert decision["owner"]
    planning_metrics = json.loads((output / "planning_metrics.json").read_text())
    assert "not an official" in planning_metrics["metric_status"]
    assert planning_metrics["scene_diversity"]["duplicate_selected_clips"] == 5
    assert planning_metrics["scene_diversity"]["max_clips_per_combination"] == 3
    assert planning_metrics["greedy_poc"]["estimated_weighted_coverage"] > \
        planning_metrics["random_baseline"]["estimated_weighted_coverage"]
    random_baseline = candidates.assign(
        score=np.random.default_rng(42).random(len(candidates))
    ).sort_values("score", ascending=False, kind="mergesort").head(30)
    assert planning_metrics["random_baseline"]["selected_clip_ids"] == random_baseline.clip_id.tolist()
    assert (output / "contact_sheet.html").exists()
    assert (output / "coverage_plot.png").exists()
    reviewed_template = pd.read_csv(output / "review_queue_reviewed.csv")
    assert {"reviewer", "reviewer_decision", "override_value", "review_note"} <= set(reviewed_template.columns)
    assert reviewed_template.reviewer_decision.fillna("").eq("").all()
    assert {"C13", "C15"} <= set(reviewed_template.item_id)

    second_output = tmp_path / "repeat"
    assert _run(second_output) == 0
    second_pick = pd.read_csv(second_output / "pick.csv")
    assert pick.clip_id.tolist() == second_pick.clip_id.tolist()


def test_t01_review_edit_replaces_clip_and_is_audited(tmp_path):
    output = tmp_path / "run"
    assert _run(output) == 0
    queue_path = output / "review_queue.csv"
    queue = pd.read_csv(queue_path, dtype=str).fillna("")
    candidates = pd.read_csv("fixtures/t01_coverage/candidates.csv", dtype={"clip_id": str})
    original_pick = pd.read_csv(output / "pick.csv")
    original = queue.loc[0, "item_id"]
    replacement = next(
        clip_id for clip_id in candidates.clip_id
        if clip_id not in set(original_pick.clip_id)
    )
    queue.loc[0, ["reviewer", "reviewer_decision", "override_value", "review_note"]] = [
        "reviewer-1", "edit", replacement, "replace after review",
    ]
    queue.to_csv(output / "review_queue_reviewed.csv", index=False)

    assert _run(output) == 0
    final_pick = pd.read_csv(output / "pick.csv")
    assert len(final_pick) == 30
    assert final_pick.clip_id.is_unique
    assert replacement in set(final_pick.clip_id)
    assert original not in set(final_pick.clip_id)
    events = [json.loads(line) for line in (output / "audit_log.jsonl").read_text().splitlines()]
    assert any(event["action"] == "review:edit" and event["actor"] == "reviewer-1" for event in events)


def test_t01_review_reject_refills_selection_to_k(tmp_path):
    output = tmp_path / "run"
    assert _run(output) == 0
    queue = pd.read_csv(output / "review_queue.csv", dtype=str).fillna("")
    rejected = queue.loc[0, "item_id"]
    queue.loc[0, ["reviewer", "reviewer_decision", "review_note"]] = [
        "reviewer-1", "reject", "candidate does not meet the target",
    ]
    queue.to_csv(output / "review_queue_reviewed.csv", index=False)

    assert _run(output) == 0
    final_pick = pd.read_csv(output / "pick.csv")
    assert len(final_pick) == 30
    assert final_pick.clip_id.is_unique
    assert rejected not in set(final_pick.clip_id)


def test_t01_target_gap_escalation_is_audited_and_keeps_rework(tmp_path):
    output = tmp_path / "run"
    assert _run(output) == 0
    reviewed = pd.read_csv(output / "review_queue_reviewed.csv", dtype=str).fillna("")
    c13_row = reviewed.index[reviewed.item_id == "C13"][0]
    reviewed.loc[c13_row, ["reviewer", "reviewer_decision", "override_value", "review_note"]] = [
        "reviewer-1",
        "escalate",
        "verify_metadata",
        "Verify the suggested clip metadata before treating C13 as covered.",
    ]
    target_row = reviewed.index[reviewed.item_id == "C15"][0]
    reviewed.loc[target_row, ["reviewer", "reviewer_decision", "override_value", "review_note"]] = [
        "reviewer-1",
        "escalate",
        "capture_new_clip",
        "No compatible candidate exists; capture a new clip and retain REWORK.",
    ]
    reviewed.to_csv(output / "review_queue_reviewed.csv", index=False)
    original_pick = pd.read_csv(output / "pick.csv")

    assert _run(output) == 0
    final_pick = pd.read_csv(output / "pick.csv")
    assert final_pick.clip_id.tolist() == original_pick.clip_id.tolist()
    decision = json.loads((output / "decision.json").read_text())
    assert decision["status"] == "REWORK"
    events = [json.loads(line) for line in (output / "audit_log.jsonl").read_text().splitlines()]
    assert any(
        event["action"] == "review:target_escalate"
        and event["item_id"] == "C13"
        and event["after"] == "verify_metadata"
        for event in events
    )
    assert any(
        event["action"] == "review:target_escalate"
        and event["item_id"] == "C15"
        and event["after"] == "capture_new_clip"
        for event in events
    )


def test_t01_replaying_review_decisions_does_not_duplicate_audit_events(tmp_path):
    output = tmp_path / "run"
    assert _run(output) == 0
    reviewed = pd.read_csv(output / "review_queue_reviewed.csv", dtype=str).fillna("")
    clip_rows = reviewed.item_id.str.startswith("clip_")
    reviewed.loc[clip_rows, "reviewer"] = "reviewer-1"
    reviewed.loc[clip_rows, "reviewer_decision"] = "approve"
    reviewed.loc[clip_rows, "review_note"] = "approved after review"
    c15_row = reviewed.index[reviewed.item_id == "C15"][0]
    reviewed.loc[c15_row, ["reviewer", "reviewer_decision", "override_value", "review_note"]] = [
        "reviewer-1",
        "escalate",
        "capture_new_clip",
        "No compatible candidate exists; capture a new clip.",
    ]
    c13_row = reviewed.index[reviewed.item_id == "C13"][0]
    reviewed.loc[c13_row, ["reviewer", "reviewer_decision", "override_value", "review_note"]] = [
        "reviewer-1",
        "escalate",
        "verify_metadata",
        "Verify metadata before treating C13 as covered.",
    ]
    reviewed.to_csv(output / "review_queue_reviewed.csv", index=False)

    assert _run(output) == 0
    audit_path = output / "audit_log.jsonl"
    assert len(audit_path.read_text().splitlines()) == 5

    assert _run(output) == 0
    assert len(audit_path.read_text().splitlines()) == 5


def test_t01_invalid_review_override_fails_clearly(tmp_path, capsys):
    output = tmp_path / "run"
    assert _run(output) == 0
    queue = pd.read_csv(output / "review_queue.csv", dtype=str).fillna("")
    queue.loc[0, ["reviewer", "reviewer_decision", "override_value", "review_note"]] = [
        "reviewer-1", "edit", "not-a-candidate", "invalid candidate",
    ]
    queue.to_csv(output / "review_queue_reviewed.csv", index=False)

    assert _run(output) == 2
    assert "references unknown clip_id" in capsys.readouterr().err


def test_t01_holdout_config_selects_only_holdout_candidates(tmp_path):
    output = tmp_path / "holdout"
    assert runner.main([
        "--config", "configs/lab_t01_holdout.yaml",
        "--output", str(output),
        "--offline",
        "--seed", "42",
    ]) == 0
    pick = pd.read_csv(output / "pick.csv")
    holdout = pd.read_csv("fixtures/t01_coverage/holdout_candidates.csv")
    assert len(pick) == 6
    assert pick.clip_id.is_unique
    assert set(pick.clip_id) <= set(holdout.clip_id)


def test_t01_unknown_metadata_is_uncertain_but_wildcards_are_not():
    candidate = pd.Series({
        "route": "urban",
        "weather": "unknown",
        "light": "day",
        "object": "car",
    })
    target = pd.Series({
        "route": "urban",
        "weather": "rain",
        "light": "*",
        "object": "car",
    })
    assert _match_probability(candidate, target, 0.5) == 0.5
    target["weather"] = "*"
    assert _match_probability(candidate, target, 0.5) == 1.0
    target["route"] = "alley"
    assert _match_probability(candidate, target, 0.5) == 0.0
