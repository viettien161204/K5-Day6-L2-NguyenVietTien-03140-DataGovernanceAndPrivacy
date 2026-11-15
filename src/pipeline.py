"""T01 weighted-coverage planner with an auditable human-review step."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.io import write_json
from .common.manifest import verify_manifest
from .common.review import AuditLog, load_review_decisions, make_review_queue, write_review_queue

DIMENSIONS = ("route", "weather", "light", "object")
UNKNOWN_TAGS = {"", "unknown", "unk", "na", "n/a"}
REQUIRED_CANDIDATE_COLUMNS = {"clip_id", *DIMENSIONS, "cost_min"}
REQUIRED_TARGET_COLUMNS = {"cell_id", *DIMENSIONS, "weight", "is_hard"}
RISK_ORDER = {"high": 2, "med": 1, "medium": 1, "low": 0}


def load(ctx: RunContext) -> dict[str, pd.DataFrame]:
    manifest = ctx.input_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"input provenance manifest not found: {manifest}")
    problems = verify_manifest(manifest)
    if problems:
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
        if ctx.audit:
            ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])

    candidate_file = str(ctx.param("candidate_file", "candidates.csv"))
    if candidate_file not in {"candidates.csv", "holdout_candidates.csv"}:
        raise ValueError(f"T01 candidate_file must be candidates.csv or holdout_candidates.csv, got {candidate_file!r}")
    candidates = pd.read_csv(ctx.input_dir / candidate_file, dtype={"clip_id": str})
    targets = pd.read_csv(ctx.input_dir / "targets.csv", dtype={"cell_id": str})
    _validate_inputs(candidates, targets)
    return {"candidates": candidates, "targets": targets}


def _validate_inputs(candidates: pd.DataFrame, targets: pd.DataFrame) -> None:
    missing_candidates = sorted(REQUIRED_CANDIDATE_COLUMNS - set(candidates.columns))
    missing_targets = sorted(REQUIRED_TARGET_COLUMNS - set(targets.columns))
    if missing_candidates or missing_targets:
        raise ValueError(
            f"T01 input schema invalid: candidate columns missing={missing_candidates}; "
            f"target columns missing={missing_targets}"
        )
    if candidates.empty or targets.empty:
        raise ValueError("T01 requires non-empty candidates and targets")
    candidates["clip_id"] = candidates.clip_id.fillna("").astype(str).str.strip()
    targets["cell_id"] = targets.cell_id.fillna("").astype(str).str.strip()
    if candidates.clip_id.eq("").any() or candidates.clip_id.duplicated().any():
        raise ValueError("T01 candidates must have unique, non-empty clip_id values")
    if targets.cell_id.eq("").any() or targets.cell_id.duplicated().any():
        raise ValueError("T01 targets must have unique, non-empty cell_id values")

    candidates["cost_min"] = pd.to_numeric(candidates.cost_min, errors="coerce")
    if not np.isfinite(candidates.cost_min).all() or (candidates.cost_min < 0).any():
        raise ValueError("T01 cost_min must contain non-negative numbers")
    targets["weight"] = pd.to_numeric(targets.weight, errors="coerce")
    if not np.isfinite(targets.weight).all() or (targets.weight <= 0).any():
        raise ValueError("T01 target weights must be positive numbers")
    targets["is_hard"] = pd.to_numeric(targets.is_hard, errors="coerce")
    if not targets.is_hard.isin([0, 1]).all():
        raise ValueError("T01 is_hard must contain only 0 or 1")
    for column in DIMENSIONS:
        candidates[column] = candidates[column].fillna("unknown").astype(str).str.strip()
        targets[column] = targets[column].fillna("*").astype(str).str.strip()


def _match_probability(candidate: pd.Series, target: pd.Series, unknown_probability: float) -> float:
    unknown_dimensions = 0
    for dimension in DIMENSIONS:
        wanted = target[dimension]
        if wanted == "*":
            continue
        observed = str(candidate[dimension]).strip()
        if observed.lower() in UNKNOWN_TAGS:
            unknown_dimensions += 1
        elif observed != wanted:
            return 0.0
    return unknown_probability ** unknown_dimensions


def _coverage_probabilities(
    selected_ids: list[str],
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    unknown_probability: float,
) -> np.ndarray:
    by_id = candidates.set_index("clip_id")
    coverage = np.zeros(len(targets), dtype=float)
    for clip_id in selected_ids:
        probabilities = np.array(
            [_match_probability(by_id.loc[clip_id], target, unknown_probability)
             for _, target in targets.iterrows()],
            dtype=float,
        )
        coverage = 1.0 - (1.0 - coverage) * (1.0 - probabilities)
    return coverage


def _select(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    k: int,
    unknown_probability: float,
    excluded_ids: set[str] | None = None,
    forced_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if k <= 0:
        raise ValueError("T01 k must be a positive integer")
    if not 0 < unknown_probability < 1:
        raise ValueError("T01 unknown_match_probability must be strictly between 0 and 1")
    if len(candidates) < k:
        raise ValueError(f"T01 needs at least K={k} candidates; got {len(candidates)}")

    excluded = excluded_ids or set()
    forced = forced_ids or []
    candidates = candidates.sort_values("clip_id", kind="mergesort")
    ids = set(candidates.clip_id.astype(str))
    if len(set(forced)) != len(forced) or any(item not in ids or item in excluded for item in forced):
        raise ValueError("T01 human overrides must be distinct candidate clip_ids and cannot be rejected")
    if len(forced) > k:
        raise ValueError(f"T01 has {len(forced)} human-selected clips but K={k}")

    by_id = candidates.set_index("clip_id", drop=False)
    chosen = list(forced)
    coverage = _coverage_probabilities(chosen, candidates, targets, unknown_probability)
    available = [str(item) for item in candidates.clip_id if str(item) not in excluded | set(chosen)]
    target_weights = targets.weight.to_numpy(dtype=float)
    risk_rank = candidates.set_index("clip_id")["risk"] if "risk" in candidates.columns else pd.Series(dtype=str)
    picked_rows: list[dict] = []

    for clip_id in chosen:
        row = by_id.loc[clip_id]
        picked_rows.append({
            "clip_id": clip_id,
            "marginal_weighted_gain": 0.0,
            "new_cells": "",
            "uncertain_cells": "",
            "cost_min": float(row.cost_min),
        })

    while len(chosen) < k:
        best_id = None
        best_key = None
        best_probabilities = None
        best_gain = 0.0
        for clip_id in available:
            row = by_id.loc[clip_id]
            probabilities = np.array(
                [_match_probability(row, target, unknown_probability) for _, target in targets.iterrows()],
                dtype=float,
            )
            gain = float((target_weights * (1.0 - coverage) * probabilities).sum())
            risk = str(risk_rank.get(clip_id, "")).lower()
            key = (gain, -float(row.cost_min), RISK_ORDER.get(risk, -1))
            if best_key is None or key > best_key:
                best_id, best_key = clip_id, key
                best_probabilities, best_gain = probabilities, gain
        if best_id is None:
            raise ValueError(f"T01 could not fill selection to K={k}")

        assert best_probabilities is not None
        newly_covered = [
            str(targets.iloc[i].cell_id)
            for i, probability in enumerate(best_probabilities)
            if probability > 0 and coverage[i] == 0
        ]
        uncertain = [
            str(targets.iloc[i].cell_id)
            for i, probability in enumerate(best_probabilities)
            if 0 < probability < 1
        ]
        row = by_id.loc[best_id]
        picked_rows.append({
            "clip_id": best_id,
            "marginal_weighted_gain": best_gain,
            "new_cells": ";".join(newly_covered),
            "uncertain_cells": ";".join(uncertain),
            "cost_min": float(row.cost_min),
        })
        chosen.append(best_id)
        coverage = 1.0 - (1.0 - coverage) * (1.0 - best_probabilities)
        available.remove(best_id)

    selection = pd.DataFrame(picked_rows)
    selection["rank"] = range(1, len(selection) + 1)
    return selection, coverage


def _coverage_table(
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    selected: pd.DataFrame,
    coverage: np.ndarray,
    unknown_probability: float,
) -> pd.DataFrame:
    rows = []
    by_id = candidates.set_index("clip_id")
    selected_ids = selected.clip_id.astype(str).tolist()
    for index, (_, target) in enumerate(targets.iterrows()):
        possible = []
        exact = []
        for clip_id, candidate in by_id.iterrows():
            probability = _match_probability(candidate, target, unknown_probability)
            if probability > 0:
                possible.append(str(clip_id))
                if probability == 1:
                    exact.append(str(clip_id))
        chosen_for_cell = [
            clip_id for clip_id in selected_ids
            if _match_probability(by_id.loc[clip_id], target, unknown_probability) > 0
        ]
        if coverage[index] >= 1.0 - 1e-12:
            status = "covered_by_known_tags"
        elif coverage[index] > 0:
            status = "uncertain_tag_coverage"
        elif possible:
            status = "not_selected_within_k"
        else:
            status = "no_candidate_in_pool"
        rows.append({
            "cell_id": target.cell_id,
            **{dimension: target[dimension] for dimension in DIMENSIONS},
            "weight": float(target.weight),
            "is_hard": int(target.is_hard),
            "status": status,
            "estimated_coverage_probability": float(coverage[index]),
            "exact_candidate_count": len(exact),
            "possible_candidate_count": len(possible),
            "selected_candidates": ";".join(chosen_for_cell),
            "candidate_suggestions": ";".join(exact[:5] or possible[:5]),
        })
    return pd.DataFrame(rows)


def _shot_list(
    coverage: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    candidate_costs = candidates.set_index("clip_id").cost_min.to_dict()
    gaps = coverage[coverage.status != "covered_by_known_tags"]
    for row in gaps.to_dict(orient="records"):
        suggestion_value = row["candidate_suggestions"]
        suggestions = (
            []
            if pd.isna(suggestion_value)
            else [clip_id for clip_id in str(suggestion_value).split(";") if clip_id]
        )
        suggested_clip = suggestions[0] if suggestions else ""
        estimated_cost = float(candidate_costs[suggested_clip]) if suggested_clip else np.nan
        if row["status"] == "uncertain_tag_coverage":
            action = "verify_suggested_clip_metadata"
            reason = "The selected candidate has unknown metadata for a constrained target dimension."
        elif row["status"] == "no_candidate_in_pool":
            action = "capture_new_clip"
            reason = "No compatible candidate exists in the current pool."
        else:
            action = "consider_candidate_outside_current_k"
            reason = "Compatible candidates exist, but none was selected within the current budget."
        rows.append({
            "cell_id": row["cell_id"],
            "route": row["route"],
            "weather": row["weather"],
            "light": row["light"],
            "object": row["object"],
            "weight": row["weight"],
            "is_hard": row["is_hard"],
            "status": row["status"],
            "action": action,
            "reason": reason,
            "suggested_clip_id": suggested_clip,
            "estimated_cost_min": estimated_cost,
            "cost_per_weight": estimated_cost / float(row["weight"]) if suggested_clip else np.nan,
            "cost_basis": "existing candidate metadata" if suggested_clip else "not estimable from current pool",
        })
    backup_targets = coverage[
        (coverage.is_hard == 1)
        & (coverage.status == "covered_by_known_tags")
        & (coverage.exact_candidate_count == 1)
    ].sort_values(
        ["weight", "exact_candidate_count", "cell_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    if not backup_targets.empty:
        target = backup_targets.iloc[0]
        rows.append({
            "cell_id": target.cell_id,
            "route": target.route,
            "weather": target.weather,
            "light": target.light,
            "object": target.object,
            "weight": float(target.weight),
            "is_hard": int(target.is_hard),
            "status": "single_candidate_hard_target",
            "action": "capture_redundant_backup",
            "reason": "A hard target is covered by only one exact candidate; collect a second independent clip.",
            "suggested_clip_id": "",
            "estimated_cost_min": np.nan,
            "cost_per_weight": np.nan,
            "cost_basis": "no backup candidate in current pool; new-capture cost is unknown",
        })
    shot_list = pd.DataFrame(rows)
    if shot_list.empty:
        shot_list["priority_rank"] = pd.Series(dtype=int)
        return shot_list
    action_priority = {
        "verify_suggested_clip_metadata": 0,
        "capture_new_clip": 1,
        "capture_redundant_backup": 2,
        "consider_candidate_outside_current_k": 3,
    }
    shot_list["_action_priority"] = shot_list.action.map(action_priority).fillna(4)
    shot_list = shot_list.sort_values(
        ["_action_priority", "cost_per_weight", "weight", "cell_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
        na_position="last",
    ).drop(columns="_action_priority").reset_index(drop=True)
    shot_list.insert(0, "priority_rank", range(1, len(shot_list) + 1))
    return shot_list


def _review_priority(
    selected: pd.DataFrame,
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    by_id = candidates.set_index("clip_id")
    hard_weights = targets.loc[targets.is_hard == 1].set_index("cell_id").weight.to_dict()
    weights = targets.set_index("cell_id").weight.to_dict()
    for item in selected.itertuples(index=False):
        candidate = by_id.loc[item.clip_id]
        newly = [cell for cell in item.new_cells.split(";") if cell]
        uncertain = [cell for cell in item.uncertain_cells.split(";") if cell]
        hard_gain = sum(hard_weights.get(cell, 0) for cell in newly)
        uncertainty_weight = sum(weights.get(cell, 0) for cell in uncertain)
        score = 2.0 * hard_gain + uncertainty_weight
        reason = (
            f"new target cells={','.join(newly) or 'none'}; "
            f"uncertain={','.join(uncertain) or 'none'}; "
            f"route={candidate.route}, weather={candidate.weather}, "
            f"light={candidate.light}, object={candidate.object}; cost={candidate.cost_min} min"
        )
        rows.append({
            "clip_id": item.clip_id,
            "review_score": score,
            "review_reason": reason,
            "risk_slice": str(candidate.get("risk", "unknown")),
        })
    return pd.DataFrame(rows)


def _review_template(
    queue: pd.DataFrame,
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    selected: pd.DataFrame,
    coverage: np.ndarray,
    unknown_probability: float,
) -> pd.DataFrame:
    target_coverage = _coverage_table(
        candidates, targets, selected, coverage, unknown_probability
    )
    gaps = target_coverage[target_coverage.status != "covered_by_known_tags"]
    target_rows = []
    for row in gaps.to_dict(orient="records"):
        action = (
            "verify_metadata"
            if row["status"] == "uncertain_tag_coverage"
            else "capture_new_clip"
            if row["status"] == "no_candidate_in_pool"
            else "consider_candidate_outside_current_k"
        )
        dimensions = ", ".join(f"{dimension}={row[dimension]}" for dimension in DIMENSIONS)
        target_rows.append({
            "rank": len(queue) + len(target_rows) + 1,
            "item_id": row["cell_id"],
            "score": float(row["weight"]),
            "reason": f"target gap: {row['status']}; {dimensions}",
            "suggested_action": action,
            "slice": "target_gap",
            "evidence_path": "",
            "reviewer": "",
            "reviewer_decision": "",
            "override_value": "",
            "review_note": "",
            "reviewed_at": "",
        })
    if not target_rows:
        return queue.copy()
    return pd.concat([queue, pd.DataFrame(target_rows, columns=queue.columns)], ignore_index=True)


def _log_review_once(
    audit: AuditLog,
    actor: str,
    action: str,
    item_id: str,
    before: str,
    after: str,
    reason: str,
    override_value: str,
) -> None:
    for event in audit.read():
        extra = event.get("extra") or {}
        if (
            event.get("actor") == actor
            and event.get("action") == action
            and event.get("item_id") == item_id
            and event.get("before") == before
            and event.get("after") == after
            and event.get("reason") == reason
            and extra.get("override_value", "") == override_value
        ):
            return
    audit.log(
        actor=actor,
        action=action,
        item_id=item_id,
        before=before,
        after=after,
        reason=reason,
        override_value=override_value,
    )


def _apply_review(
    ctx: RunContext,
    initial: pd.DataFrame,
    candidates: pd.DataFrame,
    targets: pd.DataFrame,
    k: int,
    unknown_probability: float,
) -> pd.DataFrame:
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if not reviewed_path.exists():
        return initial

    reviewed = load_review_decisions(reviewed_path)
    initial_ids = set(initial.clip_id.astype(str))
    candidate_ids = set(candidates.clip_id.astype(str))
    initial_target_coverage = _coverage_table(
        candidates,
        targets,
        initial,
        _coverage_probabilities(initial.clip_id.astype(str).tolist(), candidates, targets, unknown_probability),
        unknown_probability,
    ).set_index("cell_id")
    target_ids = set(targets.cell_id.astype(str))
    excluded: set[str] = set()
    forced: list[str] = []
    target_reviewed = 0
    for row in reviewed.itertuples(index=False):
        decision = row.reviewer_decision
        before = row.item_id
        after = before
        if not str(row.review_note).strip():
            raise ValueError(f"review decision for {before!r} requires a non-empty review_note")
        if before in target_ids:
            if decision != "escalate":
                raise ValueError(
                    f"target review for {before!r} only supports reviewer_decision=escalate"
                )
            target_status = str(initial_target_coverage.loc[before, "status"])
            expected_action = (
                "capture_new_clip"
                if target_status == "no_candidate_in_pool"
                else "verify_metadata"
                if target_status == "uncertain_tag_coverage"
                else "consider_candidate_outside_current_k"
            )
            if str(row.override_value).strip() != expected_action:
                raise ValueError(
                    f"target escalation for {before!r} requires "
                    f"override_value={expected_action!r}"
                )
            if ctx.audit:
                _log_review_once(
                    ctx.audit,
                    actor=row.reviewer,
                    action="review:target_escalate",
                    item_id=before,
                    before=target_status,
                    after=expected_action,
                    reason=row.review_note,
                    override_value=row.override_value,
                )
            target_reviewed += 1
            continue
        if before not in initial_ids:
            raise ValueError(f"review item {before!r} is not in the generated selection or target gaps")
        if decision == "reject":
            excluded.add(before)
        elif decision == "edit":
            replacement = str(row.override_value).strip()
            if not replacement:
                raise ValueError(f"review edit for {before!r} requires override_value=clip_id")
            if replacement not in candidate_ids:
                raise ValueError(f"review edit for {before!r} references unknown clip_id {replacement!r}")
            if replacement == before:
                raise ValueError(f"review edit for {before!r} must select a different clip_id")
            excluded.add(before)
            forced.append(replacement)
            after = replacement
        if ctx.audit:
            _log_review_once(
                ctx.audit,
                actor=row.reviewer,
                action=f"review:{decision}",
                item_id=before,
                before=before,
                after=after,
                reason=row.review_note,
                override_value=row.override_value,
            )

    if len(set(forced)) != len(forced) or excluded.intersection(forced):
        raise ValueError("human review contains conflicting or duplicate clip selections")
    final, _ = _select(
        candidates, targets, k, unknown_probability, excluded_ids=excluded, forced_ids=forced
    )
    ctx.stats["n_reviewed"] = len(reviewed)
    ctx.stats["n_target_reviews"] = target_reviewed
    return final


def run(ctx: RunContext) -> None:
    if ctx.track != "t01":
        raise NotImplementedError(f"src.pipeline implements T01 only (requested {ctx.track})")
    data = load(ctx)
    candidates, targets = data["candidates"], data["targets"]
    k = int(ctx.param("k", 30))
    unknown_probability = float(ctx.param("unknown_match_probability", 0.5))
    initial, initial_coverage = _select(candidates, targets, k, unknown_probability)
    ctx.stats["n_items"] = len(candidates)
    ctx.stats["n_targets"] = len(targets)
    review_budget = int(ctx.param("review_budget", 3))

    review_items = _review_priority(initial, candidates, targets)
    queue = make_review_queue(
        review_items,
        id_col="clip_id",
        score_col="review_score",
        budget=review_budget,
        reason_col="review_reason",
        slice_col="risk_slice",
    )
    write_review_queue(queue, ctx.out("review_queue.csv"))
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    reviewed_template = _review_template(
        queue, candidates, targets, initial, initial_coverage, unknown_probability
    )
    if not reviewed_path.exists():
        reviewed_template.to_csv(reviewed_path, index=False)
    else:
        existing_review = pd.read_csv(reviewed_path, dtype=str).fillna("")
        missing_columns = [column for column in reviewed_template.columns if column not in existing_review.columns]
        if missing_columns:
            raise ValueError(
                f"review file {reviewed_path} missing columns: {missing_columns}"
            )
        existing_ids = set(existing_review.item_id.astype(str))
        additions = reviewed_template[
            ~reviewed_template.item_id.astype(str).isin(existing_ids)
        ]
        if not additions.empty:
            existing_review = existing_review.reindex(columns=reviewed_template.columns)
            pd.concat([existing_review, additions], ignore_index=True).to_csv(
                reviewed_path, index=False
            )
    render_contact_sheet(
        queue,
        ctx.out("contact_sheet.html"),
        title="T01 coverage selection review",
        fields=("rank", "item_id", "score", "reason", "suggested_action", "slice"),
    )

    final = _apply_review(ctx, initial, candidates, targets, k, unknown_probability)
    final_coverage = _coverage_probabilities(
        final.clip_id.astype(str).tolist(), candidates, targets, unknown_probability
    )
    coverage_table = _coverage_table(
        candidates, targets, final, final_coverage, unknown_probability
    )
    total_weight = float(targets.weight.sum())
    estimated_coverage = float(
        np.dot(targets.weight.to_numpy(dtype=float), final_coverage) / total_weight
    )

    final[["clip_id", "rank"]].to_csv(ctx.out("pick.csv"), index=False)
    final.to_csv(ctx.out("selection_details.csv"), index=False)
    missing = coverage_table[coverage_table.status != "covered_by_known_tags"]
    coverage_table.to_csv(ctx.out("coverage_details.csv"), index=False)
    missing.to_csv(ctx.out("coverage.csv"), index=False)
    missing.to_csv(ctx.out("capture_plan.csv"), index=False)
    _shot_list(coverage_table, candidates).to_csv(ctx.out("shot_list.csv"), index=False)

    budget_rows = []
    for budget in sorted(set(int(value) for value in ctx.param("budgets_to_try", [10, 20, 30]))):
        if budget > len(candidates):
            continue
        plan, budget_coverage = _select(candidates, targets, budget, unknown_probability)
        budget_rows.append({
            "k": budget,
            "estimated_weighted_coverage": float(
                np.dot(targets.weight.to_numpy(dtype=float), budget_coverage) / total_weight
            ),
            "selected_cost_min": float(plan.cost_min.sum()),
        })
    pd.DataFrame(budget_rows).to_csv(ctx.out("budget_curve.csv"), index=False)

    random_baseline = candidates.assign(_random_score=ctx.rng.random(len(candidates)))
    random_baseline = random_baseline.sort_values(
        "_random_score", ascending=False, kind="mergesort"
    ).head(k)
    random_coverage = _coverage_probabilities(
        random_baseline.clip_id.astype(str).tolist(), candidates, targets, unknown_probability
    )
    hard_mask = targets.is_hard.to_numpy(dtype=int) == 1
    random_weighted_coverage = float(
        np.dot(targets.weight.to_numpy(dtype=float), random_coverage) / total_weight
    )
    selected_scenes = final[["clip_id"]].merge(
        candidates[["clip_id", *DIMENSIONS]], on="clip_id", validate="one_to_one"
    )
    scene_counts = selected_scenes.groupby(list(DIMENSIONS), dropna=False).size()
    repeated_scenes = [
        {
            **dict(zip(DIMENSIONS, scene)),
            "selected_clip_count": int(count),
        }
        for scene, count in scene_counts[scene_counts > 1].sort_index().items()
    ]
    write_json(ctx.out("planning_metrics.json"), {
        "metric_status": "metadata-only planning proxy; not an official ground-truth evaluation",
        "seed": ctx.seed,
        "k": k,
        "unknown_match_probability": unknown_probability,
        "scene_diversity": {
            "unique_metadata_combinations": int(len(scene_counts)),
            "repeated_combinations": len(repeated_scenes),
            "duplicate_selected_clips": int((scene_counts - 1).clip(lower=0).sum()),
            "max_clips_per_combination": int(scene_counts.max()),
            "repeated_scenes": repeated_scenes,
        },
        "random_baseline": {
            "method": "seeded random selection",
            "estimated_weighted_coverage": random_weighted_coverage,
            "hard_targets_covered_confidently": int((random_coverage[hard_mask] == 1.0).sum()),
            "selection_cost_min": float(random_baseline.cost_min.sum()),
            "selected_clip_ids": random_baseline.clip_id.astype(str).tolist(),
        },
        "greedy_poc": {
            "method": "greedy weighted marginal coverage",
            "estimated_weighted_coverage": estimated_coverage,
            "hard_targets_covered_confidently": int((final_coverage[hard_mask] == 1.0).sum()),
            "selection_cost_min": float(final.cost_min.sum()),
            "selected_clip_ids": final.clip_id.astype(str).tolist(),
        },
    })
    fig, (budget_ax, compare_ax) = plt.subplots(1, 2, figsize=(10, 4))
    budget_ax.plot(
        [row["k"] for row in budget_rows],
        [row["estimated_weighted_coverage"] for row in budget_rows],
        marker="o",
    )
    budget_ax.set(title="Greedy coverage by K", xlabel="K", ylabel="Estimated weighted coverage")
    budget_ax.set_ylim(0, 1)
    compare_ax.bar(
        ["Random", "Greedy PoC"],
        [random_weighted_coverage, estimated_coverage],
        color=["#8795a1", "#2f855a"],
    )
    compare_ax.set(title=f"Planning proxy at K={k}", ylabel="Estimated weighted coverage")
    compare_ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(ctx.out("coverage_plot.png"), dpi=150)
    plt.close(fig)

    hard_gaps = coverage_table[
        (coverage_table.is_hard == 1)
        & (coverage_table.estimated_coverage_probability < 1.0 - 1e-12)
    ]
    unavailable_hard = coverage_table[
        (coverage_table.is_hard == 1) & (coverage_table.status == "no_candidate_in_pool")
    ]
    unavailable_targets = coverage_table[coverage_table.status == "no_candidate_in_pool"]
    threshold = float(ctx.param("go_min_estimated_weighted_coverage", 0.8))
    if not unavailable_targets.empty:
        status = "REWORK"
        reason_code = "CANDIDATE_POOL_GAP"
        reason = f"{len(unavailable_targets)} target cells have no compatible candidate in the current pool."
    elif estimated_coverage < threshold or not hard_gaps.empty:
        status = "HOLD"
        reason_code = "COVERAGE_GAP_REVIEW_REQUIRED"
        reason = (
            f"Estimated weighted coverage {estimated_coverage:.3f}; "
            f"{len(hard_gaps)} hard target cells are not confidently covered."
        )
    else:
        status = "GO"
        reason_code = "COVERAGE_PLAN_READY"
        reason = f"Estimated weighted coverage {estimated_coverage:.3f}; all hard target cells are confidently covered."
    decision = Decision(
        status=status,
        reason_code=reason_code,
        reason=reason,
        owner=str(ctx.config.get("owner", "")).strip(),
        scope="t01",
        evidence=[
            str(ctx.out("pick.csv")),
            str(ctx.out("coverage.csv")),
            str(ctx.out("coverage_details.csv")),
            str(ctx.out("capture_plan.csv")),
            str(ctx.out("shot_list.csv")),
        ],
        metrics={
            "estimated_weighted_coverage_from_observed_tags": estimated_coverage,
            "go_threshold": threshold,
            "hard_target_gaps": len(hard_gaps),
            "hard_targets_unavailable_in_pool": len(unavailable_hard),
            "selected_k": len(final),
            "candidate_count": len(candidates),
        },
    )
    write_decision(decision, ctx.out("decision.json"), ctx.out("decision_log.csv"))
    ctx.stats.update({
        "k": k,
        "estimated_weighted_coverage_from_observed_tags": estimated_coverage,
        "estimated_covered_weight": estimated_coverage * total_weight,
        "total_target_weight": total_weight,
        "estimated_hard_target_gaps": len(hard_gaps),
        "unavailable_hard_targets": len(unavailable_hard),
        "capture_plan_cells": len(missing),
        "selected_cost_min": float(final.cost_min.sum()),
        "selection_changed_by_human_review": not final.clip_id.equals(initial.clip_id),
        "initial_estimated_weighted_coverage": float(
            np.dot(targets.weight.to_numpy(dtype=float), initial_coverage) / total_weight
        ),
    })
