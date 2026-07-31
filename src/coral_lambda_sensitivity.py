"""Frozen CORAL additive-ridge sensitivity diagnostic.

This module deliberately contains no Earth Engine dependency.  Adaptation and
prediction functions accept label-blind target feature frames; target labels
are accepted only by the metric and bootstrap functions.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from core.paths import PROJECT_ROOT
from core.step10_shared import (
    CATEGORICAL_FEATURES,
    FEATURE_LISTS,
    MODEL_FAMILIES,
    MODEL_NAME,
    PRIMARY_POPULATION,
    Step10Error,
    _sym_matrix_power,
    apply_coral,
    apply_regionwise_zscore,
    assert_label_blind,
    canonical_json,
    compute_analysis_id as _step10_analysis_id,
    compute_regionwise_zscore_stats,
    fit_coral_alignment,
    sha256_file as _step10_sha256_file,
)
from src.step8b_train_baseline_vs_thermal_model import build_pipeline
from src.step9a_audit_cross_region_inputs import TARGET_COLUMN, resolve_step8a_dataset_path
from src.step9b_run_cross_region_transfer import population_subset

SCHEMA_VERSION = "coral_lambda_sensitivity.v1"
DIAGNOSTIC_NAMESPACE = "coral_lambda_sensitivity"
DIAGNOSTIC_CLASS = "coral_regularisation_parameter_sensitivity"
STAGES = ("plan", "fit", "bootstrap", "summarize")
STAGE_REQUIRES = {"plan": (), "fit": ("plan",), "bootstrap": ("plan", "fit"),
                  "summarize": ("plan", "fit", "bootstrap")}
PRIMARY_EXPERIMENTS = ("manavgat_2021", "bejis_2022", "mugla_2021")
PRIMARY_DIRECTIONS = ("bejis_2022_to_mugla_2021", "mugla_2021_to_bejis_2022")
SECONDARY_DIRECTIONS = ("manavgat_2021_to_mugla_2021", "mugla_2021_to_manavgat_2021")
DIRECTIONS = PRIMARY_DIRECTIONS + SECONDARY_DIRECTIONS
CONTEXTUAL_ONLY_DIRECTIONS = ("manavgat_2021_to_bejis_2022", "bejis_2022_to_manavgat_2021")
LAMBDA_GRID = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
LAMBDA_TOKEN_SEQUENCE = ("lambda_0", "lambda_1e_m8", "lambda_1e_m7", "lambda_1e_m6",
                         "lambda_1e_m5", "lambda_1e_m4", "lambda_1e_m3",
                         "lambda_1e_m2", "lambda_1e_m1")
LAMBDA_TOKENS = dict(zip(LAMBDA_GRID, LAMBDA_TOKEN_SEQUENCE))
TOKEN_TO_LAMBDA = dict(zip(LAMBDA_TOKEN_SEQUENCE, LAMBDA_GRID))
CANONICAL_LAMBDA = 1e-5
CANONICAL_LAMBDA_INDEX = 4
EIGENVALUE_FLOOR = 1e-12
METRICS = ("roc_auc", "pr_auc", "brier_score")
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 42
EXPECTED_SCIENTIFIC_FITS = 72
CANONICAL_GATE_CHECKS = 8
TIER1_TOLERANCE = 1e-12
TIER2_PROBABILITY_TOLERANCE = 1e-12
TIER2_ROC_BASE_TOLERANCE = 1e-6
TIER2_PR_TOLERANCE = 1e-6
TIER2_BRIER_TOLERANCE = 1e-12
ALLOWED_NUMERICAL_STATUSES = (
    "pass", "singular_unregularised_covariance", "eigenvalue_floor_required",
    "nonfinite_matrix_transform", "nonfinite_transformed_features", "model_fit_failure",
)
CANONICAL_STEP8A_SHA256 = {
    "manavgat_2021": "054a1961fc0582a33d36413263668b63074b21ae8b03d12269b6e228787f3439",
    "bejis_2022": "3dec785a7d8e31db2d67ed283546bbfbca1559f56df46663488d0afc24d9e393",
    "mugla_2021": "c4ab107db2207f9f20775ccc0b3bf39381173fd07d4e82f6821ce7f40be7db8e",
}


class CoralLambdaSensitivityError(SystemExit):
    pass


def lambda_token(value: float) -> str:
    try:
        return LAMBDA_TOKENS[float(value)]
    except KeyError as exc:
        raise CoralLambdaSensitivityError(f"lambda grid disinda: {value!r}") from exc


def lambda_value(token: str) -> float:
    try:
        return TOKEN_TO_LAMBDA[token]
    except KeyError as exc:
        raise CoralLambdaSensitivityError(f"bilinmeyen lambda token: {token!r}") from exc


def numeric_features(model_family: str) -> list[str]:
    if model_family not in MODEL_FAMILIES:
        raise CoralLambdaSensitivityError(f"bilinmeyen model family: {model_family}")
    result = [c for c in FEATURE_LISTS[model_family] if c not in CATEGORICAL_FEATURES]
    expected = 3 if model_family == "baseline" else 9
    if len(FEATURE_LISTS[model_family]) != (4 if model_family == "baseline" else 10) or len(result) != expected:
        raise CoralLambdaSensitivityError("feature cardinality/order contract drift")
    return result


def split_direction(direction: str) -> tuple[str, str]:
    if direction not in DIRECTIONS and direction not in CONTEXTUAL_ONLY_DIRECTIONS:
        raise CoralLambdaSensitivityError(f"direction scope disinda: {direction}")
    source, target = direction.split("_to_", 1)
    return source, target


def scientific_config() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "diagnostic_class": DIAGNOSTIC_CLASS,
        "population": PRIMARY_POPULATION, "experiments": list(PRIMARY_EXPERIMENTS),
        "directions": list(DIRECTIONS), "contextual_only_not_rerun": list(CONTEXTUAL_ONLY_DIRECTIONS),
        "model_families": list(MODEL_FAMILIES), "lambda_grid": list(LAMBDA_GRID),
        "lambda_tokens": list(LAMBDA_TOKEN_SEQUENCE), "canonical_lambda": CANONICAL_LAMBDA,
        "canonical_lambda_index": CANONICAL_LAMBDA_INDEX,
        "numeric_feature_order": {f: numeric_features(f) for f in MODEL_FAMILIES},
        "lambda_semantics": "additive ridge lambda*I on source and target covariance",
        "covariance": {"rowvar": False, "ddof": 0}, "target_transformed": False,
        "reproduction_gate": {"tier1": TIER1_TOLERANCE, "tier2_probability": TIER2_PROBABILITY_TOLERANCE,
                              "tier2_roc_base": TIER2_ROC_BASE_TOLERANCE,
                              "tier2_pr": TIER2_PR_TOLERANCE, "tier2_brier": TIER2_BRIER_TOLERANCE},
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "paired": True, "model_refit": False},
        "sensitivity_thresholds": {"roc_auc": [0.005, 0.020], "pr_auc": [0.005, 0.020],
                                   "brier_score": [0.001, 0.005]},
        "expected_scientific_fits": EXPECTED_SCIENTIFIC_FITS, "lambda_selection_performed": False,
    }


def compute_analysis_id(config: Mapping[str, Any] | None = None) -> str:
    return _step10_analysis_id(dict(config or scientific_config()))


def diagnostics_root(output_root: Path | str | None = None) -> Path:
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
    return root / "diagnostics" / DIAGNOSTIC_NAMESPACE


def analysis_root(output_root: Path | str | None = None, analysis_id: str | None = None) -> Path:
    return diagnostics_root(output_root) / (analysis_id or compute_analysis_id())


def assert_inside_namespace(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise CoralLambdaSensitivityError(f"namespace disi write reddedildi: {path}") from exc


def _atomic_write_bytes(path: Path, payload: bytes, root: Path) -> None:
    assert_inside_namespace(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: Path, value: Any, root: Path) -> None:
    _atomic_write_bytes(path, (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode(), root)


def atomic_write_csv(path: Path, frame: pd.DataFrame, root: Path) -> None:
    _atomic_write_bytes(path, frame.to_csv(index=False).encode(), root)


def atomic_write_parquet(path: Path, frame: pd.DataFrame, root: Path) -> None:
    assert_inside_namespace(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(name, index=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def sha256_file(path: Path | str) -> str | None:
    return _step10_sha256_file(Path(path))


def assert_canonical_step8a_hashes(paths: Mapping[str, Path | str]) -> dict[str, dict[str, Any]]:
    if set(paths) != set(PRIMARY_EXPERIMENTS):
        raise CoralLambdaSensitivityError("Step8A hash gate exact three AOI paths requires")
    result = {}
    for experiment in PRIMARY_EXPERIMENTS:
        path = Path(paths[experiment]); actual = sha256_file(path); expected = CANONICAL_STEP8A_SHA256[experiment]
        result[experiment] = {"path": str(path), "expected_sha256": expected, "sha256": actual, "match": actual == expected}
        if actual != expected:
            raise CoralLambdaSensitivityError(f"Step8A hash mismatch: {experiment}")
    return result


def resolve_step10_reference(source_id: str, target_id: str, output_root: Path | str | None = None) -> dict[str, Any]:
    direction = f"{source_id}_to_{target_id}"
    split_direction(direction)
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
    canonical = root / "cross_region" / f"{source_id}__{target_id}" / "step10"
    reversed_dir = root / "cross_region" / f"{target_id}__{source_id}" / "step10"
    if not canonical.is_dir():
        raise CoralLambdaSensitivityError(f"canonical S__T Step10 reference bulunamadi: {canonical}")
    def inventory(directory: Path) -> list[dict[str, Any]]:
        return [{"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                for p in sorted(directory.rglob("*")) if p.is_file()]
    return {"direction": direction, "resolution_rule": "outputs/cross_region/{S}__{T}/step10",
            "selected": {"path": str(canonical), "artifacts": inventory(canonical)},
            "rejected_duplicate": ({"path": str(reversed_dir), "reason": "rejected_duplicate",
                                    "artifacts": inventory(reversed_dir)} if reversed_dir.is_dir() else None)}


def compare_reference_probabilities(selected: pd.DataFrame, rejected: pd.DataFrame,
                                    probability_column: str = "prediction_probability") -> dict[str, Any]:
    a, b = selected[probability_column].to_numpy(float), rejected[probability_column].to_numpy(float)
    return {"same_length": len(a) == len(b), "max_abs_probability_difference":
            float(np.max(np.abs(a - b))) if len(a) == len(b) and len(a) else None}


def lambda_grid_frame() -> pd.DataFrame:
    return pd.DataFrame([{"lambda_index": i, "lambda_value": value, "lambda_token": lambda_token(value),
                          "is_canonical": i == CANONICAL_LAMBDA_INDEX, "is_unregularised": value == 0,
                          "grid_position": "canonical" if i == 4 else ("below_canonical" if i < 4 else "above_canonical")}
                         for i, value in enumerate(LAMBDA_GRID)])


def fit_identity(direction: str, model_family: str, token: str) -> tuple[str, str, str]:
    split_direction(direction); numeric_features(model_family); lambda_value(token)
    return direction, model_family, token


def expected_fit_identities() -> set[tuple[str, str, str]]:
    return {fit_identity(d, f, t) for d in DIRECTIONS for f in MODEL_FAMILIES for t in LAMBDA_TOKEN_SEQUENCE}


def zscore_pair(source_X: pd.DataFrame, target_X: pd.DataFrame, model_family: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    assert_label_blind(target_X, "coral lambda sensitivity zscore")
    features = list(FEATURE_LISTS[model_family]); nums = numeric_features(model_family)
    if list(source_X[features].columns) != features or list(target_X[features].columns) != features:
        raise CoralLambdaSensitivityError("ordered feature contract mismatch")
    ss = compute_regionwise_zscore_stats(source_X[features], nums)
    ts = compute_regionwise_zscore_stats(target_X[features], nums)
    return (apply_regionwise_zscore(source_X[features], ss, nums),
            apply_regionwise_zscore(target_X[features], ts, nums), ss, ts)


def _condition(matrix: np.ndarray) -> float:
    try: return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError: return float("inf")


def coral_cell(Xs_z: np.ndarray, Xt_z: np.ndarray, lambda_: float) -> dict[str, Any]:
    """Apply exact Step10 CORAL, but refuse a floor-requiring lambda=0 cell."""
    Xs = np.asarray(Xs_z, dtype=float); Xt = np.asarray(Xt_z, dtype=float)
    if Xs.ndim != 2 or Xt.ndim != 2 or Xs.shape[1] != Xt.shape[1]:
        raise CoralLambdaSensitivityError("source/target numeric covariance dimension mismatch")
    d = Xs.shape[1]
    Cs0 = np.atleast_2d(np.cov(Xs, rowvar=False, ddof=0)); Ct0 = np.atleast_2d(np.cov(Xt, rowvar=False, ddof=0))
    try:
        es0, et0 = np.linalg.eigvalsh(Cs0), np.linalg.eigvalsh(Ct0)
    except np.linalg.LinAlgError:
        es0 = et0 = np.array([np.nan])
    Cs, Ct = Cs0 + float(lambda_) * np.eye(d), Ct0 + float(lambda_) * np.eye(d)
    try:
        es, et = np.linalg.eigvalsh(Cs), np.linalg.eigvalsh(Ct)
    except np.linalg.LinAlgError:
        es = et = np.array([np.nan])
    floor_required = bool(not np.isfinite(es).all() or not np.isfinite(et).all() or
                          np.min(es) < EIGENVALUE_FLOOR or np.min(et) < EIGENVALUE_FLOOR)
    diag = {"source_covariance_shape": list(Cs.shape), "target_covariance_shape": list(Ct.shape),
            "min_source_eigenvalue_before_ridge": float(np.min(es0)), "min_target_eigenvalue_before_ridge": float(np.min(et0)),
            "min_source_eigenvalue_after_ridge": float(np.min(es)), "min_target_eigenvalue_after_ridge": float(np.min(et)),
            "source_condition_number_before": _condition(Cs0), "target_condition_number_before": _condition(Ct0),
            "source_condition_number_after": _condition(Cs), "target_condition_number_after": _condition(Ct),
            "eigenvalue_floor_threshold": EIGENVALUE_FLOOR, "eigenvalue_floor_required": floor_required,
            "eigenvalue_floor_applied": False, "source_square_root_finite": False,
            "source_inverse_square_root_finite": False, "target_square_root_finite": False,
            "transform_finite": False, "transformed_source_finite": False,
            "covariance_mismatch_frobenius_norm": np.nan, "max_absolute_transformed_value": np.nan,
            "probabilities_finite": False, "numerical_status": "pass"}
    if floor_required:
        diag["numerical_status"] = "singular_unregularised_covariance" if float(lambda_) == 0 else "eigenvalue_floor_required"
        return {"A": None, "Cs": Cs, "Ct": Ct, "Xs_coral": None, "Xt_model": Xt.copy(), "diagnostics": diag}
    try:
        # Exact reused Step10 implementation; explicit lambda is mandatory.
        fitted = fit_coral_alignment(Xs, Xt, lambda_=float(lambda_))
        A = fitted["A"]
        transformed = apply_coral(Xs, fitted)
        diag.update({"source_square_root_finite": bool(np.isfinite(_sym_matrix_power(Cs, .5)).all()),
                     "source_inverse_square_root_finite": bool(np.isfinite(_sym_matrix_power(Cs, -.5)).all()),
                     "target_square_root_finite": bool(np.isfinite(_sym_matrix_power(Ct, .5)).all()),
                     "transform_finite": bool(np.isfinite(A).all()),
                     "transformed_source_finite": bool(np.isfinite(transformed).all())})
        diag["covariance_mismatch_frobenius_norm"] = float(np.linalg.norm(np.cov(transformed, rowvar=False, ddof=0) - Ct0, ord="fro"))
        diag["max_absolute_transformed_value"] = float(np.max(np.abs(transformed)))
        return {"A": A, "Cs": Cs, "Ct": Ct, "Xs_coral": transformed, "Xt_model": Xt.copy(), "diagnostics": diag}
    except (Step10Error, ValueError, np.linalg.LinAlgError, FloatingPointError):
        diag["numerical_status"] = "nonfinite_matrix_transform"
        return {"A": None, "Cs": Cs, "Ct": Ct, "Xs_coral": None, "Xt_model": Xt.copy(), "diagnostics": diag}


def fit_and_predict(source_z: pd.DataFrame, target_z: pd.DataFrame, y_source: Sequence[int],
                    model_family: str, lambda_: float, random_state: int = 42,
                    pipeline_builder: Callable[..., Any] = build_pipeline) -> dict[str, Any]:
    assert_label_blind(target_z, "immediately before predict_proba")
    feats, nums = list(FEATURE_LISTS[model_family]), numeric_features(model_family)
    cell = coral_cell(source_z[nums].to_numpy(float), target_z[nums].to_numpy(float), lambda_)
    if cell["diagnostics"]["numerical_status"] != "pass":
        return {"probabilities": None, **cell}
    source_model = source_z[feats].copy(); target_model = target_z[feats].copy()
    source_model.loc[:, nums] = cell["Xs_coral"]
    # categorical columns remain bit-for-bit unchanged; target is never CORAL transformed.
    if not source_model[[c for c in feats if c not in nums]].equals(source_z[[c for c in feats if c not in nums]]):
        raise CoralLambdaSensitivityError("unchanged categorical feature drift")
    try:
        model = pipeline_builder(feats, MODEL_NAME, random_state)
        model.fit(source_model, np.asarray(y_source, dtype=int))
        prob = np.asarray(model.predict_proba(target_model)[:, 1], dtype=float)
        finite = bool(np.isfinite(prob).all()); cell["diagnostics"]["probabilities_finite"] = finite
        if not finite: cell["diagnostics"]["numerical_status"] = "nonfinite_transformed_features"
        return {"probabilities": prob if finite else None, **cell}
    except Exception as exc:  # per-cell retention is part of the contract
        cell["diagnostics"]["numerical_status"] = "model_fit_failure"
        return {"probabilities": None, "error": type(exc).__name__, **cell}


def compute_all_metrics(y_true: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int); p = np.asarray(probabilities, dtype=float)
    if not np.isfinite(p).all() or set(np.unique(y)) != {0, 1}:
        return {m: np.nan for m in METRICS}
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier_score": float(brier_score_loss(y, p))}


def metric_origin(metric: str) -> str:
    return "recomputed_from_persisted_probabilities" if metric == "brier_score" else "persisted_step10_metric"


def natural_delta(metric: str, candidate: float, reference: float) -> float:
    return float(candidate - reference)


def oriented_delta(metric: str, candidate: float, reference: float) -> float:
    return float(reference - candidate) if metric == "brier_score" else float(candidate - reference)


def rank_quantum(y_true: Sequence[int]) -> float:
    y = np.asarray(y_true); pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return float("inf") if not pos or not neg else 1.0 / (pos * neg)


def tier2_metric_tolerance(metric: str, y_true: Sequence[int]) -> float:
    if metric == "roc_auc": return max(TIER2_ROC_BASE_TOLERANCE, 8.0 * rank_quantum(y_true))
    if metric == "pr_auc": return TIER2_PR_TOLERANCE
    if metric == "brier_score": return TIER2_BRIER_TOLERANCE
    raise CoralLambdaSensitivityError(f"unknown metric: {metric}")


def run_tier1_gate(y_true: Sequence[int], persisted_probabilities: Sequence[float],
                   reference_metrics: Mapping[str, float]) -> list[dict[str, Any]]:
    reproduced = compute_all_metrics(y_true, persisted_probabilities); rows = []
    for metric in METRICS:
        # Brier's reference is, by definition, derived from this persisted vector.
        stored = reproduced[metric] if metric == "brier_score" else float(reference_metrics[metric])
        deviation = abs(reproduced[metric] - stored)
        rows.append({"tier": "tier1_exact_from_persisted", "metric": metric, "stored_value": stored,
                     "reproduced_value": reproduced[metric], "absolute_deviation": deviation,
                     "tolerance": TIER1_TOLERANCE, "reference_origin": metric_origin(metric),
                     "gate_status": "pass" if deviation <= TIER1_TOLERANCE else "fail"})
    return rows


def run_tier2_gate(y_true: Sequence[int], persisted_probabilities: Sequence[float],
                   audit_probabilities: Sequence[float]) -> list[dict[str, Any]]:
    persisted = np.asarray(persisted_probabilities, float); audit = np.asarray(audit_probabilities, float)
    max_prob = float(np.max(np.abs(persisted - audit))) if persisted.shape == audit.shape else float("inf")
    a, b = compute_all_metrics(y_true, persisted), compute_all_metrics(y_true, audit); rows = []
    for metric in METRICS:
        tol = tier2_metric_tolerance(metric, y_true); deviation = abs(a[metric] - b[metric])
        rows.append({"tier": "tier2_independent_refit", "metric": metric, "stored_value": a[metric],
                     "reproduced_value": b[metric], "absolute_deviation": deviation, "tolerance": tol,
                     "rank_quantum": rank_quantum(y_true) if metric == "roc_auc" else np.nan,
                     "max_abs_probability_deviation": max_prob, "reference_origin": metric_origin(metric),
                     "gate_status": "pass" if max_prob <= TIER2_PROBABILITY_TOLERANCE and deviation <= tol else "fail"})
    return rows


def assert_gate_passed(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows or any(row.get("gate_status") != "pass" for row in rows):
        raise CoralLambdaSensitivityError("canonical reproduction gate FAIL; grid/bootstrap blocked")


def paired_bootstrap(frame: pd.DataFrame, probability_columns: Mapping[str, str],
                     y_col: str = "burned", block_col: str = "spatial_block_id",
                     n_replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """One RNG stream and one block draw per replicate for every series."""
    rng = np.random.default_rng(seed); blocks = frame[block_col].drop_duplicates().to_numpy()
    indices = {b: np.flatnonzero(frame[block_col].to_numpy() == b) for b in blocks}
    records, invalid = [], 0
    for replicate in range(n_replicates):
        drawn = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([indices[b] for b in drawn]); y = frame.iloc[idx][y_col].to_numpy(int)
        row: dict[str, Any] = {"replicate": replicate, "draw_hash": hashlib.sha256(idx.tobytes()).hexdigest(),
                               "n_blocks_drawn": len(drawn), "valid": len(np.unique(y)) == 2}
        if not row["valid"]:
            invalid += 1; records.append(row); continue
        for name, col in probability_columns.items():
            p = frame.iloc[idx][col].to_numpy(float); scores = compute_all_metrics(y, p)
            for metric, value in scores.items(): row[f"{metric}__{name}"] = value
        records.append(row)
    return {"replicates_df": pd.DataFrame(records), "n_requested": n_replicates,
            "n_valid": n_replicates - invalid, "n_invalid": invalid, "seed": seed,
            "single_call_per_direction": True, "model_refit": False}


def bootstrap_delta_summary(candidate_values: Sequence[float], reference_values: Sequence[float],
                            metric: str, point_estimate: float | None = None,
                            numerical_failure: bool = False) -> dict[str, Any]:
    if numerical_failure:
        return {"point_estimate": np.nan, "interval_lower": np.nan, "interval_upper": np.nan,
                "valid_replicates": 0, "invalid_replicates": BOOTSTRAP_REPLICATES,
                "support_token": "unavailable_due_to_numerical_failure"}
    c, r = np.asarray(candidate_values, float), np.asarray(reference_values, float)
    delta = r - c if metric == "brier_score" else c - r; delta = delta[np.isfinite(delta)]
    lo, hi = (float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))) if len(delta) else (np.nan, np.nan)
    token = "bootstrap_supported_positive" if lo > 0 else ("bootstrap_supported_negative" if hi < 0 else "interval_includes_zero")
    return {"point_estimate": float(np.mean(delta) if point_estimate is None else point_estimate),
            "interval_lower": lo, "interval_upper": hi, "valid_replicates": len(delta),
            "invalid_replicates": BOOTSTRAP_REPLICATES - len(delta), "support_token": token}


def sensitivity_token(metric: str, maximum_absolute_deviation: float) -> str:
    low, high = (0.001, 0.005) if metric == "brier_score" else (0.005, 0.020)
    value = abs(float(maximum_absolute_deviation))
    return "insensitive_over_grid" if value <= low else ("modest_lambda_sensitivity" if value <= high else "material_lambda_sensitivity")


def verify_complete_fit_identities(identities: Sequence[tuple[str, str, str]]) -> bool:
    return len(identities) == EXPECTED_SCIENTIFIC_FITS and len(set(identities)) == len(identities) and set(identities) == expected_fit_identities()


def planned_output_layout(output_root: Path | str | None = None) -> dict[str, Any]:
    root = analysis_root(output_root)
    return {"schema_version": SCHEMA_VERSION, "analysis_id": root.name, "analysis_root": str(root),
            "stages": list(STAGES), "scientific_fit_identities": EXPECTED_SCIENTIFIC_FITS,
            "writes": ["config.json", "input_hashes.json", "repository_inventory.json", "canonical_reproduction.csv",
                       "lambda_grid.csv", "adaptation_statistics.parquet", "predictions.parquet", "metrics.csv",
                       "bootstrap_replicates.parquet", "bootstrap_summary.csv", "sensitivity_summary.csv",
                       "numerical_diagnostics.csv", "summary.json", "report.md", "manifest.json"]}


def _validate_stage_range(from_stage: str, to_stage: str) -> tuple[str, ...]:
    if from_stage not in STAGES or to_stage not in STAGES or STAGES.index(from_stage) > STAGES.index(to_stage):
        raise CoralLambdaSensitivityError("invalid stage range")
    return STAGES[STAGES.index(from_stage):STAGES.index(to_stage) + 1]


def quarantine_namespace(root: Path) -> Path | None:
    if not root.exists(): return None
    parent = root.parent / "_quarantine"; parent.mkdir(parents=True, exist_ok=True)
    suffix = hashlib.sha256(str(sorted(str(p) for p in root.rglob("*"))).encode()).hexdigest()[:12]
    destination = parent / f"{root.name}.{suffix}"
    if destination.exists(): raise CoralLambdaSensitivityError(f"quarantine destination exists: {destination}")
    shutil.move(str(root), str(destination)); return destination


def write_stage_marker(root: Path, stage: str, payload: Mapping[str, Any]) -> None:
    atomic_write_json(root / "stages" / f"{stage}.json",
                      {"schema_version": SCHEMA_VERSION, "stage": stage, "status": "pass", **payload}, root)


def stage_is_reusable(root: Path, stage: str) -> bool:
    path = root / "stages" / f"{stage}.json"
    if not path.exists(): return False
    try: marker = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError): return False
    if marker.get("status") != "pass" or marker.get("analysis_id") != root.name: return False
    if stage == "fit":
        if not verify_complete_fit_identities([tuple(x) for x in marker.get("fit_identities", [])]): return False
        partition_hashes = marker.get("prediction_partition_hashes", {})
        if len(partition_hashes) != EXPECTED_SCIENTIFIC_FITS: return False
        if any(sha256_file(root / rel) != digest for rel, digest in partition_hashes.items()): return False
    return all(sha256_file(root / rel) == digest for rel, digest in marker.get("file_hashes", {}).items())


def _require_stage(root: Path, stage: str) -> None:
    if not stage_is_reusable(root, stage):
        raise CoralLambdaSensitivityError(f"required hash-bound PASS stage unavailable: {stage}")
    config_path = root / "config.json"
    grid_path = root / "lambda_grid.csv"
    if not config_path.is_file() or not grid_path.is_file():
        raise CoralLambdaSensitivityError("plan artifacts incomplete")
    config = json.loads(config_path.read_text())
    if config.get("analysis_id") != root.name or compute_analysis_id(config.get("scientific_config", {})) != root.name:
        raise CoralLambdaSensitivityError("plan analysis_id/hash binding mismatch")
    grid = pd.read_csv(grid_path)
    if tuple(grid["lambda_token"]) != LAMBDA_TOKEN_SEQUENCE or tuple(grid["lambda_value"].astype(float)) != LAMBDA_GRID:
        raise CoralLambdaSensitivityError("partial or drifted lambda grid")


def _assert_no_unmarked_partial_fit(root: Path) -> None:
    marker = root / "stages" / "fit.json"
    scientific = ("canonical_reproduction.csv", "adaptation_statistics.parquet",
                  "numerical_diagnostics.csv", "metrics.csv", "predictions.parquet")
    present = [name for name in scientific if (root / name).exists()]
    if marker.exists() and not stage_is_reusable(root, "fit"):
        raise CoralLambdaSensitivityError("non-PASS or hash-drifted fit marker; refusing resume")
    if not marker.exists() and present:
        raise CoralLambdaSensitivityError(f"unmarked partial fit outputs present; refusing resume: {present}")


def _production_roots(output_root: Path | str | None, experiments_root: Path | str | None) -> tuple[Path, Path]:
    outputs = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
    experiments = Path(experiments_root) if experiments_root is not None else outputs / "experiments"
    return outputs, experiments


def _load_production_inputs(output_root: Path | str | None = None,
                            experiments_root: Path | str | None = None) -> dict[str, Any]:
    """Resolve all frozen production inputs without writing anywhere."""
    outputs, experiments = _production_roots(output_root, experiments_root)
    frames, paths = {}, {}
    for experiment in PRIMARY_EXPERIMENTS:
        path = resolve_step8a_dataset_path(experiment, experiments_root=experiments)
        paths[experiment] = path
    hashes = assert_canonical_step8a_hashes(paths)
    for experiment, path in paths.items():
        frames[experiment] = pd.read_parquet(path)
    references = {}
    for direction in DIRECTIONS:
        source, target = split_direction(direction)
        inventory = resolve_step10_reference(source, target, outputs)
        directory = Path(inventory["selected"]["path"])
        predictions_path, metrics_path = directory / "step10_predictions.parquet", directory / "step10_metrics.csv"
        if not predictions_path.is_file() or not metrics_path.is_file():
            raise CoralLambdaSensitivityError(f"Step10 reference incomplete: {direction}")
        references[direction] = {"inventory": inventory, "predictions": pd.read_parquet(predictions_path),
                                 "metrics": pd.read_csv(metrics_path)}
    return {"outputs_root": outputs, "experiments_root": experiments, "frames": frames,
            "step8a_hashes": hashes, "references": references}


def _model_population(frame: pd.DataFrame, *, source: bool) -> pd.DataFrame:
    """Primary model population; spatial evaluation metadata is not required."""
    result = population_subset(frame, PRIMARY_POPULATION).copy()
    required = {"cell_id", *FEATURE_LISTS["thermal"]}
    if source: required.add(TARGET_COLUMN)
    missing = required - set(result.columns)
    if missing: raise CoralLambdaSensitivityError(f"model population columns missing: {sorted(missing)}")
    if result.empty or not result["cell_id"].is_unique or result["cell_id"].isna().any():
        raise CoralLambdaSensitivityError("model population cell_id must be non-null and unique")
    return result


def _source_model_population(frame: pd.DataFrame) -> pd.DataFrame:
    return _model_population(frame, source=True)


def _target_model_population(frame: pd.DataFrame) -> pd.DataFrame:
    # Drop the label structurally before adaptation, covariance and prediction.
    return _model_population(frame, source=False).drop(columns=[TARGET_COLUMN], errors="ignore")


def resolve_target_block_mapping(target_frame: pd.DataFrame, persisted_predictions: pd.DataFrame,
                                 direction: str) -> pd.DataFrame:
    """Resolve canonical target blocks from Step10 persisted predictions."""
    target = _model_population(target_frame, source=False)
    required = {"direction", "population", "target_cell_id", "target_spatial_block_id"}
    missing = required - set(persisted_predictions.columns)
    if missing: raise CoralLambdaSensitivityError(f"persisted block mapping columns missing: {sorted(missing)}")
    rows = persisted_predictions[(persisted_predictions["direction"] == direction)
                                 & (persisted_predictions["population"] == PRIMARY_POPULATION)][
                                     ["target_cell_id", "target_spatial_block_id"]].copy()
    if rows.empty: raise CoralLambdaSensitivityError(f"persisted target block mapping empty: {direction}")
    if rows[["target_cell_id", "target_spatial_block_id"]].isna().any().any():
        raise CoralLambdaSensitivityError(f"null persisted target block metadata: {direction}")
    conflicts = rows.groupby("target_cell_id", sort=False)["target_spatial_block_id"].nunique(dropna=False)
    if (conflicts != 1).any(): raise CoralLambdaSensitivityError(f"conflicting target block mapping: {direction}")
    mapping = rows.drop_duplicates(["target_cell_id", "target_spatial_block_id"]).rename(
        columns={"target_cell_id": "cell_id", "target_spatial_block_id": "spatial_block_id"})
    if not mapping["cell_id"].is_unique: raise CoralLambdaSensitivityError(f"duplicate target block mapping: {direction}")
    target_ids, mapped_ids = set(target["cell_id"]), set(mapping["cell_id"])
    if target_ids != mapped_ids:
        raise CoralLambdaSensitivityError(
            f"target block coverage mismatch: {direction}; missing={len(target_ids - mapped_ids)}, extra={len(mapped_ids - target_ids)}")
    ordered = target[["cell_id"]].merge(mapping, on="cell_id", how="left", validate="one_to_one")
    if ordered["spatial_block_id"].isna().any() or len(ordered) != len(target):
        raise CoralLambdaSensitivityError(f"target block mapping join incomplete: {direction}")
    return ordered


def target_evaluation_metadata(target_frame: pd.DataFrame, block_mapping: pd.DataFrame) -> pd.DataFrame:
    """Bind target labels only after label-blind prediction has completed."""
    target = _model_population(target_frame, source=False)
    if TARGET_COLUMN not in target.columns: raise CoralLambdaSensitivityError("target evaluation label missing")
    labels = target[["cell_id", TARGET_COLUMN]].copy()
    merged = block_mapping.merge(labels, on="cell_id", how="left", validate="one_to_one")
    if merged[TARGET_COLUMN].isna().any() or len(merged) != len(target):
        raise CoralLambdaSensitivityError("target evaluation metadata coverage mismatch")
    merged[TARGET_COLUMN] = merged[TARGET_COLUMN].astype(int)
    return merged


def _assert_same_target_mapping(existing: pd.DataFrame, candidate: pd.DataFrame, target_id: str) -> None:
    left = existing.sort_values("cell_id").reset_index(drop=True)
    right = candidate.sort_values("cell_id").reset_index(drop=True)
    if not left.equals(right):
        raise CoralLambdaSensitivityError(f"cross-direction target block mapping drift: {target_id}")


def _reference_series(inputs: Mapping[str, Any], direction: str, family: str, method: str,
                      target: pd.DataFrame) -> np.ndarray:
    pred = inputs["references"][direction]["predictions"]
    subset = pred[(pred["direction"] == direction) & (pred["model_family"] == family)
                  & (pred["adaptation_method"] == method)]
    mapping = subset.set_index("target_cell_id")["prediction_probability"]
    values = target["cell_id"].map(mapping)
    if values.isna().any() or len(mapping) != len(target):
        raise CoralLambdaSensitivityError(f"reference prediction coverage mismatch: {direction}/{family}/{method}")
    return values.to_numpy(float)


def _reference_metric(inputs: Mapping[str, Any], direction: str, family: str,
                      method: str, metric: str, y: np.ndarray, probabilities: np.ndarray) -> float:
    if metric == "brier_score": return compute_all_metrics(y, probabilities)[metric]
    table = inputs["references"][direction]["metrics"]
    row = table[(table["direction"] == direction) & (table["model_family"] == family) & (table["method"] == method)]
    if len(row) != 1: raise CoralLambdaSensitivityError(f"reference metric coverage mismatch: {direction}/{family}/{method}")
    return float(row.iloc[0][metric])


def _prediction_rows(direction: str, source: str, target_id: str, family: str,
                     token: str, target_metadata: pd.DataFrame, probabilities: np.ndarray | None,
                     status: str) -> pd.DataFrame:
    probability = probabilities if probabilities is not None else np.full(len(target_metadata), np.nan)
    return pd.DataFrame({"direction": direction, "source_experiment": source,
                         "target_experiment": target_id, "population": PRIMARY_POPULATION,
                         "target_cell_id": target_metadata["cell_id"].to_numpy(),
                         "target_spatial_block_id": target_metadata["spatial_block_id"].to_numpy(),
                         "model_family": family, "lambda_token": token,
                         "prediction_probability": probability, "numerical_status": status})


def _atomic_write_partitioned_predictions(root: Path, frames: Mapping[tuple[str, str, str], pd.DataFrame]) -> None:
    destination = root / "predictions.parquet"
    staging = root / f".predictions.parquet.{os.getpid()}.tmp"
    assert_inside_namespace(staging, root)
    if staging.exists(): raise CoralLambdaSensitivityError("prediction staging collision")
    try:
        for (direction, family, token), frame in frames.items():
            leaf = staging / f"direction={direction}" / f"model_family={family}" / f"lambda_token={token}"
            leaf.mkdir(parents=True, exist_ok=True)
            # Hive keys live in the path and must not be duplicated with a
            # conflicting Arrow string encoding inside every leaf payload.
            frame.drop(columns=["direction", "model_family", "lambda_token"]).to_parquet(
                leaf / "part.parquet", index=False)
        if destination.exists(): raise CoralLambdaSensitivityError("predictions dataset already exists")
        os.replace(staging, destination)
    except Exception:
        if staging.exists(): shutil.rmtree(staging)
        raise


def run_fit_stage(root: Path, inputs: Mapping[str, Any],
                  fit_predictor: Callable[..., dict[str, Any]] = fit_and_predict) -> dict[str, Any]:
    """Gate canonical fits first, then complete the remaining frozen grid."""
    _require_stage(root, "plan")
    cache: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, dict, dict]] = {}
    target_mappings: dict[str, pd.DataFrame] = {}
    canonical_results: dict[tuple[str, str], dict[str, Any]] = {}; gate_rows = []
    # Phase 1: all eight canonical audits. Nothing is written before all pass.
    for direction in DIRECTIONS:
        source_id, target_id = split_direction(direction)
        source = _source_model_population(inputs["frames"][source_id])
        target_X = _target_model_population(inputs["frames"][target_id]); assert_label_blind(target_X, "fit target")
        block_mapping = resolve_target_block_mapping(inputs["frames"][target_id],
            inputs["references"][direction]["predictions"], direction)
        if target_id in target_mappings: _assert_same_target_mapping(target_mappings[target_id], block_mapping, target_id)
        else: target_mappings[target_id] = block_mapping
        for family in MODEL_FAMILIES:
            source_z, target_z, ss, ts = zscore_pair(source[list(FEATURE_LISTS[family])], target_X, family)
            y_source = source[TARGET_COLUMN].to_numpy(int)
            result = fit_predictor(source_z, target_z, y_source, family, CANONICAL_LAMBDA)
            if result.get("probabilities") is None:
                raise CoralLambdaSensitivityError(f"canonical audit fit failed: {direction}/{family}")
            evaluation = target_evaluation_metadata(inputs["frames"][target_id], block_mapping)
            y_target = evaluation[TARGET_COLUMN].to_numpy(int)
            persisted = _reference_series(inputs, direction, family, "coral_after_regionwise_zscore", evaluation)
            refs = {m: _reference_metric(inputs, direction, family, "coral_after_regionwise_zscore", m, y_target, persisted)
                    for m in ("roc_auc", "pr_auc")}
            rows = run_tier1_gate(y_target, persisted, refs) + run_tier2_gate(y_target, persisted, result["probabilities"])
            for row in rows: row.update({"direction": direction, "model_family": family})
            gate_rows.extend(rows); canonical_results[(direction, family)] = result
            cache[(direction, family)] = (source_z, target_z, y_source, evaluation, ss, ts)
    assert_gate_passed(gate_rows)

    predictions: dict[tuple[str, str, str], pd.DataFrame] = {}; adaptation, diagnostics, metrics = [], [], []
    identities: list[tuple[str, str, str]] = []
    for direction in DIRECTIONS:
        source_id, target_id = split_direction(direction)
        for family in MODEL_FAMILIES:
            source_z, target_z, y_source, evaluation, ss, ts = cache[(direction, family)]
            y_target = evaluation[TARGET_COLUMN].to_numpy(int)
            refs_prob = {method: _reference_series(inputs, direction, family, method, evaluation)
                         for method in ("raw_source_only", "regionwise_zscore", "coral_after_regionwise_zscore")}
            refs = {metric: {method: _reference_metric(inputs, direction, family, method, metric, y_target, prob)
                             for method, prob in refs_prob.items()} for metric in METRICS}
            for value, token in zip(LAMBDA_GRID, LAMBDA_TOKEN_SEQUENCE):
                result = canonical_results[(direction, family)] if value == CANONICAL_LAMBDA else fit_predictor(source_z, target_z, y_source, family, value)
                identity = fit_identity(direction, family, token); identities.append(identity)
                status = result["diagnostics"]["numerical_status"]
                probabilities = result.get("probabilities")
                predictions[identity] = _prediction_rows(direction, source_id, target_id, family, token, evaluation, probabilities, status)
                diag = {"direction": direction, "model_family": family, "lambda_value": value,
                        "lambda_token": token, **result["diagnostics"]}; diagnostics.append(diag)
                adaptation.append({"direction": direction, "source_experiment": source_id,
                                   "target_experiment": target_id, "model_family": family,
                                   "lambda_value": value, "lambda_token": token,
                                   "numeric_feature_order": json.dumps(numeric_features(family)),
                                   "numeric_dimension": len(numeric_features(family)),
                                   "n_source_rows": len(source_z), "n_target_rows": len(target_z),
                                   "source_zscore_stats": json.dumps(ss, sort_keys=True),
                                   "target_zscore_stats": json.dumps(ts, sort_keys=True)})
                candidate = compute_all_metrics(y_target, probabilities) if probabilities is not None else {m: np.nan for m in METRICS}
                for metric in METRICS:
                    row = {"direction": direction, "model_family": family, "lambda_value": value,
                           "lambda_token": token, "metric": metric, "metric_value": candidate[metric],
                           "numerical_status": status, "reference_origin": metric_origin(metric)}
                    for label, method in (("raw", "raw_source_only"), ("zscore", "regionwise_zscore"),
                                          ("canonical", "coral_after_regionwise_zscore")):
                        reference = refs[metric][method]; row[f"{label}_reference_value"] = reference
                        row[f"natural_delta_vs_{label}"] = natural_delta(metric, candidate[metric], reference)
                        row[f"oriented_delta_vs_{label}"] = oriented_delta(metric, candidate[metric], reference)
                    metrics.append(row)
    if not verify_complete_fit_identities(identities): raise CoralLambdaSensitivityError("scientific fit grid incomplete")
    # Commit outputs only after gate and full in-memory grid completion.
    atomic_write_csv(root / "canonical_reproduction.csv", pd.DataFrame(gate_rows), root)
    atomic_write_parquet(root / "adaptation_statistics.parquet", pd.DataFrame(adaptation), root)
    atomic_write_csv(root / "numerical_diagnostics.csv", pd.DataFrame(diagnostics), root)
    atomic_write_csv(root / "metrics.csv", pd.DataFrame(metrics), root)
    _atomic_write_partitioned_predictions(root, predictions)
    files = ["canonical_reproduction.csv", "adaptation_statistics.parquet", "numerical_diagnostics.csv", "metrics.csv"]
    partition_hashes = {str(path.relative_to(root)): sha256_file(path)
                        for path in sorted((root / "predictions.parquet").rglob("part.parquet"))}
    if len(partition_hashes) != EXPECTED_SCIENTIFIC_FITS:
        raise CoralLambdaSensitivityError("persisted prediction partition grid incomplete")
    write_stage_marker(root, "fit", {"analysis_id": root.name, "fit_identities": [list(x) for x in identities],
                      "expected_scientific_fits": 72, "completed_scientific_fits": 72,
                      "canonical_gate_checks": 8, "audit_fit_executions": 8,
                      "duplicate_fit_attempts": 0, "prediction_partition_hashes": partition_hashes,
                      "file_hashes": {p: sha256_file(root / p) for p in files}})
    return {"scientific_fits": 72, "audit_fit_executions": 8, "model_fit_executions": 72}


def run_bootstrap_stage(root: Path, inputs: Mapping[str, Any],
                        bootstrapper: Callable[..., dict[str, Any]] = paired_bootstrap) -> dict[str, Any]:
    _require_stage(root, "fit")
    predictions = pd.read_parquet(root / "predictions.parquet"); metric_table = pd.read_csv(root / "metrics.csv")
    replicate_frames, summaries = [], []
    for direction in DIRECTIONS:
        _, target_id = split_direction(direction)
        block_mapping = resolve_target_block_mapping(inputs["frames"][target_id],
            inputs["references"][direction]["predictions"], direction)
        target = target_evaluation_metadata(inputs["frames"][target_id], block_mapping)
        wide = target[["cell_id", "spatial_block_id", TARGET_COLUMN]].copy()
        columns = {}
        for family in MODEL_FAMILIES:
            for method in ("raw_source_only", "regionwise_zscore", "coral_after_regionwise_zscore"):
                name = f"{family}__{method}"; wide[name] = _reference_series(inputs, direction, family, method, target); columns[name] = name
            for token in LAMBDA_TOKEN_SEQUENCE:
                name = f"{family}__{token}"; subset = predictions[(predictions.direction == direction) &
                    (predictions.model_family == family) & (predictions.lambda_token == token)]
                wide[name] = target.cell_id.map(subset.set_index("target_cell_id").prediction_probability); columns[name] = name
        result = bootstrapper(wide, columns, n_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
        reps = result["replicates_df"]; reps.insert(0, "direction", direction); replicate_frames.append(reps)
        for family in MODEL_FAMILIES:
            for token in LAMBDA_TOKEN_SEQUENCE:
                metric_rows = metric_table[(metric_table.direction == direction) & (metric_table.model_family == family)
                                           & (metric_table.lambda_token == token)]
                for metric in METRICS:
                    for contrast, reference in (("raw", "raw_source_only"), ("zscore", "regionwise_zscore"),
                                                ("canonical", "coral_after_regionwise_zscore")):
                        candidate_col, reference_col = f"{metric}__{family}__{token}", f"{metric}__{family}__{reference}"
                        failed = candidate_col not in reps or reps[candidate_col].isna().all()
                        selected_metric = metric_rows[metric_rows.metric == metric]
                        point = selected_metric.iloc[0][f"oriented_delta_vs_{contrast}"] if len(selected_metric) else np.nan
                        summary = bootstrap_delta_summary(reps.get(candidate_col, []), reps.get(reference_col, []), metric,
                                                          point_estimate=point, numerical_failure=failed)
                        summaries.append({"direction": direction, "model_family": family, "lambda_token": token,
                                          "metric": metric, "contrast": f"candidate_vs_{contrast}", **summary})
    atomic_write_parquet(root / "bootstrap_replicates.parquet", pd.concat(replicate_frames, ignore_index=True), root)
    atomic_write_csv(root / "bootstrap_summary.csv", pd.DataFrame(summaries), root)
    files = ["bootstrap_replicates.parquet", "bootstrap_summary.csv"]
    write_stage_marker(root, "bootstrap", {"analysis_id": root.name, "model_refit": False, "model_fits": 0,
                      "replicates_per_direction": 1000, "seed": 42, "single_call_per_direction": True,
                      "file_hashes": {p: sha256_file(root / p) for p in files}})
    return {"model_fits": 0, "replicates": 4000}


def run_summarize_stage(root: Path) -> dict[str, Any]:
    _require_stage(root, "bootstrap")
    metrics = pd.read_csv(root / "metrics.csv"); diagnostics = pd.read_csv(root / "numerical_diagnostics.csv")
    rows = []
    for (direction, family, metric), group in metrics.groupby(["direction", "model_family", "metric"], sort=False):
        maximum = float(group["natural_delta_vs_canonical"].abs().max())
        failures = int((group["numerical_status"] != "pass").sum())
        rows.append({"direction": direction, "model_family": family, "metric": metric,
                     "max_absolute_deviation_from_canonical": maximum,
                     "magnitude_token": sensitivity_token(metric, maximum),
                     "numerical_instability_present": failures > 0,
                     "instability_token": "numerical_instability_present" if failures else None,
                     "n_numerical_failures": failures})
    sensitivity = pd.DataFrame(rows); atomic_write_csv(root / "sensitivity_summary.csv", sensitivity, root)
    summary = {"schema_version": SCHEMA_VERSION, "analysis_id": root.name, "earth_engine_used": False,
               "fit_accounting": {"expected_scientific_fits": 72, "completed_scientific_fits": 72,
                                  "canonical_gate_checks": 8, "audit_fit_executions": 8,
                                  "duplicate_fit_attempts": 0}, "bootstrap": {"model_refit": False,
                                  "replicates_per_direction": 1000, "seed": 42},
               "lambda_selection_performed": False, "numerical_failure_count": int((diagnostics.numerical_status != "pass").sum())}
    atomic_write_json(root / "summary.json", summary, root)
    _atomic_write_bytes(root / "report.md", ("# CORAL lambda sensitivity\n\n"
        "Frozen-grid descriptive sensitivity results. Bootstrap support tokens and numerical diagnostics are reported without lambda selection.\n").encode(), root)
    manifest_files = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"):
        manifest_files.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {"schema_version": SCHEMA_VERSION, "analysis_id": root.name,
                "logical_datasets": {"predictions": {"path": "predictions.parquet", "partitions": 72}},
                "files": manifest_files}
    atomic_write_json(root / "manifest.json", manifest, root)
    files = ["sensitivity_summary.csv", "summary.json", "report.md", "manifest.json"]
    write_stage_marker(root, "summarize", {"analysis_id": root.name, "model_fits": 0,
                      "bootstrap_runs": 0, "file_hashes": {p: sha256_file(root / p) for p in files}})
    return {"model_fits": 0, "bootstrap_runs": 0, "summary_rows": len(sensitivity)}


def run_analysis(from_stage: str = "plan", to_stage: str = "summarize", dry_run: bool = False,
                 resume: bool = False, force: bool = False, output_root: Path | str | None = None,
                 experiments_root: Path | str | None = None,
                 input_loader: Callable[..., Mapping[str, Any]] | None = None,
                 stage_runners: Mapping[str, Callable[..., dict[str, Any]]] | None = None) -> dict[str, Any]:
    selected = _validate_stage_range(from_stage, to_stage); plan = planned_output_layout(output_root)
    if dry_run:
        return {**plan, "ran": False, "dry_run": True, "selected_stages": list(selected)}
    root = Path(plan["analysis_root"])
    if root.exists() and force: quarantine_namespace(root)
    if root.exists() and not (resume or force): raise CoralLambdaSensitivityError("analysis namespace already exists; use --resume or --force")
    runners = {"fit": run_fit_stage, "bootstrap": run_bootstrap_stage, "summarize": run_summarize_stage,
               **dict(stage_runners or {})}
    results = {}
    for stage in selected:
        if resume and stage_is_reusable(root, stage):
            results[stage] = {"reused": True}; continue
        if stage == "plan":
            root.mkdir(parents=True, exist_ok=True)
            cfg = scientific_config(); atomic_write_json(root / "config.json", {"schema_version": SCHEMA_VERSION,
                              "analysis_id": root.name, "diagnostic_class": DIAGNOSTIC_CLASS, "scientific_config": cfg}, root)
            atomic_write_csv(root / "lambda_grid.csv", lambda_grid_frame(), root)
            write_stage_marker(root, "plan", {"analysis_id": root.name, "canonical_gate_status": "pending",
                                              "model_fits": 0, "bootstrap_replicates": 0})
            results[stage] = {"model_fits": 0}; continue
        for prerequisite in STAGE_REQUIRES[stage]: _require_stage(root, prerequisite)
        if stage == "fit": _assert_no_unmarked_partial_fit(root)
        if stage in ("fit", "bootstrap"):
            loader = input_loader or _load_production_inputs
            inputs = loader(output_root=output_root, experiments_root=experiments_root)
            results[stage] = runners[stage](root, inputs)
        else:
            results[stage] = runners[stage](root)
    return {**plan, "ran": True, "dry_run": False, "selected_stages": list(selected), "stage_results": results}


assert len(expected_fit_identities()) == EXPECTED_SCIENTIFIC_FITS
assert lambda_token(CANONICAL_LAMBDA) == "lambda_1e_m5"
