"""
Old-vs-new transfer metric deltas around the Manavgat Step8A repair.

Compares the pre-repair four-AOI decomposition (preserved under
`outputs/diagnostics/advisor_followup_provenance/pre_repair_baseline/`) against
the regenerated one, per direction x model family x adaptation method x metric.

Rules
-----
* If the old artefact was not preserved, the row is emitted with
  `comparison_available = false` and `unavailable_reason =
  old_artifact_not_preserved`. A value is NEVER invented.
* Brier is only emitted when both artefacts carry it under the same protocol.
  It is not preregistered for the Step10 adapted models, so in practice it is
  reported as unavailable rather than computed.
* The delta table is a provenance record, not evidence of effect size. It does
  not license any claim that the single corrupted elevation cell had zero
  scientific impact -- that claim requires reading these deltas.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.paths import PROJECT_ROOT

SCHEMA_VERSION = "advisor_followup.old_new_manavgat_metric_deltas.v1"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "advisor_followup_provenance"
BASELINE_ROOT = OUTPUT_ROOT / "pre_repair_baseline"
DECOMPOSITION_ROOT = PROJECT_ROOT / "outputs" / "diagnostics" / "four_aoi_transfer_decomposition"

OLD_DECOMPOSITION = BASELINE_ROOT / "OLD_four_aoi_decomposition.csv"

KEYS = ["source_experiment_id", "target_experiment_id", "model_family",
        "adaptation_method", "metric"]

VALUE_COLUMNS = {
    "within_target_auc": "within_target_auc",
    "raw_auc": "raw_auc",
    "adapted_auc": "adapted_auc",
    "adaptation_effect": "adapted_minus_raw",
    "remaining_gap": "remaining_gap",
    "recovered_fraction": "recovered_fraction",
}

UNAVAILABLE_OLD = "old_artifact_not_preserved"
UNAVAILABLE_ROW = "row_absent_from_old_artifact"
UNAVAILABLE_BRIER = "brier_not_preregistered_for_step10_adapted_models"


class OldNewDeltaError(SystemExit):
    """Fatal, contract-violating condition."""


def _canonical_set_id(aois: list[str]) -> str:
    return "__".join(sorted(aois))


def _new_decomposition_path(aois: list[str]) -> Path:
    return DECOMPOSITION_ROOT / _canonical_set_id(aois) / "four_aoi_decomposition.csv"


def _input_hash(payload_path: Path, key: str) -> str | None:
    if not payload_path.is_file():
        return None
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for entry in payload.get("entries", []):
        if entry.get("path", "").endswith(key):
            return entry.get("sha256")
    return None


def build(aois: list[str], write: bool = True) -> dict[str, Any]:
    new_path = _new_decomposition_path(aois)
    if not new_path.is_file():
        raise OldNewDeltaError(
            f"regenerated decomposition not found: {new_path}. Run the "
            "transfer-decomposition step first."
        )
    new = pd.read_csv(new_path)

    old_available = OLD_DECOMPOSITION.is_file()
    old = pd.read_csv(OLD_DECOMPOSITION) if old_available else None

    old_hash = _input_hash(
        OUTPUT_ROOT / "pre_rerun_snapshot" / "pre_rerun_manifest.json",
        "manavgat_2021/step8a/step8a_500m_modeling_dataset.parquet",
    )
    new_hash = _input_hash(
        OUTPUT_ROOT / "frozen_hash_inventory_after.json",
        "manavgat_2021/step8a/step8a_500m_modeling_dataset.parquet",
    ) or _input_hash(
        OUTPUT_ROOT / "frozen_hash_inventory_before.json",
        "manavgat_2021/step8a/step8a_500m_modeling_dataset.parquet",
    )

    rows: list[dict[str, Any]] = []
    old_index = None
    if old_available:
        old_index = old.set_index(KEYS)

    for _, record in new.iterrows():
        key = tuple(record[k] for k in KEYS)
        for source_column, metric_name in VALUE_COLUMNS.items():
            new_value = record.get(source_column)
            entry: dict[str, Any] = {
                "source_experiment_id": record["source_experiment_id"],
                "target_experiment_id": record["target_experiment_id"],
                "model_family": record["model_family"],
                "adaptation_method": record["adaptation_method"],
                "metric": record["metric"],
                "quantity": metric_name,
                "old_value": None,
                "new_value": None if pd.isna(new_value) else float(new_value),
                "delta": None,
                "absolute_delta": None,
                "old_input_hash": old_hash,
                "new_input_hash": new_hash,
                "comparison_available": False,
                "unavailable_reason": UNAVAILABLE_OLD,
            }
            if old_available:
                try:
                    old_record = old_index.loc[key]
                except KeyError:
                    entry["unavailable_reason"] = UNAVAILABLE_ROW
                    rows.append(entry)
                    continue
                if isinstance(old_record, pd.DataFrame):
                    old_record = old_record.iloc[0]
                old_value = old_record.get(source_column)
                if pd.isna(old_value) or pd.isna(new_value):
                    entry["old_value"] = None if pd.isna(old_value) else float(old_value)
                    entry["unavailable_reason"] = "value_missing_in_one_artifact"
                else:
                    entry["old_value"] = float(old_value)
                    entry["delta"] = float(new_value) - float(old_value)
                    entry["absolute_delta"] = abs(entry["delta"])
                    entry["comparison_available"] = True
                    entry["unavailable_reason"] = None
            rows.append(entry)

        # Brier: only if both artefacts carry it under the same protocol.
        rows.append({
            "source_experiment_id": record["source_experiment_id"],
            "target_experiment_id": record["target_experiment_id"],
            "model_family": record["model_family"],
            "adaptation_method": record["adaptation_method"],
            "metric": record["metric"],
            "quantity": "brier",
            "old_value": None, "new_value": None, "delta": None, "absolute_delta": None,
            "old_input_hash": old_hash, "new_input_hash": new_hash,
            "comparison_available": False,
            "unavailable_reason": UNAVAILABLE_BRIER,
        })

    table = pd.DataFrame(rows)
    comparable = table[table["comparison_available"]]
    manavgat_rows = comparable[
        (comparable["source_experiment_id"] == "manavgat_2021")
        | (comparable["target_experiment_id"] == "manavgat_2021")
    ]
    other_rows = comparable[
        (comparable["source_experiment_id"] != "manavgat_2021")
        & (comparable["target_experiment_id"] != "manavgat_2021")
    ]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "canonical_set_id": _canonical_set_id(aois),
        "old_artifact_available": old_available,
        "old_artifact_path": str(OLD_DECOMPOSITION.relative_to(PROJECT_ROOT)) if old_available else None,
        "new_artifact_path": str(new_path.relative_to(PROJECT_ROOT)),
        "old_manavgat_step8a_hash": old_hash,
        "new_manavgat_step8a_hash": new_hash,
        "row_count": int(len(table)),
        "comparable_row_count": int(len(comparable)),
        "manavgat_involving": {
            "rows": int(len(manavgat_rows)),
            "max_absolute_delta": float(manavgat_rows["absolute_delta"].max()) if len(manavgat_rows) else None,
            "mean_absolute_delta": float(manavgat_rows["absolute_delta"].mean()) if len(manavgat_rows) else None,
        },
        "manavgat_free": {
            "rows": int(len(other_rows)),
            "max_absolute_delta": float(other_rows["absolute_delta"].max()) if len(other_rows) else None,
            "note": (
                "Directions not involving Manavgat should be numerically identical; "
                "any non-zero delta here indicates an unintended change."
            ),
        },
        "interpretation_limits": [
            "This table is a provenance record, not an effect-size claim.",
            "Do not describe the scientific impact of the corrupted elevation cell "
            "as zero without reading these deltas.",
            "No significance testing is performed or implied.",
        ],
    }

    if write:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUTPUT_ROOT / "old_new_manavgat_metric_deltas.csv", index=False)
        (OUTPUT_ROOT / "old_new_manavgat_metric_deltas.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
        (OUTPUT_ROOT / "old_new_manavgat_metric_deltas.md").write_text(
            _render_markdown(payload, table)
        )
    return payload


def _render_markdown(payload: dict[str, Any], table: pd.DataFrame) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Old vs new transfer metrics — Manavgat Step8A repair")
    add("")
    add(f"- Generated: `{payload['created_at']}`")
    add(f"- Old artefact available: **{payload['old_artifact_available']}**")
    add(f"- Old Manavgat Step8A hash: `{payload['old_manavgat_step8a_hash']}`")
    add(f"- New Manavgat Step8A hash: `{payload['new_manavgat_step8a_hash']}`")
    add(f"- Rows: {payload['row_count']} ({payload['comparable_row_count']} comparable)")
    add("")
    add("## Directions involving Manavgat")
    add("")
    mi = payload["manavgat_involving"]
    add(f"- Rows: {mi['rows']}")
    add(f"- Max |Δ|: `{mi['max_absolute_delta']}`")
    add(f"- Mean |Δ|: `{mi['mean_absolute_delta']}`")
    add("")
    add("## Directions NOT involving Manavgat")
    add("")
    mf = payload["manavgat_free"]
    add(f"- Rows: {mf['rows']}")
    add(f"- Max |Δ|: `{mf['max_absolute_delta']}`")
    add(f"- {mf['note']}")
    add("")
    add("## Largest absolute changes (Manavgat directions, ROC-AUC)")
    add("")
    focus = table[
        table["comparison_available"]
        & (table["metric"] == "roc_auc")
        & ((table["source_experiment_id"] == "manavgat_2021")
           | (table["target_experiment_id"] == "manavgat_2021"))
    ].nlargest(15, "absolute_delta")
    if focus.empty:
        add("_No comparable Manavgat ROC-AUC rows._")
    else:
        add("| Direction | Family | Method | Quantity | old | new | Δ |")
        add("|---|---|---|---|---|---|---|")
        for _, r in focus.iterrows():
            add(
                f"| {r['source_experiment_id']}→{r['target_experiment_id']} | "
                f"{r['model_family']} | {r['adaptation_method']} | {r['quantity']} | "
                f"{r['old_value']:.6f} | {r['new_value']:.6f} | {r['delta']:+.2e} |"
            )
    add("")
    add("## Interpretation limits")
    add("")
    for item in payload["interpretation_limits"]:
        add(f"- {item}")
    add("")
    return "\n".join(lines)
