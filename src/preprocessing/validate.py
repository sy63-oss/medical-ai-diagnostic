"""
DVC Stage 2 — Raw Data Validation

Runs data quality checks (Great Expectations-style) on the raw parquet files
produced by the ingestion stage. Fails loudly if the data doesn't meet
schema, volume, PHI-absence, or ICD-10 validity requirements.

DVC command:
    python -m src.preprocessing.validate --input data/raw --report data/raw/validation_report.json
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# First letter of every valid ICD-10 chapter
VALID_ICD10_PREFIXES = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

REQUIRED_SYMPTOM_COLS = {
    "record_id", "symptoms", "severity_scores", "onset_days", "age_group",
    "biological_sex", "bmi_category",
}
REQUIRED_DIAGNOSIS_COLS = {"record_id", "primary_diagnosis"}

VALID_AGE_GROUPS = {"0-12", "13-17", "18-30", "31-50", "51-65", "65+"}
VALID_SEX_VALUES = {"M", "F", "other"}
VALID_BMI_CATS   = {"underweight", "normal", "overweight", "obese"}


def _check(name: str, result: bool, checks: Dict[str, bool]) -> None:
    checks[name] = result
    status = "PASS" if result else "FAIL"
    logger.info(f"  [{status}] {name}")


def validate(input_dir: Path, report_path: Path) -> bool:
    sym_path  = input_dir / "symptoms.parquet"
    diag_path = input_dir / "diagnoses.parquet"

    if not sym_path.exists() or not diag_path.exists():
        raise FileNotFoundError(f"Required parquet files missing in {input_dir}")

    sym  = pd.read_parquet(sym_path)
    diag = pd.read_parquet(diag_path)

    checks: Dict[str, bool] = {}
    logger.info("Running data validation checks…")

    # Schema
    _check("symptom_schema_valid",  REQUIRED_SYMPTOM_COLS.issubset(sym.columns),   checks)
    _check("diagnosis_schema_valid", REQUIRED_DIAGNOSIS_COLS.issubset(diag.columns), checks)

    # Volume
    _check("sufficient_records", len(sym) >= 10_000, checks)

    # Nulls
    _check("no_null_record_ids",   sym["record_id"].notna().all(),              checks)
    _check("no_null_diagnoses",    diag["primary_diagnosis"].notna().all(),      checks)

    # ICD-10 code format (first character must be a valid letter)
    valid_icd10 = diag["primary_diagnosis"].apply(
        lambda x: isinstance(x, str) and len(x) >= 3 and x[0].upper() in VALID_ICD10_PREFIXES
    )
    _check("icd10_codes_valid", bool(valid_icd10.all()), checks)

    # Categorical columns within expected domains
    if "age_group" in sym.columns:
        _check("age_groups_valid",
               set(sym["age_group"].dropna().unique()).issubset(VALID_AGE_GROUPS), checks)
    if "biological_sex" in sym.columns:
        _check("sex_values_valid",
               set(sym["biological_sex"].dropna().unique()).issubset(VALID_SEX_VALUES), checks)
    if "bmi_category" in sym.columns:
        _check("bmi_categories_valid",
               set(sym["bmi_category"].dropna().unique()).issubset(VALID_BMI_CATS), checks)

    # PHI leakage: record_ids must be integers (not names)
    _check("no_phi_in_record_ids", pd.api.types.is_integer_dtype(sym["record_id"]), checks)

    # Record-level join integrity
    sym_ids  = set(sym["record_id"])
    diag_ids = set(diag["record_id"])
    _check("record_ids_consistent", sym_ids == diag_ids, checks)

    all_passed = all(checks.values())
    failed = [k for k, v in checks.items() if not v]

    report = {
        "passed":        all_passed,
        "record_count":  len(sym),
        "checks":        checks,
        "failed_checks": failed,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if not all_passed:
        raise ValueError(f"Data validation FAILED: {failed}")

    logger.info(f"✅  All {len(checks)} checks passed ({len(sym):,} records)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVC Stage 2: data validation")
    parser.add_argument("--input",  required=True, help="Raw data directory (data/raw)")
    parser.add_argument("--report", required=True, help="Output validation report path")
    args = parser.parse_args()
    validate(Path(args.input), Path(args.report))
