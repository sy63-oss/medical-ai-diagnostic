#!/usr/bin/env python3
"""
CI gate: scan Python source files for accidental PHI leakage.

Detects patterns that could indicate real patient data was committed to the
codebase (SSNs, real phone numbers, MRNs, patient name strings).
Test fixtures using clearly synthetic values (555-xxx, 123-45-6789) are exempt.

Usage:
    python scripts/check_phi_patterns.py src/ tests/
Exit codes: 0 = clean, 1 = violations found.
"""
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

# (pattern, label, scan_in_test_files)
PHI_PATTERNS: List[Tuple[str, str, bool]] = [
    # SSNs — exclude the obvious fake 123-45-6789 and invalid ranges
    (r'\b(?!000|666|9\d\d)(?!123-45)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b', "SSN", False),
    # Phone numbers with real area codes (not 555-xxx which is fictional)
    (r'\b(?!555)\d{3}[-. ]\d{3}[-. ]\d{4}\b', "Real phone number", False),
    # Medical Record Numbers
    (r'\bMRN\s*[:#]\s*\d{6,}\b', "Medical Record Number", True),
    # Hardcoded patient name patterns
    (r'\bpatient\s+name\s*[=:]\s*["\']?[A-Z][a-z]+\s+[A-Z][a-z]+', "Patient name literal", True),
    # Real dates of birth (not in regex pattern strings)
    (r'(?<!sub\()(?<!re\.)(?<!\[)\bDOB\s*[=:]\s*\d{1,2}/\d{1,2}/(?:19|20)\d{2}', "DOB literal", True),
]

# Line content that marks clearly safe/test/regex context — skip these
SAFE_MARKERS = [
    "anonymize_free_text",
    "re.sub(",
    "re.compile(",
    r"r'\\b",
    r'r"\b',
    "assert ",
    "test_removes",
    "[PHONE]",
    "[SSN]",
    "[EMAIL]",
    "[DATE]",
    "# ",
    "regex",
    "pattern",
    "PHI_PATTERN",
]


def is_safe_line(line: str) -> bool:
    stripped = line.strip()
    return any(marker in stripped for marker in SAFE_MARKERS)


def scan_file(path: Path, is_test: bool) -> List[Tuple[int, str, str]]:
    violations = []
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return violations

    for lineno, line in enumerate(content.splitlines(), start=1):
        if is_safe_line(line):
            continue
        for pattern, label, check_in_tests in PHI_PATTERNS:
            if is_test and not check_in_tests:
                continue
            if re.search(pattern, line, re.IGNORECASE):
                violations.append((lineno, label, line.strip()[:120]))
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan for accidental PHI in source code")
    parser.add_argument("dirs", nargs="+", help="Directories to scan")
    args = parser.parse_args()

    total = 0
    for dir_str in args.dirs:
        base = Path(dir_str)
        if not base.exists():
            print(f"[skip] {dir_str} — directory not found", file=sys.stderr)
            continue
        for py_file in sorted(base.rglob("*.py")):
            is_test = "test" in py_file.name or "test" in str(py_file.parent)
            hits = scan_file(py_file, is_test)
            for lineno, label, snippet in hits:
                print(f"[PHI] {py_file}:{lineno}  {label}: {snippet}")
                total += 1

    if total:
        print(f"\n❌  {total} potential PHI pattern(s) detected — review before committing.")
        sys.exit(1)

    print(f"✅  No PHI patterns detected in: {', '.join(args.dirs)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
