#!/usr/bin/env python3
"""
Production smoke tests — run automatically after every deploy-production CI job.

Stricter than staging:
  - TLS certificate fully verified (no --verify=False)
  - /docs endpoint must return 404 (Swagger hidden in production)
  - All checks must pass; a single failure aborts the release

Usage:
    python tests/smoke/test_production.py \\
        --base-url https://api.medical-diagnostic.yourdomain.com
Exit codes: 0 = all checks passed, 1 = one or more checks failed.
"""
import argparse
import logging
import sys

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = "✅" if ok else "❌"
    logger.info(f"{icon}  {label}" + (f"  ({detail})" if detail else ""))
    return ok


def run(base_url: str) -> bool:
    base_url = base_url.rstrip("/")
    logger.info("=" * 60)
    logger.info(f"  Production smoke tests — {base_url}")
    logger.info("=" * 60)

    results = []

    # TLS strictly enforced (no verify=False)
    with httpx.Client(verify=True, timeout=15) as client:

        # 1. Health endpoint
        try:
            resp = client.get(f"{base_url}/health")
            data = resp.json()
            results.append(check(
                "/health returns 200 + status=healthy",
                resp.status_code == 200 and data.get("status") == "healthy",
                f"status={resp.status_code}  version={data.get('version')}",
            ))
        except Exception as exc:
            results.append(check("/health", False, str(exc)))

        # 2. Readiness probe
        try:
            resp = client.get(f"{base_url}/ready")
            results.append(check(
                "/ready returns 200",
                resp.status_code == 200,
                f"status={resp.status_code}",
            ))
        except Exception as exc:
            results.append(check("/ready", False, str(exc)))

        # 3. Prometheus metrics
        try:
            resp = client.get(f"{base_url}/metrics")
            results.append(check(
                "/metrics exposes diagnostic_requests_total",
                resp.status_code == 200 and b"diagnostic_requests_total" in resp.content,
                f"status={resp.status_code}",
            ))
        except Exception as exc:
            results.append(check("/metrics", False, str(exc)))

        # 4. Swagger UI hidden in production (ENV=production → docs_url=None)
        try:
            resp = client.get(f"{base_url}/docs")
            results.append(check(
                "/docs hidden (404) in production",
                resp.status_code == 404,
                f"status={resp.status_code}",
            ))
        except Exception as exc:
            results.append(check("/docs hidden", False, str(exc)))

        # 5. Auth enforcement
        try:
            resp = client.post(f"{base_url}/api/v2/diagnose", json={})
            results.append(check(
                "Auth enforced on /api/v2/diagnose",
                resp.status_code in (401, 422),
                f"status={resp.status_code}",
            ))
        except Exception as exc:
            results.append(check("Auth enforced", False, str(exc)))

    passed = sum(results)
    total  = len(results)
    logger.info("=" * 60)
    logger.info(f"  {'✅' if all(results) else '❌'}  {passed}/{total} production smoke tests passed")
    return all(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production smoke tests")
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    sys.exit(0 if run(args.base_url) else 1)
