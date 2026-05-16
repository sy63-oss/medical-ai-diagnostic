#!/usr/bin/env python3
"""
Staging smoke tests — run automatically after every deploy-staging CI job.

Verifies the deployed API is responsive, auth-protected, and returns valid
Prometheus metrics. Uses --verify=False so self-signed staging certs are
accepted; production tests enforce TLS strictly.

Usage:
    python tests/smoke/test_staging.py \\
        --base-url https://staging-api.medical-diagnostic.yourdomain.com
Exit codes: 0 = all checks passed, 1 = one or more checks failed.
"""
import argparse
import logging
import sys
import time

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def wait_for_ready(client: httpx.Client, base_url: str, timeout_s: int = 120) -> bool:
    deadline = time.monotonic() + timeout_s
    attempt  = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            resp = client.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                logger.info(f"  API ready after {attempt} attempt(s)")
                return True
        except httpx.RequestError:
            pass
        logger.info(f"  Not ready yet (attempt {attempt}) — retrying in 5 s…")
        time.sleep(5)
    return False


def check_health(client: httpx.Client, base_url: str) -> bool:
    resp = client.get(f"{base_url}/health", timeout=10)
    ok   = resp.status_code == 200 and resp.json().get("status") == "healthy"
    logger.info(f"{'✅' if ok else '❌'}  /health  →  {resp.status_code}")
    if ok:
        data = resp.json()
        logger.info(f"     version={data.get('version')}  model_loaded={data.get('model_loaded')}")
    return ok


def check_readiness(client: httpx.Client, base_url: str) -> bool:
    resp = client.get(f"{base_url}/ready", timeout=10)
    ok   = resp.status_code == 200
    logger.info(f"{'✅' if ok else '❌'}  /ready   →  {resp.status_code}")
    return ok


def check_metrics(client: httpx.Client, base_url: str) -> bool:
    resp = client.get(f"{base_url}/metrics", timeout=10)
    ok   = resp.status_code == 200 and b"diagnostic_requests_total" in resp.content
    logger.info(f"{'✅' if ok else '❌'}  /metrics →  {resp.status_code}")
    return ok


def check_auth_enforced(client: httpx.Client, base_url: str) -> bool:
    # POST without a token must return 401 (or 422 for missing body fields)
    resp = client.post(f"{base_url}/api/v2/diagnose", json={}, timeout=10)
    ok   = resp.status_code in (401, 422)
    logger.info(f"{'✅' if ok else '❌'}  auth enforcement  →  {resp.status_code} (expected 401/422)")
    return ok


def run(base_url: str, max_wait: int) -> bool:
    base_url = base_url.rstrip("/")
    logger.info("=" * 60)
    logger.info(f"  Staging smoke tests — {base_url}")
    logger.info("=" * 60)

    # Staging may use self-signed certs
    with httpx.Client(verify=False) as client:
        if not wait_for_ready(client, base_url, timeout_s=max_wait):
            logger.error("API did not become ready within the timeout")
            return False

        results = [
            check_health(client, base_url),
            check_readiness(client, base_url),
            check_metrics(client, base_url),
            check_auth_enforced(client, base_url),
        ]

    passed = sum(results)
    total  = len(results)
    logger.info("=" * 60)
    logger.info(f"  {'✅' if all(results) else '❌'}  {passed}/{total} smoke tests passed")
    return all(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Staging smoke tests")
    parser.add_argument("--base-url",  required=True)
    parser.add_argument("--max-wait",  type=int, default=120,
                        help="Seconds to wait for the API to become ready (default: 120)")
    args = parser.parse_args()
    sys.exit(0 if run(args.base_url, args.max_wait) else 1)
