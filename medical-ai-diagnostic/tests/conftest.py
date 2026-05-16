"""
Pytest configuration — runs before any test module is imported.

Sets required environment variables so that `src.api.main` can be imported
without raising KeyError on JWT_SECRET_KEY / CONSENT_SECRET / AUDIT_SALT.
"""
import os

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-not-production")
os.environ.setdefault("CONSENT_SECRET", "test-consent-secret")
os.environ.setdefault("AUDIT_SALT", "test-salt")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
