"""
Centralized Configuration Module
================================
Single source of truth for all project settings. Loads from environment
variables (.env file) with sensible defaults.

Usage:
    from config import get_config
    cfg = get_config()
    print(cfg.database.host)
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv

# ────────────────────────────────────────────────
# Load .env at import time
# ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent

for _candidate in [_PROJECT_ROOT / ".env", Path.cwd() / ".env"]:
    if _candidate.exists():
        load_dotenv(dotenv_path=_candidate)
        break
else:
    load_dotenv(find_dotenv(usecwd=True))


# ────────────────────────────────────────────────
# Configuration Dataclasses
# ────────────────────────────────────────────────

@dataclass(frozen=True)
class DatabaseConfig:
    """MySQL connection parameters."""
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "vendor_performance"
    pool_size: int = 5
    max_overflow: int = 10
    pool_recycle: int = 1800

    @property
    def connection_string(self) -> str:
        """SQLAlchemy-compatible connection URL."""
        return (
            f"mysql+mysqlconnector://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?charset=utf8mb4"
        )


@dataclass(frozen=True)
class EmailConfig:
    """Email alert settings for the pipeline."""
    enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    recipient_emails: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PathConfig:
    """Project directory paths."""
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"
    log_dir: Path = _PROJECT_ROOT / "logs"
    output_dir: Path = _PROJECT_ROOT / "output"

    def ensure_dirs(self):
        """Create directories if they don't exist."""
        for d in [self.data_dir, self.log_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline execution parameters."""
    batch_size: int = 1000
    log_level: str = "INFO"
    max_retries: int = 3
    retry_delay: float = 2.0
    schedule_interval: int = 300  # seconds


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration — aggregates all sub-configs."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)


# ────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────

_cached_config: Optional[AppConfig] = None


def get_config(*, reload: bool = False) -> AppConfig:
    """
    Build and cache the application configuration from environment variables.

    Args:
        reload: Force re-read from environment (useful for testing).

    Returns:
        Fully-populated AppConfig instance.
    """
    global _cached_config
    if _cached_config is not None and not reload:
        return _cached_config

    def _env(key: str, default: str = "") -> str:
        return os.getenv(key, default)

    def _env_int(key: str, default: int = 0) -> int:
        return int(os.getenv(key, str(default)))

    def _env_bool(key: str, default: bool = False) -> bool:
        return os.getenv(key, str(default)).lower() in ("true", "1", "yes")

    def _env_list(key: str, default: str = "") -> List[str]:
        raw = os.getenv(key, default)
        return [s.strip() for s in raw.split(",") if s.strip()] if raw else []

    db = DatabaseConfig(
        host=_env("MYSQL_HOST", "localhost"),
        port=_env_int("MYSQL_PORT", 3306),
        user=_env("MYSQL_USER", "root"),
        password=_env("MYSQL_PASSWORD", ""),
        database=_env("MYSQL_DATABASE", "vendor_performance"),
        pool_size=_env_int("MYSQL_POOL_SIZE", 5),
    )

    email = EmailConfig(
        enabled=_env_bool("EMAIL_ENABLED", False),
        smtp_server=_env("SMTP_SERVER", "smtp.gmail.com"),
        smtp_port=_env_int("SMTP_PORT", 587),
        sender_email=_env("SENDER_EMAIL"),
        sender_password=_env("SENDER_PASSWORD"),
        recipient_emails=_env_list("RECIPIENT_EMAILS"),
    )

    paths = PathConfig(
        project_root=_PROJECT_ROOT,
        data_dir=Path(_env("DATA_DIR", str(_PROJECT_ROOT / "data"))),
        log_dir=Path(_env("LOG_DIR", str(_PROJECT_ROOT / "logs"))),
        output_dir=Path(_env("OUTPUT_DIR", str(_PROJECT_ROOT / "output"))),
    )

    pipeline = PipelineConfig(
        batch_size=_env_int("PIPELINE_BATCH_SIZE", 1000),
        log_level=_env("PIPELINE_LOG_LEVEL", "INFO"),
    )

    _cached_config = AppConfig(
        database=db,
        email=email,
        paths=paths,
        pipeline=pipeline,
    )

    paths.ensure_dirs()
    return _cached_config


if __name__ == "__main__":
    cfg = get_config()
    print(f"Database : {cfg.database.host}:{cfg.database.port}/{cfg.database.database}")
    print(f"Data dir : {cfg.paths.data_dir}")
    print(f"Log dir  : {cfg.paths.log_dir}")
    print(f"Batch    : {cfg.pipeline.batch_size}")
    print(f"Email    : {'enabled' if cfg.email.enabled else 'disabled'}")
