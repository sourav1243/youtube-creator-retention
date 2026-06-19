from __future__ import annotations

import logging
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

logger = logging.getLogger(__name__)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):

        def _replacer(m: re.Match) -> str:
            var_name = m.group(1)
            env_val = os.getenv(var_name)
            if env_val is None:
                warnings.warn(f"Environment variable {var_name} is not set - using literal '${{{var_name}}}'")
                return m.group(0)
            return env_val

        return _ENV_VAR_PATTERN.sub(_replacer, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


@dataclass
class ExtractionConfig:
    max_ids_per_call: int = 50
    quota_daily_default: int = 10000
    retry_max_attempts: int = 5
    retry_base_delay_s: float = 2.0
    request_timeout: int = 30
    max_pages_per_channel: int = 1


@dataclass
class CleaningConfig:
    outlier_method: str = "cap_99th_percentile"
    outlier_cap_flag: bool = True
    log_transform: str = "log1p"


@dataclass
class FeaturesConfig:
    window_30d: int = 30
    window_90d: int = 90
    min_videos_for_scoring: int = 2


@dataclass
class ClusteringConfig:
    k_min: int = 3
    k_max: int = 5
    n_init: int = 10
    random_state: int = 42
    scaler: str = "RobustScaler"


@dataclass
class MySQLConfig:
    host: str = "localhost"
    user: str = "root"
    password: str = ""
    database: str = "youtube_creator_retention"
    port: int = 3306

    @property
    def dsn(self) -> str:
        return f"mysql+mysqlconnector://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/pipeline.log"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


@dataclass
class PipelineConfig:
    n_channels_total: int = 5000
    sample_size_tier_b: int = 1000
    random_seed: int = 42


@dataclass
class Settings:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    mysql: MySQLConfig = field(default_factory=MySQLConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def update_from_dict(self, section: str, data: dict) -> None:
        target = getattr(self, section, None)
        if target is None:
            logger.warning("Unknown config section '%s' — ignored", section)
            return
        for key, value in data.items():
            if hasattr(target, key):
                setattr(target, key, value)

    @property
    def youtube_api_key(self) -> str:
        key = os.getenv("YOUTUBE_API_KEY")
        if not key:
            raise ValueError("YOUTUBE_API_KEY not set in .env")
        return key


def load_settings() -> Settings:
    load_dotenv(ROOT_DIR / ".env")

    config_path = ROOT_DIR / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        raw = _expand_env_vars(raw)
    else:
        raw = {}

    settings = Settings()

    section_map = {
        "pipeline": "pipeline",
        "extraction": "extraction",
        "cleaning": "cleaning",
        "features": "features",
        "clustering": "clustering",
        "mysql": "mysql",
        "logging": "logging",
    }
    for yaml_key, attr in section_map.items():
        if yaml_key in raw:
            settings.update_from_dict(attr, raw[yaml_key])

    return settings


settings = load_settings()

if __name__ == "__main__":
    print(f"Settings loaded: {settings}")
    print(f"  Pipeline: {settings.pipeline}")
    print(f"  Extraction: {settings.extraction}")
    print(f"  MySQL DSN: {settings.mysql.dsn}")
    print(f"  API Key present: {bool(settings.youtube_api_key)}")
