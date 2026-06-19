from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class ExtractionConfig:
    max_ids_per_call: int = 50
    quota_daily_default: int = 10000
    retry_max_attempts: int = 5
    retry_base_delay_s: float = 2.0
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
    k_min: int = 2
    k_max: int = 10
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
            raw = yaml.safe_load(f)
    else:
        raw = {}

    settings = Settings()

    if "pipeline" in raw:
        settings.pipeline = PipelineConfig(**raw["pipeline"])
    if "extraction" in raw:
        settings.extraction = ExtractionConfig(**raw["extraction"])
    if "cleaning" in raw:
        settings.cleaning = CleaningConfig(**raw["cleaning"])
    if "features" in raw:
        settings.features = FeaturesConfig(**raw["features"])
    if "clustering" in raw:
        settings.clustering = ClusteringConfig(**raw["clustering"])
    if "mysql" in raw:
        settings.mysql = MySQLConfig(**raw["mysql"])
    if "logging" in raw:
        settings.logging = LoggingConfig(**raw["logging"])

    return settings


settings = load_settings()

if __name__ == "__main__":
    print(f"Settings loaded: {settings}")
    print(f"  Pipeline: {settings.pipeline}")
    print(f"  Extraction: {settings.extraction}")
    print(f"  MySQL DSN: {settings.mysql.dsn}")
    print(f"  API Key present: {bool(settings.youtube_api_key)}")
