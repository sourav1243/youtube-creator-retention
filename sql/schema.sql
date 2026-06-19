CREATE DATABASE IF NOT EXISTS youtube_creator_retention
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE youtube_creator_retention;

CREATE TABLE IF NOT EXISTS channels (
    channel_id          VARCHAR(24) PRIMARY KEY,
    title               VARCHAR(255) NOT NULL,
    description         TEXT,
    country             VARCHAR(8),
    published_at        DATETIME,
    uploads_playlist_id VARCHAR(64),
    fetched_at          DATETIME NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS channel_snapshots (
    snapshot_id         BIGINT AUTO_INCREMENT PRIMARY KEY,
    channel_id          VARCHAR(24) NOT NULL,
    snapshot_date       DATE NOT NULL,
    subscriber_count    BIGINT NULL,
    subscriber_hidden   BOOLEAN NOT NULL DEFAULT FALSE,
    view_count_total    BIGINT NULL,
    video_count         INT NULL,
    UNIQUE KEY uq_channel_date (channel_id, snapshot_date),
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS videos (
    video_id            VARCHAR(24) PRIMARY KEY,
    channel_id          VARCHAR(24) NOT NULL,
    published_at        DATETIME NOT NULL,
    duration_seconds    INT,
    view_count          BIGINT NULL,
    like_count          BIGINT NULL,
    comment_count       BIGINT NULL,
    comments_disabled   BOOLEAN NOT NULL DEFAULT FALSE,
    fetched_at          DATETIME NOT NULL,
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id),
    INDEX idx_channel_published (channel_id, published_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS creator_features (
    channel_id              VARCHAR(24) PRIMARY KEY,
    computed_at             DATETIME NOT NULL,
    upload_freq_30d         DECIMAL(8,2),
    upload_freq_90d         DECIMAL(8,2),
    freq_trend_ratio        DECIMAL(8,4),
    momentum_ratio          DECIMAL(8,4),
    avg_engagement_rate     DECIMAL(8,4),
    days_since_last_upload  INT,
    upload_regularity       DECIMAL(8,4),
    duration_trend          DECIMAL(8,4),
    insufficient_history    BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS creator_clusters (
    channel_id          VARCHAR(24) PRIMARY KEY,
    model_version       VARCHAR(32) NOT NULL,
    algorithm           VARCHAR(16) NOT NULL DEFAULT 'kmeans',
    cluster_id          INT NOT NULL,
    cluster_label       VARCHAR(64) NOT NULL,
    risk_flag           ENUM('Healthy','Watch','At-Risk','Unscored') NOT NULL DEFAULT 'Unscored',
    risk_score          DECIMAL(6,4),
    confidence          DECIMAL(6,4),
    distance_to_centroid DECIMAL(10,4),
    scored_at           DATETIME NOT NULL,
    FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
