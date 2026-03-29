CREATE TABLE IF NOT EXISTS scenes (
    scene_token VARCHAR(64) PRIMARY KEY,
    scene_name VARCHAR(64) NOT NULL,
    location VARCHAR(64) NOT NULL,
    num_samples INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    sample_idx INTEGER PRIMARY KEY,
    token VARCHAR(64) UNIQUE NOT NULL,
    scene_token VARCHAR(64) REFERENCES scenes(scene_token),
    timestamp BIGINT NOT NULL,
    location VARCHAR(64) NOT NULL,
    ego_translation DOUBLE PRECISION[],
    ego_rotation DOUBLE PRECISION[],
    num_annotations INTEGER NOT NULL,
    num_lidar_points INTEGER NOT NULL,
    cam_front_path VARCHAR(256),
    cam_front_left_path VARCHAR(256),
    cam_front_right_path VARCHAR(256),
    cam_back_path VARCHAR(256),
    cam_back_left_path VARCHAR(256),
    cam_back_right_path VARCHAR(256),
    lidar_top_path VARCHAR(256),
    radar_front_path VARCHAR(256),
    radar_front_left_path VARCHAR(256),
    radar_front_right_path VARCHAR(256),
    radar_back_left_path VARCHAR(256),
    radar_back_right_path VARCHAR(256)
);

CREATE TABLE IF NOT EXISTS annotations (
    id SERIAL PRIMARY KEY,
    sample_idx INTEGER REFERENCES samples(sample_idx),
    category VARCHAR(128) NOT NULL,
    translation DOUBLE PRECISION[],
    size DOUBLE PRECISION[],
    rotation DOUBLE PRECISION[],
    num_lidar_pts INTEGER,
    num_radar_pts INTEGER
);

CREATE INDEX IF NOT EXISTS idx_samples_location ON samples(location);
CREATE INDEX IF NOT EXISTS idx_samples_num_annotations ON samples(num_annotations);
CREATE INDEX IF NOT EXISTS idx_annotations_category ON annotations(category);
CREATE INDEX IF NOT EXISTS idx_annotations_sample ON annotations(sample_idx);

CREATE TABLE IF NOT EXISTS mining_sessions (
    session_id VARCHAR(32) PRIMARY KEY,
    label TEXT NOT NULL DEFAULT '',
    query TEXT NOT NULL DEFAULT '',
    mode VARCHAR(24) NOT NULL DEFAULT 'hybrid',
    modality_weights JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mining_session_examples (
    session_id VARCHAR(32) NOT NULL REFERENCES mining_sessions(session_id) ON DELETE CASCADE,
    sample_idx INTEGER NOT NULL REFERENCES samples(sample_idx) ON DELETE CASCADE,
    polarity VARCHAR(16) NOT NULL CHECK (polarity IN ('positive', 'negative')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (session_id, sample_idx)
);

CREATE INDEX IF NOT EXISTS idx_mining_session_examples_polarity
    ON mining_session_examples(session_id, polarity);

CREATE TABLE IF NOT EXISTS mining_cohorts (
    cohort_id VARCHAR(32) PRIMARY KEY,
    session_id VARCHAR(32) REFERENCES mining_sessions(session_id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    query TEXT NOT NULL DEFAULT '',
    filters JSONB NOT NULL DEFAULT '{}'::jsonb,
    sample_ids INTEGER[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mining_cohorts_created_at
    ON mining_cohorts(created_at DESC);
