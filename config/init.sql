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

CREATE TABLE IF NOT EXISTS tracks (
    track_id VARCHAR(64) PRIMARY KEY,
    scene_token VARCHAR(64) NOT NULL REFERENCES scenes(scene_token) ON DELETE CASCADE,
    scene_name VARCHAR(64) NOT NULL,
    location VARCHAR(64) NOT NULL,
    category VARCHAR(128) NOT NULL,
    start_timestamp BIGINT NOT NULL,
    end_timestamp BIGINT NOT NULL,
    sample_ids INTEGER[] NOT NULL DEFAULT '{}',
    sample_count INTEGER NOT NULL DEFAULT 0,
    annotation_count INTEGER NOT NULL DEFAULT 0,
    avg_num_lidar_pts DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_num_radar_pts DOUBLE PRECISION NOT NULL DEFAULT 0,
    max_num_lidar_pts INTEGER NOT NULL DEFAULT 0,
    max_num_radar_pts INTEGER NOT NULL DEFAULT 0,
    visibility_tokens VARCHAR(16)[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tracks_scene_category
    ON tracks(scene_token, category);

CREATE INDEX IF NOT EXISTS idx_tracks_location
    ON tracks(location);

CREATE TABLE IF NOT EXISTS track_observations (
    id BIGSERIAL PRIMARY KEY,
    track_id VARCHAR(64) NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
    sample_idx INTEGER NOT NULL REFERENCES samples(sample_idx) ON DELETE CASCADE,
    sample_token VARCHAR(64) NOT NULL,
    annotation_token VARCHAR(64) NOT NULL,
    observation_idx INTEGER NOT NULL,
    timestamp BIGINT NOT NULL,
    category VARCHAR(128) NOT NULL,
    translation DOUBLE PRECISION[],
    size DOUBLE PRECISION[],
    rotation DOUBLE PRECISION[],
    num_lidar_pts INTEGER NOT NULL DEFAULT 0,
    num_radar_pts INTEGER NOT NULL DEFAULT 0,
    visibility_token VARCHAR(16) NOT NULL DEFAULT '',
    attribute_tokens VARCHAR(64)[] NOT NULL DEFAULT '{}',
    UNIQUE(track_id, sample_idx, annotation_token)
);

CREATE INDEX IF NOT EXISTS idx_track_observations_track_sample
    ON track_observations(track_id, sample_idx);

CREATE INDEX IF NOT EXISTS idx_track_observations_timestamp
    ON track_observations(timestamp);

CREATE TABLE IF NOT EXISTS review_tasks (
    task_id VARCHAR(32) PRIMARY KEY,
    source_type VARCHAR(24) NOT NULL CHECK (source_type IN ('cohort', 'track', 'manual')),
    source_id VARCHAR(64),
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status VARCHAR(24) NOT NULL CHECK (
        status IN ('queued', 'assigned', 'in_progress', 'submitted', 'qa_failed', 'qa_passed', 'closed')
    ),
    assignee TEXT NOT NULL DEFAULT '',
    priority VARCHAR(16) NOT NULL DEFAULT 'normal',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    submitted_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_review_tasks_status_created
    ON review_tasks(status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_review_tasks_source
    ON review_tasks(source_type, source_id);

CREATE TABLE IF NOT EXISTS task_events (
    id BIGSERIAL PRIMARY KEY,
    task_id VARCHAR(32) NOT NULL REFERENCES review_tasks(task_id) ON DELETE CASCADE,
    event_type VARCHAR(32) NOT NULL,
    actor TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_task_events_task_created
    ON task_events(task_id, created_at);

CREATE TABLE IF NOT EXISTS cohort_exports (
    export_id VARCHAR(32) PRIMARY KEY,
    cohort_id VARCHAR(32) REFERENCES mining_cohorts(cohort_id) ON DELETE CASCADE,
    task_id VARCHAR(32) REFERENCES review_tasks(task_id) ON DELETE SET NULL,
    export_format VARCHAR(16) NOT NULL,
    manifest_version VARCHAR(16) NOT NULL DEFAULT 'v1',
    output_path TEXT NOT NULL,
    row_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cohort_exports_cohort_created
    ON cohort_exports(cohort_id, created_at DESC);
