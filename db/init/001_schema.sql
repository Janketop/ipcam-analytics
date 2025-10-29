CREATE TABLE IF NOT EXISTS cameras (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  rtsp_url TEXT NOT NULL,
  active BOOLEAN DEFAULT TRUE,
  detect_person BOOLEAN DEFAULT TRUE,
  detect_car BOOLEAN DEFAULT TRUE,
  capture_entry_time BOOLEAN DEFAULT TRUE,
  idle_alert_time INTEGER NOT NULL DEFAULT 300,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS events (
  id BIGSERIAL PRIMARY KEY,
  camera_id INT REFERENCES cameras(id),
  type TEXT NOT NULL, -- 'PHONE_USAGE' | 'NOT_WORKING'
  start_ts TIMESTAMPTZ NOT NULL,
  end_ts TIMESTAMPTZ,
  confidence REAL,
  snapshot_url TEXT,
  meta JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS events_time_idx ON events(start_ts);
CREATE INDEX IF NOT EXISTS events_type_idx ON events(type);

CREATE TABLE IF NOT EXISTS audit_logs (
  id BIGSERIAL PRIMARY KEY,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  resource TEXT NOT NULL,
  ts TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS employees (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS face_samples (
  id BIGSERIAL PRIMARY KEY,
  event_id BIGINT UNIQUE REFERENCES events(id) ON DELETE CASCADE,
  employee_id INT REFERENCES employees(id) ON DELETE SET NULL,
  camera_id INT REFERENCES cameras(id) ON DELETE SET NULL,
  snapshot_url TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'unverified',
  candidate_key TEXT,
  embedding BYTEA,
  embedding_dim INTEGER,
  embedding_model TEXT,
  captured_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS face_samples_status_idx ON face_samples(status);
CREATE INDEX IF NOT EXISTS face_samples_employee_idx ON face_samples(employee_id);
CREATE INDEX IF NOT EXISTS face_samples_captured_idx ON face_samples(captured_at);
