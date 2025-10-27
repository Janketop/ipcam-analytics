CREATE TABLE IF NOT EXISTS cameras (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  rtsp_url TEXT NOT NULL,
  active BOOLEAN DEFAULT TRUE,
  detect_person BOOLEAN DEFAULT TRUE,
  detect_car BOOLEAN DEFAULT TRUE,
  capture_entry_time BOOLEAN DEFAULT TRUE,
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
