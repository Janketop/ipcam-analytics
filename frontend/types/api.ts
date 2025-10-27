export type EventMeta = {
  plate?: string | null;
  entry_ts?: string | null;
};

export type EventItem = {
  id?: number;
  type: string;
  start_ts: string;
  confidence?: number;
  snapshot_url?: string;
  camera?: string;
  meta?: EventMeta;
};

export type Stat = {
  type: string;
  cnt: number;
};

export type CameraStatus = 'online' | 'offline' | 'starting' | 'stopping' | 'no_signal' | 'unknown';

export type Camera = {
  id: number;
  name: string;
  status?: CameraStatus;
  fps?: number | null;
  lastFrameTs?: string | null;
  uptimeSec?: number | null;
};

export type RuntimeWorker = {
  camera: string;
  preferred_device: string;
  selected_device: string;
  actual_device: string;
  using_gpu: boolean;
  visualize_enabled: boolean;
  device_error?: string | null;
  gpu_unavailable_reason?: string | null;
  started_at?: string | null;
  last_frame_at?: string | null;
  uptime_seconds?: number | null;
  fps?: number | null;
};

export type RuntimeSystem = {
  torch_available: boolean;
  torch_version?: string | null;
  cuda_available: boolean;
  cuda_device_count: number;
  cuda_name?: string | null;
  mps_available?: boolean;
  env_device?: string | null;
  cuda_visible_devices?: string | null;
};

export type RuntimeSummary = {
  total_workers: number;
  alive_workers: number;
  avg_fps?: number | null;
  max_uptime_seconds?: number | null;
  latest_frame_at?: string | null;
};

export type RuntimeInfo = {
  system: RuntimeSystem;
  workers: RuntimeWorker[];
  summary?: RuntimeSummary;
};
