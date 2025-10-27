import { RuntimeInfo } from '../types/api';

type RuntimeStatusProps = {
  runtime: RuntimeInfo | null;
  error: string | null;
};

export const RuntimeStatus = ({ runtime, error }: RuntimeStatusProps) => {
  if (error) {
    return <p style={{ margin: 0, color: '#b91c1c' }}>Не удалось получить статус: {error}</p>;
  }

  if (!runtime) {
    return <p style={{ margin: 0, color: '#475569' }}>Загружаем информацию о вычислениях…</p>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 14, color: '#0f172a' }}>
      <div>
        <strong>PyTorch:</strong>{' '}
        {runtime.system.torch_available
          ? `есть${runtime.system.torch_version ? ` (версия ${runtime.system.torch_version})` : ''}`
          : 'не обнаружен'}
      </div>
      <div>
        <strong>CUDA:</strong>{' '}
        {runtime.system.cuda_available
          ? `доступна${runtime.system.cuda_name ? ` (${runtime.system.cuda_name})` : ''} — устройств: ${runtime.system.cuda_device_count}`
          : 'не обнаружена'}
      </div>
      {runtime.system.mps_available && (
        <div>
          <strong>Apple MPS:</strong> доступен
        </div>
      )}
      <div>
        <strong>YOLO_DEVICE:</strong>{' '}
        {(runtime.system.env_device && runtime.system.env_device.trim()) || 'auto'}
      </div>
      {runtime.system.cuda_visible_devices && (
        <div>
          <strong>CUDA_VISIBLE_DEVICES:</strong>{' '}
          {runtime.system.cuda_visible_devices}
        </div>
      )}
      <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: 8, marginTop: 4 }}>
        <div style={{ fontWeight: 600, marginBottom: 6 }}>Воркеры обработки</div>
        {runtime.workers.length === 0 ? (
          <p style={{ margin: 0, color: '#475569' }}>Воркеры ещё не запущены или камеры не активны.</p>
        ) : (
          <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'grid', gap: 8 }}>
            {runtime.workers.map(worker => (
              <li
                key={worker.camera}
                style={{
                  background: '#fff',
                  border: '1px solid #e2e8f0',
                  borderRadius: 6,
                  padding: 10,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6,
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ fontWeight: 600 }}>{worker.camera}</span>
                  <span style={{ color: worker.using_gpu ? '#0f766e' : '#b91c1c', fontWeight: 600 }}>
                    {worker.using_gpu ? 'GPU' : 'CPU'}
                  </span>
                </div>
                <div style={{ fontSize: 13, color: '#475569' }}>
                  Фактическое устройство: {worker.actual_device}
                </div>
                <div style={{ fontSize: 13, color: '#475569' }}>
                  Предпочтение: {worker.preferred_device || 'auto'} → выбранное: {worker.selected_device}
                </div>
                {worker.gpu_unavailable_reason && (
                  <div style={{ fontSize: 13, color: '#b91c1c' }}>
                    <strong>Почему не GPU:</strong> {worker.gpu_unavailable_reason}
                  </div>
                )}
                {worker.device_error && worker.device_error !== worker.gpu_unavailable_reason && (
                  <div style={{ fontSize: 13, color: '#b91c1c' }}>
                    <strong>Ошибка модели:</strong> {worker.device_error}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default RuntimeStatus;
