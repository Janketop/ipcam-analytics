import { useCallback, useEffect, useMemo, useState } from 'react';
import Layout from '../components/Layout';
import { useApiBase } from '../hooks/useApiBase';
import { Employee, FaceSample } from '../types/api';

const formatDateTime = (value?: string | null) => {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString();
};

const IdentificationPage = () => {
  const { normalizedApiBase, buildAbsoluteUrl } = useApiBase();
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [samples, setSamples] = useState<FaceSample[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingSampleId, setPendingSampleId] = useState<number | null>(null);
  const [selectedEmployee, setSelectedEmployee] = useState<Record<number, number | ''>>({});
  const [newEmployeeName, setNewEmployeeName] = useState<Record<number, string>>({});
  const [newEmployeeAccountId, setNewEmployeeAccountId] = useState<Record<number, string>>({});

  const fetchEmployees = useCallback(async () => {
    const response = await fetch(`${normalizedApiBase}/employees`);
    if (!response.ok) {
      throw new Error('Не удалось загрузить список сотрудников');
    }
    const data = await response.json();
    const items = Array.isArray(data?.employees) ? (data.employees as Employee[]) : [];
    setEmployees(items);
  }, [normalizedApiBase]);

  const fetchSamples = useCallback(async () => {
    const response = await fetch(`${normalizedApiBase}/face-samples?status=unverified&limit=60`);
    if (!response.ok) {
      throw new Error('Не удалось загрузить снимки лиц');
    }
    const data = await response.json();
    const items = Array.isArray(data?.faceSamples) ? (data.faceSamples as FaceSample[]) : [];
    setSamples(items);
    setSelectedEmployee(prev => {
      const next: Record<number, number | ''> = {};
      for (const sample of items) {
        next[sample.id] = prev[sample.id] ?? '';
      }
      return next;
    });
    setNewEmployeeName(prev => {
      const next: Record<number, string> = {};
      for (const sample of items) {
        next[sample.id] = prev[sample.id] ?? '';
      }
      return next;
    });
    setNewEmployeeAccountId(prev => {
      const next: Record<number, string> = {};
      for (const sample of items) {
        next[sample.id] = prev[sample.id] ?? '';
      }
      return next;
    });
  }, [normalizedApiBase]);

  const refreshAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await Promise.all([fetchEmployees(), fetchSamples()]);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'Неизвестная ошибка загрузки данных');
    } finally {
      setLoading(false);
    }
  }, [fetchEmployees, fetchSamples]);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  const employeeOptions = useMemo(() => {
    return employees.map(employee => {
      const label = employee.accountId
        ? `${employee.name} · ${employee.accountId}`
        : employee.name;
      return (
        <option key={employee.id} value={employee.id}>
          {label}
        </option>
      );
    });
  }, [employees]);

  const assignSample = useCallback(
    async (sampleId: number) => {
      const employeeId = selectedEmployee[sampleId];
      if (!employeeId) {
        setError('Сначала выберите сотрудника для назначения.');
        return;
      }
      setPendingSampleId(sampleId);
      setError(null);
      try {
        const response = await fetch(`${normalizedApiBase}/face-samples/${sampleId}/assign`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ employee_id: employeeId }),
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          const detail = payload?.detail;
          throw new Error(typeof detail === 'string' ? detail : 'Не удалось сохранить привязку к сотруднику');
        }
        await Promise.all([fetchEmployees(), fetchSamples()]);
      } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : 'Неизвестная ошибка при назначении сотрудника');
      } finally {
        setPendingSampleId(null);
      }
    },
    [fetchEmployees, fetchSamples, normalizedApiBase, selectedEmployee]
  );

  const createAndAssign = useCallback(
    async (sampleId: number) => {
      const rawName = newEmployeeName[sampleId] ?? '';
      const name = rawName.trim();
      const rawAccountId = newEmployeeAccountId[sampleId] ?? '';
      const accountId = rawAccountId.trim();
      if (!name) {
        setError('Введите имя для нового сотрудника.');
        return;
      }
      setPendingSampleId(sampleId);
      setError(null);
      try {
        const requestPayload: Record<string, unknown> = { name };
        if (accountId) {
          requestPayload.account_id = accountId;
        }
        const createResponse = await fetch(`${normalizedApiBase}/employees`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestPayload),
        });
        if (!createResponse.ok) {
          const payload = await createResponse.json().catch(() => ({}));
          const detail = payload?.detail;
          throw new Error(typeof detail === 'string' ? detail : 'Не удалось создать сотрудника');
        }
        const created = await createResponse.json();
        const employee = created?.employee as Employee | undefined;
        if (!employee || typeof employee.id !== 'number') {
          throw new Error('Ответ сервера не содержит идентификатор сотрудника');
        }
        setSelectedEmployee(prev => ({ ...prev, [sampleId]: employee.id }));
        setNewEmployeeName(prev => ({ ...prev, [sampleId]: '' }));
        setNewEmployeeAccountId(prev => ({ ...prev, [sampleId]: '' }));
        const assignResponse = await fetch(`${normalizedApiBase}/face-samples/${sampleId}/assign`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ employee_id: employee.id }),
        });
        if (!assignResponse.ok) {
          const payload = await assignResponse.json().catch(() => ({}));
          const detail = payload?.detail;
          throw new Error(typeof detail === 'string' ? detail : 'Не удалось привязать снимок к сотруднику');
        }
        await Promise.all([fetchEmployees(), fetchSamples()]);
      } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : 'Неизвестная ошибка при создании сотрудника');
      } finally {
        setPendingSampleId(null);
      }
    },
    [
      fetchEmployees,
      fetchSamples,
      newEmployeeAccountId,
      newEmployeeName,
      normalizedApiBase,
    ]
  );

  const markAsClient = useCallback(
    async (sampleId: number) => {
      setPendingSampleId(sampleId);
      setError(null);
      try {
        const response = await fetch(`${normalizedApiBase}/face-samples/${sampleId}/mark`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: 'client' }),
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          const detail = payload?.detail;
          throw new Error(typeof detail === 'string' ? detail : 'Не удалось пометить снимок как клиента');
        }
        await fetchSamples();
      } catch (err) {
        console.error(err);
        setError(err instanceof Error ? err.message : 'Неизвестная ошибка при обновлении статуса');
      } finally {
        setPendingSampleId(null);
      }
    },
    [fetchSamples, normalizedApiBase]
  );

  return (
    <Layout title="Определение сотрудников">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
        <section
          style={{
            background: '#fff',
            padding: 24,
            borderRadius: 16,
            border: '1px solid #e2e8f0',
            boxShadow: '0 16px 32px rgba(15,23,42,0.08)',
          }}
        >
          <h1 style={{ marginTop: 0, marginBottom: 16 }}>Подбор эталонных лиц сотрудников</h1>
          <p style={{ margin: 0, color: '#475569', lineHeight: 1.6 }}>
            Система автоматически сохраняет снимки людей из кадров. Отметьте семерых сотрудников,
            чтобы в дальнейшем модель сопоставляла их лица и не путала с клиентами. Для каждого снимка
            можно выбрать сотрудника из списка или создать новую запись. Если на фото клиент, пометьте его
            соответствующей кнопкой — запись не попадёт в справочник сотрудников.
          </p>
        </section>

        {error && (
          <div
            style={{
              background: '#fee2e2',
              color: '#991b1b',
              padding: '12px 16px',
              borderRadius: 12,
              border: '1px solid #fecaca',
            }}
          >
            {error}
          </div>
        )}

        <section
          style={{
            background: '#fff',
            padding: 24,
            borderRadius: 16,
            border: '1px solid #e2e8f0',
            boxShadow: '0 16px 32px rgba(15,23,42,0.06)',
          }}
        >
          <h2 style={{ marginTop: 0, marginBottom: 12 }}>Сотрудники</h2>
          {employees.length === 0 ? (
            <p style={{ margin: 0, color: '#64748b' }}>
              Пока сотрудников нет. Используйте карточки ниже, чтобы создать первых сотрудников на основе снимков.
            </p>
          ) : (
            <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'grid', gap: 12 }}>
              {employees.map(employee => (
                <li
                  key={employee.id}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '12px 16px',
                    border: '1px solid #e2e8f0',
                    borderRadius: 12,
                    background: '#f8fafc',
                  }}
                >
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <span style={{ fontWeight: 600 }}>{employee.name}</span>
                    <span style={{ color: '#475569', fontSize: 14 }}>
                      Аккаунт: {employee.accountId?.trim() ? employee.accountId : 'не указан'}
                    </span>
                  </div>
                  <span style={{ color: '#475569', fontSize: 14 }}>
                    {employee.sampleCount} снимков
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section
          style={{
            background: '#fff',
            padding: 24,
            borderRadius: 16,
            border: '1px solid #e2e8f0',
            boxShadow: '0 16px 32px rgba(15,23,42,0.06)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <h2 style={{ margin: 0 }}>Непомеченные лица</h2>
            {loading && <span style={{ color: '#64748b' }}>Загрузка…</span>}
          </div>
          {samples.length === 0 ? (
            <p style={{ margin: 0, color: '#64748b' }}>
              Сейчас нет новых снимков. Как только в кадр попадут лица, здесь появятся карточки для подтверждения.
            </p>
          ) : (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                gap: 20,
              }}
            >
              {samples.map(sample => {
                const imgUrl = buildAbsoluteUrl(sample.snapshotUrl) ?? sample.snapshotUrl;
                const disabled = pendingSampleId === sample.id;
                return (
                  <div
                    key={sample.id}
                    style={{
                      border: '1px solid #e2e8f0',
                      borderRadius: 16,
                      overflow: 'hidden',
                      display: 'flex',
                      flexDirection: 'column',
                      background: '#f8fafc',
                    }}
                  >
                    <div style={{ position: 'relative', background: '#0f172a', minHeight: 180 }}>
                      {imgUrl ? (
                        <img
                          src={imgUrl}
                          alt="Снимок лица"
                          style={{ width: '100%', display: 'block', objectFit: 'cover' }}
                        />
                      ) : (
                        <div
                          style={{
                            position: 'absolute',
                            inset: 0,
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            color: '#e2e8f0',
                          }}
                        >
                          Снимок недоступен
                        </div>
                      )}
                    </div>
                    <div style={{ padding: '16px 16px 20px', display: 'flex', flexDirection: 'column', gap: 12 }}>
                      <div style={{ display: 'grid', gap: 4, fontSize: 14, color: '#475569' }}>
                        <span><strong>Камера:</strong> {sample.camera ?? '—'}</span>
                        <span><strong>Ключ в кадре:</strong> {sample.candidateKey ?? 'не определён'}</span>
                        <span><strong>Время:</strong> {formatDateTime(sample.capturedAt)}</span>
                      </div>

                      <div style={{ display: 'grid', gap: 8 }}>
                        <label style={{ fontSize: 14, color: '#0f172a', fontWeight: 600 }}>Назначить существующего сотрудника</label>
                        <select
                          value={selectedEmployee[sample.id] ?? ''}
                          onChange={event => {
                            const value = event.target.value;
                            setSelectedEmployee(prev => ({
                              ...prev,
                              [sample.id]: value ? Number.parseInt(value, 10) : '',
                            }));
                          }}
                          disabled={disabled || employees.length === 0}
                          style={{
                            padding: '8px 12px',
                            borderRadius: 8,
                            border: '1px solid #cbd5f5',
                            fontSize: 14,
                          }}
                        >
                          <option value="">— Не выбрано —</option>
                          {employeeOptions}
                        </select>
                        <button
                          type="button"
                          onClick={() => assignSample(sample.id)}
                          disabled={disabled || !selectedEmployee[sample.id]}
                          style={{
                            padding: '10px 14px',
                            borderRadius: 8,
                            border: 'none',
                            background: disabled || !selectedEmployee[sample.id] ? '#cbd5f5' : '#2563eb',
                            color: '#fff',
                            fontWeight: 600,
                            cursor: disabled || !selectedEmployee[sample.id] ? 'not-allowed' : 'pointer',
                          }}
                        >
                          Назначить сотрудника
                        </button>
                      </div>

                      <div style={{ display: 'grid', gap: 8 }}>
                        <label style={{ fontSize: 14, color: '#0f172a', fontWeight: 600 }}>Создать нового сотрудника</label>
                        <input
                          type="text"
                          value={newEmployeeName[sample.id] ?? ''}
                          onChange={event => {
                            const value = event.target.value;
                            setNewEmployeeName(prev => ({ ...prev, [sample.id]: value }));
                          }}
                          placeholder="Имя сотрудника"
                          disabled={disabled}
                          style={{
                            padding: '8px 12px',
                            borderRadius: 8,
                            border: '1px solid #cbd5f5',
                            fontSize: 14,
                          }}
                        />
                        <label style={{ fontSize: 14, color: '#0f172a' }}>Аккаунт (опционально)</label>
                        <input
                          type="text"
                          value={newEmployeeAccountId[sample.id] ?? ''}
                          onChange={event => {
                            const value = event.target.value;
                            setNewEmployeeAccountId(prev => ({ ...prev, [sample.id]: value }));
                          }}
                          placeholder="Идентификатор аккаунта (опционально)"
                          disabled={disabled}
                          style={{
                            padding: '8px 12px',
                            borderRadius: 8,
                            border: '1px solid #cbd5f5',
                            fontSize: 14,
                          }}
                        />
                        <button
                          type="button"
                          onClick={() => createAndAssign(sample.id)}
                          disabled={disabled || !(newEmployeeName[sample.id] ?? '').trim()}
                          style={{
                            padding: '10px 14px',
                            borderRadius: 8,
                            border: 'none',
                            background: disabled || !(newEmployeeName[sample.id] ?? '').trim() ? '#cbd5f5' : '#0f172a',
                            color: '#fff',
                            fontWeight: 600,
                            cursor:
                              disabled || !(newEmployeeName[sample.id] ?? '').trim()
                                ? 'not-allowed'
                                : 'pointer',
                          }}
                        >
                          Создать и назначить
                        </button>
                      </div>

                      <button
                        type="button"
                        onClick={() => markAsClient(sample.id)}
                        disabled={disabled}
                        style={{
                          marginTop: 4,
                          padding: '10px 14px',
                          borderRadius: 8,
                          border: '1px solid #f87171',
                          background: '#fee2e2',
                          color: '#b91c1c',
                          fontWeight: 600,
                          cursor: disabled ? 'not-allowed' : 'pointer',
                        }}
                      >
                        Пометить как клиента
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </Layout>
  );
};

export default IdentificationPage;
