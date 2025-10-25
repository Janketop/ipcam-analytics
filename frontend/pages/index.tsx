import { useCallback, useEffect, useRef, useState } from 'react';
import { Chart, BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';
Chart.register(BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend);

type EventMeta = {
  plate?: string | null;
  entry_ts?: string | null;
};

type Event = {
  id?: number;
  type: string;
  start_ts: string;
  confidence?: number;
  snapshot_url?: string;
  camera?: string;
  meta?: EventMeta;
};

type Stat = { type: string; cnt: number };

type SimEvent = {
  id: number;
  label: string;
  probability: number;
  camera: string;
  ts: number;
};

type PersonSim = {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  sway: number;
  phone: boolean;
  phoneCooldown: number;
  headTilt: number;
  lastEventTs: number;
};

const CANVAS_W = 640;
const CANVAS_H = 360;

function SimulatedVisualization({ onEvent }: { onEvent: (e: SimEvent) => void }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const runningRef = useRef(true);
  const speedRef = useRef(1);
  const personsRef = useRef<PersonSim[]>([]);
  const eventIdRef = useRef(1);
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    runningRef.current = running;
  }, [running]);

  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  useEffect(() => {
    personsRef.current = new Array(3).fill(0).map((_, idx) => ({
      id: idx,
      x: 120 + idx * 160,
      y: 120 + (idx % 2) * 80,
      vx: 20 + Math.random() * 40,
      vy: 15 + Math.random() * 20,
      sway: Math.random() * Math.PI * 2,
      phone: Math.random() > 0.4,
      phoneCooldown: 2 + Math.random() * 4,
      headTilt: 0.1,
      lastEventTs: 0,
    }));
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let frameId: number;
    let lastTime = performance.now();

    const draw = (time: number) => {
      const dtBase = (time - lastTime) / 1000;
      lastTime = time;
      const dt = Math.min(dtBase * speedRef.current, 0.2);

      let persons = personsRef.current;
      if (!persons.length) {
        persons = new Array(3).fill(0).map((_, idx) => ({
          id: idx,
          x: 120 + idx * 160,
          y: 120 + (idx % 2) * 80,
          vx: 20 + Math.random() * 40,
          vy: 15 + Math.random() * 20,
          sway: Math.random() * Math.PI * 2,
          phone: Math.random() > 0.4,
          phoneCooldown: 2 + Math.random() * 4,
          headTilt: 0.1,
          lastEventTs: 0,
        }));
        personsRef.current = persons;
      }

      if (runningRef.current) {
        persons.forEach(p => {
          p.x += p.vx * dt;
          p.y += p.vy * dt * 0.4;
          if (p.x < 60 || p.x > CANVAS_W - 60) {
            p.vx *= -1;
            p.x = Math.max(60, Math.min(CANVAS_W - 60, p.x));
          }
          if (p.y < 80 || p.y > CANVAS_H - 80) {
            p.vy *= -1;
            p.y = Math.max(80, Math.min(CANVAS_H - 80, p.y));
          }

          p.sway += dt * (0.8 + Math.random() * 0.4);
          p.headTilt = 0.15 + Math.abs(Math.sin(p.sway)) * 0.55;

          p.phoneCooldown -= dt;
          if (p.phoneCooldown <= 0) {
            p.phone = Math.random() > 0.5;
            p.phoneCooldown = 2 + Math.random() * 6;
          }

          const usingPhone = p.phone && p.headTilt > 0.35;
          if (usingPhone && time - p.lastEventTs > 2500) {
            const ev: SimEvent = {
              id: eventIdRef.current++,
              label: 'PHONE_USAGE',
              probability: 0.65 + Math.random() * 0.3,
              camera: `sim_cam_${p.id + 1}`,
              ts: Date.now(),
            };
            p.lastEventTs = time;
            onEvent(ev);
          }
        });
      }

      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.lineWidth = 1;
      for (let x = 40; x < CANVAS_W; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, CANVAS_H);
        ctx.stroke();
      }
      for (let y = 40; y < CANVAS_H; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(CANVAS_W, y);
        ctx.stroke();
      }

      persons.forEach(p => {
        const boxW = 110;
        const boxH = 200;
        const x1 = p.x - boxW / 2;
        const y1 = p.y - boxH / 2;

        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, boxW, boxH);

        ctx.fillStyle = '#22c55e';
        ctx.font = '16px system-ui';
        ctx.fillText('PERSON', x1 + 4, y1 - 8);

        const joints = [
          [p.x, y1 + 24],
          [p.x - 16, y1 + 46],
          [p.x + 16, y1 + 46],
          [p.x, y1 + 86],
          [p.x - 26, y1 + 126],
          [p.x + 26, y1 + 126],
          [p.x - 24, y1 + 180],
          [p.x + 24, y1 + 180],
        ];
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 4;
        const pairs = [
          [0, 1],
          [0, 2],
          [1, 3],
          [2, 3],
          [3, 4],
          [3, 5],
          [4, 6],
          [5, 7],
        ];
        pairs.forEach(([a, b]) => {
          const p1 = joints[a];
          const p2 = joints[b];
          ctx.beginPath();
          ctx.moveTo(p1[0], p1[1]);
          ctx.lineTo(p2[0], p2[1]);
          ctx.stroke();
        });

        if (p.phone) {
          const phoneX = p.x + 32;
          const phoneY = y1 + 140;
          ctx.strokeStyle = '#fde047';
          ctx.lineWidth = 3;
          ctx.strokeRect(phoneX, phoneY, 28, 54);
          ctx.fillStyle = '#fde047';
          ctx.fillText('PHONE', phoneX - 6, phoneY - 10);
        }

        if (p.phone && p.headTilt > 0.35) {
          ctx.fillStyle = '#ef4444';
          ctx.font = '18px system-ui';
          ctx.fillText(`PHONE_USAGE (${p.headTilt.toFixed(2)})`, x1 + 6, y1 + boxH + 24);
        }

        ctx.save();
        ctx.translate(p.x, y1 + 20);
        ctx.rotate((p.headTilt - 0.2) * 0.8);
        ctx.fillStyle = '#fbbf24';
        ctx.beginPath();
        ctx.ellipse(0, 0, 18, 22, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });

      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.font = '14px system-ui';
      ctx.fillText(`скорость симуляции: x${speedRef.current.toFixed(1)}`, 12, CANVAS_H - 16);

      frameId = requestAnimationFrame(draw);
    };

    frameId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frameId);
  }, [onEvent]);

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, marginBottom: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <button
          onClick={() => setRunning(prev => !prev)}
          style={{
            backgroundColor: running ? '#ef4444' : '#22c55e',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            padding: '6px 12px',
            cursor: 'pointer',
          }}
        >
          {running ? 'Пауза' : 'Старт'}
        </button>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span>Скорость:</span>
          <input
            type="range"
            min="0.5"
            max="3"
            step="0.1"
            value={speed}
            onChange={e => setSpeed(parseFloat(e.target.value))}
          />
          <span>x{speed.toFixed(1)}</span>
        </label>
        <p style={{ margin: 0, color: '#475569', fontSize: 13 }}>
          Симуляция генерирует «детекции» телефонов, позволяя тренировать пайплайн без реального потока.
        </p>
      </div>
      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        style={{ width: '100%', maxWidth: CANVAS_W, borderRadius: 12, boxShadow: '0 12px 24px rgba(15,23,42,0.24)' }}
      />
    </div>
  );
}

export default function Home() {
  const [events, setEvents] = useState<Event[]>([]);
  const [stats, setStats] = useState<Stat[]>([]);
  const [cameras, setCameras] = useState<{id:number; name:string}[]>([]);
  const [simEvents, setSimEvents] = useState<SimEvent[]>([]);
  const [viewMode, setViewMode] = useState<'live' | 'sim'>('live');

  const resolveApiBase = useCallback(() => {
    const envBase = process.env.NEXT_PUBLIC_API_BASE;
    if (envBase && envBase.trim()) {
      return envBase.replace(/\/$/, '');
    }

    if (typeof window !== 'undefined') {
      const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
      const host = window.location.hostname;
      const port = process.env.NEXT_PUBLIC_API_PORT || '8000';
      return `${protocol}//${host}${port ? `:${port}` : ''}`;
    }

    return 'http://localhost:8000';
  }, []);

  const resolvePlate = (meta?: EventMeta) => {
    const plate = meta?.plate;
    if (!plate) return '—';
    return plate;
  };

  const resolveEntryTime = (meta?: EventMeta, fallback?: string) => {
    const ts = meta?.entry_ts || fallback;
    if (!ts) return '—';
    return new Date(ts).toLocaleString();
  };

  useEffect(() => {
    const base = resolveApiBase();

    fetch(`${base}/events`)
      .then(r => r.json())
      .then(d => setEvents(d.events || []))
      .catch(err => console.error('Не удалось получить события:', err));

    fetch(`${base}/stats`)
      .then(r => r.json()).then(d => setStats(d.stats || []))
      .catch(err => console.error('Не удалось получить статистику:', err));

    fetch(`${base}/cameras`)
      .then(r => r.json()).then(d => setCameras(d.cameras || []))
      .catch(err => console.error('Не удалось получить список камер:', err));

    const wsProtocol = base.startsWith('https') ? 'wss' : 'ws';
    const wsHost = base.replace(/^https?:\/\//, '');
    const ws = new WebSocket(`${wsProtocol}://${wsHost}/ws/events`);
    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data);
      setEvents(prev => [
        {
          type: data.type,
          start_ts: data.ts,
          confidence: data.confidence,
          snapshot_url: data.snapshot_url,
          camera: data.camera,
          meta: data.meta,
        },
        ...prev,
      ].slice(0,200));
    };
    return () => ws.close();
  }, [resolveApiBase]);

  useEffect(() => {
    const el = document.getElementById('statsChart') as HTMLCanvasElement | null;
    if (!el) return;
    const ctx = el.getContext('2d');
    if (!ctx) return;
    const anyChart = (Chart as any).instances && Object.values((Chart as any).instances)[0];
    if (anyChart && (anyChart as any).destroy) (anyChart as any).destroy();

    const labels = stats.map(s => s.type);
    const data = stats.map(s => s.cnt);
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{ label: 'События за 24 часа', data }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: true } },
        scales: { y: { beginAtZero: true } }
      }
    });
  }, [stats]);

  const firstCam = cameras[0]?.name;

  const handleSimEvent = useCallback((ev: SimEvent) => {
    setSimEvents(prev => [ev, ...prev].slice(0, 15));
  }, []);

  return (
    <main style={{ maxWidth: 1100, margin: '2rem auto', fontFamily: 'system-ui, sans-serif' }}>
      <h1>Аналитика IP-камер — События</h1>
      <p>Живые детекции будут появляться ниже. Приватность: лица размыты по умолчанию.</p>

      <section style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:16}}>
        <div style={{border:'1px solid #ddd', borderRadius:8, padding:12}}>
          <h3>Статистика (последние 24 часа)</h3>
          <canvas id="statsChart" height={200}></canvas>
        </div>
        <div style={{border:'1px solid #ddd', borderRadius:8, padding:12, display:'flex', flexDirection:'column', gap:12}}>
          <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', gap:12, flexWrap:'wrap'}}>
            <h3 style={{margin:0}}>Визуализация</h3>
            <div style={{display:'flex', gap:8, flexWrap:'wrap'}}>
              <button
                onClick={() => setViewMode('live')}
                style={{
                  padding:'6px 12px',
                  borderRadius:6,
                  border:'1px solid',
                  borderColor: viewMode === 'live' ? '#2563eb' : '#cbd5f5',
                  backgroundColor: viewMode === 'live' ? '#2563eb' : '#f8fafc',
                  color: viewMode === 'live' ? '#fff' : '#0f172a',
                  cursor:'pointer'
                }}
              >
                Реальный поток
              </button>
              <button
                onClick={() => setViewMode('sim')}
                style={{
                  padding:'6px 12px',
                  borderRadius:6,
                  border:'1px solid',
                  borderColor: viewMode === 'sim' ? '#2563eb' : '#cbd5f5',
                  backgroundColor: viewMode === 'sim' ? '#2563eb' : '#f8fafc',
                  color: viewMode === 'sim' ? '#fff' : '#0f172a',
                  cursor:'pointer'
                }}
              >
                Симулятор
              </button>
            </div>
          </div>

          {viewMode === 'live' ? (
            <div>
              <h4 style={{margin:'4px 0 12px'}}>
                Живой просмотр{firstCam ? ` — ${firstCam}`: ''}
              </h4>
              {firstCam ? (
                <img src={`http://localhost:8000/stream/${firstCam}`} alt="live" style={{width:'100%', borderRadius:8}} />
              ) : (
                <p>Камеры не настроены. Добавьте RTSP в <code>.env</code>.</p>
              )}
              <p style={{fontSize:12, color:'#666'}}>MJPEG-поток с наложенной разметкой (рамки, скелет, подписи).</p>
            </div>
          ) : (
            <div>
              <h4 style={{margin:'4px 0 12px'}}>Симулятор потоков для тестирования</h4>
              <SimulatedVisualization onEvent={handleSimEvent} />
              <div style={{marginTop:12, background:'#f1f5f9', padding:12, borderRadius:8}}>
                <h5 style={{margin:'0 0 8px'}}>Последние события симуляции</h5>
                {simEvents.length === 0 ? (
                  <p style={{margin:0, color:'#475569'}}>Пока нет сгенерированных детекций.</p>
                ) : (
                  <ul style={{listStyle:'none', margin:0, padding:0, display:'grid', gap:8}}>
                    {simEvents.map(ev => (
                      <li key={ev.id} style={{background:'#fff', borderRadius:6, padding:'8px 10px', border:'1px solid #e2e8f0'}}>
                        <div style={{fontWeight:600, display:'flex', justifyContent:'space-between'}}>
                          <span>{ev.label}</span>
                          <span>{ev.probability.toFixed(2)}</span>
                        </div>
                        <div style={{fontSize:12, color:'#475569'}}>камера: {ev.camera}</div>
                        <div style={{fontSize:12, color:'#475569'}}>{new Date(ev.ts).toLocaleTimeString()}</div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}
        </div>
      </section>

      <h2 style={{marginTop:24}}>Лента событий</h2>
      <div style={{overflowX:'auto'}}>
        <table style={{width:'100%', minWidth:720, borderCollapse:'collapse', border:'1px solid #e2e8f0'}}>
          <thead style={{background:'#f8fafc'}}>
            <tr>
              <th style={{padding:10, textAlign:'left', borderBottom:'1px solid #e2e8f0'}}>Превью</th>
              <th style={{padding:10, textAlign:'left', borderBottom:'1px solid #e2e8f0'}}>Событие</th>
              <th style={{padding:10, textAlign:'left', borderBottom:'1px solid #e2e8f0'}}>Время фиксации</th>
              <th style={{padding:10, textAlign:'left', borderBottom:'1px solid #e2e8f0'}}>Госномер</th>
              <th style={{padding:10, textAlign:'left', borderBottom:'1px solid #e2e8f0'}}>Время заезда</th>
              <th style={{padding:10, textAlign:'left', borderBottom:'1px solid #e2e8f0'}}>Уверенность</th>
            </tr>
          </thead>
          <tbody>
            {events.map((e, i) => (
              <tr key={i} style={{background: i % 2 === 0 ? '#fff' : '#f8fafc'}}>
                <td style={{padding:10, borderBottom:'1px solid #e2e8f0'}}>
                  {e.snapshot_url ? (
                    <img
                      src={`http://localhost:8000${e.snapshot_url}`}
                      alt="snapshot"
                      width={160}
                      style={{borderRadius:8, display:'block'}}
                    />
                  ) : (
                    <span style={{color:'#94a3b8'}}>—</span>
                  )}
                </td>
                <td style={{padding:10, borderBottom:'1px solid #e2e8f0', minWidth:160}}>
                  <div style={{fontWeight:600}}>{e.type}</div>
                  <div style={{color:'#475569', fontSize:13}}>{e.camera ? `Камера: ${e.camera}` : '—'}</div>
                </td>
                <td style={{padding:10, borderBottom:'1px solid #e2e8f0'}}>{new Date(e.start_ts).toLocaleString()}</td>
                <td style={{padding:10, borderBottom:'1px solid #e2e8f0', fontFamily:'monospace', fontSize:16}}>{resolvePlate(e.meta)}</td>
                <td style={{padding:10, borderBottom:'1px solid #e2e8f0'}}>{resolveEntryTime(e.meta, e.start_ts)}</td>
                <td style={{padding:10, borderBottom:'1px solid #e2e8f0'}}>
                  {e.confidence !== undefined ? e.confidence.toFixed(2) : '—'}
                </td>
              </tr>
            ))}
            {events.length === 0 && (
              <tr>
                <td colSpan={6} style={{padding:16, textAlign:'center', color:'#64748b'}}>
                  Пока нет зафиксированных событий.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  );
}
