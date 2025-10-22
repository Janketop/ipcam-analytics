import { useEffect, useState } from 'react';
import { Chart, BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';
Chart.register(BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend);

type Event = {
  id?: number;
  type: string;
  start_ts: string;
  confidence?: number;
  snapshot_url?: string;
  camera?: string;
};

type Stat = { type: string; cnt: number };

export default function Home() {
  const [events, setEvents] = useState<Event[]>([]);
  const [stats, setStats] = useState<Stat[]>([]);
  const [cameras, setCameras] = useState<{id:number; name:string}[]>([]);

  useEffect(() => {
    fetch('http://localhost:8000/events')
      .then(r => r.json())
      .then(d => setEvents(d.events || []));

    fetch('http://localhost:8000/stats')
      .then(r => r.json()).then(d => setStats(d.stats || []));

    fetch('http://localhost:8000/cameras')
      .then(r => r.json()).then(d => setCameras(d.cameras || []));

    const ws = new WebSocket('ws://localhost:8000/ws/events');
    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data);
      setEvents(prev => [{ type: data.type, start_ts: data.ts, confidence: data.confidence, snapshot_url: data.snapshot_url, camera: data.camera }, ...prev].slice(0,200));
    };
    return () => ws.close();
  }, []);

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

  return (
    <main style={{ maxWidth: 1100, margin: '2rem auto', fontFamily: 'system-ui, sans-serif' }}>
      <h1>Аналитика IP-камер — События</h1>
      <p>Живые детекции будут появляться ниже. Приватность: лица размыты по умолчанию.</p>

      <section style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:16}}>
        <div style={{border:'1px solid #ddd', borderRadius:8, padding:12}}>
          <h3>Статистика (последние 24 часа)</h3>
          <canvas id="statsChart" height={200}></canvas>
        </div>
        <div style={{border:'1px solid #ddd', borderRadius:8, padding:12}}>
          <h3>Живой просмотр{firstCam ? ` — ${firstCam}`: ''}</h3>
          {firstCam ? (
            <img src={`http://localhost:8000/stream/${firstCam}`} alt="live" style={{width:'100%', borderRadius:8}} />
          ) : (
            <p>Камеры не настроены. Добавьте RTSP в <code>.env</code>.</p>
          )}
          <p style={{fontSize:12, color:'#666'}}>MJPEG-поток с наложенной разметкой (рамки, скелет, подписи).</p>
        </div>
      </section>

      <h2 style={{marginTop:24}}>Лента событий</h2>
      <ul style={{listStyle:'none', padding:0}}>
        {events.map((e,i) => (
          <li key={i} style={{border:'1px solid #ddd', borderRadius:8, padding:12, marginBottom:12, display:'flex', gap:12}}>
            {e.snapshot_url && <img src={`http://localhost:8000${e.snapshot_url}`} alt="snapshot" width={180} style={{borderRadius:8}} />}
            <div>
              <div><strong>{e.type}</strong> {e.camera ? `— ${e.camera}`: ''}</div>
              <div>{new Date(e.start_ts).toLocaleString()}</div>
              {e.confidence !== undefined && <div>уверенность: {e.confidence.toFixed(2)}</div>}
            </div>
          </li>
        ))}
      </ul>
    </main>
  );
}
