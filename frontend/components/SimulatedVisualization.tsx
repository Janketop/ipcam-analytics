import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

const CANVAS_W = 640;
const CANVAS_H = 360;

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

type Props = {
  onEvent: (event: SimEvent) => void;
};

export const SimulatedVisualization = ({ onEvent }: Props) => {
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

  const createPerson = useCallback((idx: number): PersonSim => ({
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
  }), []);

  useEffect(() => {
    personsRef.current = new Array(3).fill(0).map((_, idx) => createPerson(idx));
  }, [createPerson]);

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
        persons = new Array(3).fill(0).map((_, idx) => createPerson(idx));
        personsRef.current = persons;
      }

      if (runningRef.current) {
        persons.forEach(person => {
          person.x += person.vx * dt;
          person.y += person.vy * dt * 0.4;

          if (person.x < 60 || person.x > CANVAS_W - 60) {
            person.vx *= -1;
            person.x = Math.max(60, Math.min(CANVAS_W - 60, person.x));
          }

          if (person.y < 80 || person.y > CANVAS_H - 80) {
            person.vy *= -1;
            person.y = Math.max(80, Math.min(CANVAS_H - 80, person.y));
          }

          person.sway += dt * (0.8 + Math.random() * 0.4);
          person.headTilt = 0.15 + Math.abs(Math.sin(person.sway)) * 0.55;

          person.phoneCooldown -= dt;
          if (person.phoneCooldown <= 0) {
            person.phone = Math.random() > 0.5;
            person.phoneCooldown = 2 + Math.random() * 6;
          }

          const usingPhone = person.phone && person.headTilt > 0.35;
          if (usingPhone && time - person.lastEventTs > 2500) {
            const event: SimEvent = {
              id: eventIdRef.current++,
              label: 'PHONE_USAGE',
              probability: 0.65 + Math.random() * 0.3,
              camera: `sim_cam_${person.id + 1}`,
              ts: Date.now(),
            };
            person.lastEventTs = time;
            onEvent(event);
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

      persons.forEach(person => {
        const boxW = 110;
        const boxH = 200;
        const x1 = person.x - boxW / 2;
        const y1 = person.y - boxH / 2;

        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, boxW, boxH);

        ctx.fillStyle = '#22c55e';
        ctx.font = '16px system-ui';
        ctx.fillText('PERSON', x1 + 4, y1 - 8);

        const joints = [
          [person.x, y1 + 24],
          [person.x - 16, y1 + 46],
          [person.x + 16, y1 + 46],
          [person.x, y1 + 86],
          [person.x - 26, y1 + 126],
          [person.x + 26, y1 + 126],
          [person.x - 24, y1 + 180],
          [person.x + 24, y1 + 180],
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

        if (person.phone) {
          const phoneX = person.x + 32;
          const phoneY = y1 + 140;
          ctx.strokeStyle = '#fde047';
          ctx.lineWidth = 3;
          ctx.strokeRect(phoneX, phoneY, 28, 54);
          ctx.fillStyle = '#fde047';
          ctx.fillText('PHONE', phoneX - 6, phoneY - 10);
        }

        if (person.phone && person.headTilt > 0.35) {
          ctx.fillStyle = '#ef4444';
          ctx.font = '18px system-ui';
          ctx.fillText(`PHONE_USAGE (${person.headTilt.toFixed(2)})`, x1 + 6, y1 + boxH + 24);
        }

        ctx.save();
        ctx.translate(person.x, y1 + 20);
        ctx.rotate((person.headTilt - 0.2) * 0.8);
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
  }, [createPerson, onEvent]);

  useEffect(() => {
    return () => {
      personsRef.current = [];
      runningRef.current = false;
    };
  }, []);

  const controlStyle = useMemo(
    () => ({ display: 'flex', gap: 12, marginBottom: 12, alignItems: 'center', flexWrap: 'wrap' as const }),
    []
  );

  return (
    <div>
      <div style={controlStyle}>
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
            onChange={event => setSpeed(parseFloat(event.target.value))}
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
};

export default SimulatedVisualization;
