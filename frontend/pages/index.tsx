import Link from 'next/link';
import Layout from '../components/Layout';

const HomePage = () => (
  <Layout title="IP-CAM Analytics — Главная">
    <h1>Добро пожаловать в IP-CAM Analytics</h1>
    <p style={{ maxWidth: 720, color: '#475569' }}>
      Используйте разделы ниже, чтобы управлять камерами, отслеживать события и следить за ключевыми метриками системы.
      Все данные обновляются в режиме реального времени.
    </p>

    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
      <Link
        href="/dashboard"
        style={{
          padding: 20,
          borderRadius: 12,
          border: '1px solid #e2e8f0',
          background: '#fff',
          boxShadow: '0 8px 16px rgba(15,23,42,0.08)',
          textDecoration: 'none',
          color: '#0f172a',
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
        }}
      >
        <span style={{ fontSize: 20, fontWeight: 700 }}>Дашборд</span>
        <span style={{ color: '#475569' }}>
          Графики по событиям, статус вычислений, живой поток или симуляция. Центральное место для мониторинга.
        </span>
      </Link>

      <Link
        href="/events"
        style={{
          padding: 20,
          borderRadius: 12,
          border: '1px solid #e2e8f0',
          background: '#fff',
          boxShadow: '0 8px 16px rgba(15,23,42,0.08)',
          textDecoration: 'none',
          color: '#0f172a',
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
        }}
      >
        <span style={{ fontSize: 20, fontWeight: 700 }}>События</span>
        <span style={{ color: '#475569' }}>
          Таблица всех детекций с мгновенным обновлением и миниатюрами для быстрого просмотра контекста.
        </span>
      </Link>

      <Link
        href="/cameras"
        style={{
          padding: 20,
          borderRadius: 12,
          border: '1px solid #e2e8f0',
          background: '#fff',
          boxShadow: '0 8px 16px rgba(15,23,42,0.08)',
          textDecoration: 'none',
          color: '#0f172a',
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
        }}
      >
        <span style={{ fontSize: 20, fontWeight: 700 }}>Камеры</span>
        <span style={{ color: '#475569' }}>
          Добавляйте новые потоки (RTSP или HTTP/MJPEG), удаляйте устаревшие камеры и держите список устройств в актуальном
          состоянии.
        </span>
      </Link>

      <Link
        href="/identification"
        style={{
          padding: 20,
          borderRadius: 12,
          border: '1px solid #e2e8f0',
          background: '#fff',
          boxShadow: '0 8px 16px rgba(15,23,42,0.08)',
          textDecoration: 'none',
          color: '#0f172a',
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
        }}
      >
        <span style={{ fontSize: 20, fontWeight: 700 }}>Определение сотрудников</span>
        <span style={{ color: '#475569' }}>
          Собирайте снимки лиц прямо из камер, отмечайте сотрудников и отделяйте клиентов, чтобы подготовить датасет без очных фотосессий.
        </span>
      </Link>
    </div>
  </Layout>
);

export default HomePage;
