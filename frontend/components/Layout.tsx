import Head from 'next/head';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { ReactNode } from 'react';

const links = [
  { href: '/dashboard', label: 'Дашборд' },
  { href: '/events', label: 'События' },
  { href: '/cameras', label: 'Камеры' },
  { href: '/identification', label: 'Определение' },
];

type LayoutProps = {
  title: string;
  children: ReactNode;
};

export const Layout = ({ title, children }: LayoutProps) => {
  const router = useRouter();

  return (
    <>
      <Head>
        <title>{title}</title>
      </Head>
      <div style={{ minHeight: '100vh', background: '#f1f5f9' }}>
        <header
          style={{
            background: '#0f172a',
            color: '#fff',
            padding: '16px 0',
            boxShadow: '0 6px 16px rgba(15,23,42,0.45)',
          }}
        >
          <nav style={{ maxWidth: 1100, margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 16px', gap: 16, flexWrap: 'wrap' }}>
            <Link href="/" style={{ color: '#fff', fontWeight: 700, fontSize: 18, textDecoration: 'none' }}>
              IP-CAM Analytics
            </Link>
            <ul style={{ display: 'flex', listStyle: 'none', margin: 0, padding: 0, gap: 12, flexWrap: 'wrap' }}>
              {links.map(link => {
                const active = router.pathname === link.href;
                return (
                  <li key={link.href}>
                    <Link
                      href={link.href}
                      style={{
                        padding: '8px 14px',
                        borderRadius: 999,
                        textDecoration: 'none',
                        background: active ? '#2563eb' : 'transparent',
                        color: '#fff',
                        border: active ? '1px solid #2563eb' : '1px solid rgba(148, 163, 184, 0.4)',
                        transition: 'all 0.2s ease',
                      }}
                    >
                      {link.label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>
        </header>
        <main style={{ maxWidth: 1100, margin: '0 auto', padding: '24px 16px 48px', fontFamily: 'system-ui, sans-serif' }}>
          {children}
        </main>
      </div>
    </>
  );
};

export default Layout;
