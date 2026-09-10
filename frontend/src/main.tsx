import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';
import './styles/index.css';
import './i18n';

const queryClient = new QueryClient();

async function enableMocking() {
  // 仅当明确开启或未关闭时启用 MSW；支持 VITE_USE_MOCK=false 关闭 mock 连接真实后端
  if (import.meta.env.DEV && import.meta.env.VITE_USE_MOCK !== 'false') {
    const { worker } = await import('./mocks/browser');
    await worker.start({
      onUnhandledRequest: 'bypass',
    });
  }
}

enableMocking().then(() => {
  ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </React.StrictMode>
  );
});
