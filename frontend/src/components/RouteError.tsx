import { useRouteError } from 'react-router-dom';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { useWorkspaceText } from '../utils/workspaceText';

export default function RouteError() {
  const error = useRouteError();
  const text = useWorkspaceText();
  const message = error instanceof Error ? error.message : '';
  const resourceError = /dynamically imported module|Importing a module script failed|Loading chunk|Failed to fetch.*module/i.test(message);
  return <main className="flex min-h-dvh items-center justify-center bg-slate-50 p-6 text-slate-700 dark:bg-slate-950 dark:text-slate-200">
    <section className="w-full max-w-lg rounded-lg border border-slate-200 bg-white p-6 dark:border-slate-700 dark:bg-slate-900" role="alert">
      <AlertCircle size={24} className="mb-4 text-amber-500"/>
      <h1 className="text-lg font-semibold">{resourceError ? text('页面资源未能加载', 'Page resources could not be loaded') : text('页面暂时无法显示', 'This page could not be displayed')}</h1>
      <p className="mt-3 text-sm leading-6 text-slate-500 dark:text-slate-400">{resourceError ? text('请重新载入页面。', 'Reload the page.') : text('请重试，或查看错误详情。', 'Try again or view the error details.')}</p>
      <button type="button" onClick={() => window.location.reload()} className="mt-5 inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm text-white"><RefreshCw size={15}/>{text('重新载入', 'Reload page')}</button>
      {message && <details className="mt-5 text-xs text-slate-500"><summary className="cursor-pointer">{text('错误详情', 'Error details')}</summary><p className="mt-2 break-all whitespace-pre-wrap">{message}</p></details>}
    </section>
  </main>;
}
