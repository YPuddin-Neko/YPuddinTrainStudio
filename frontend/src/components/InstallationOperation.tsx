import React from 'react';
import { ChevronDown, ChevronRight, Loader2 } from 'lucide-react';

export function InstallationProgress({ label, percent }: { label: string; percent?: number }) {
  return <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}
    className="h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
    <div className={`h-full rounded-full bg-blue-600 ${percent === undefined ? 'animate-pulse' : ''}`} style={{ width: `${percent ?? 25}%` }}/>
  </div>;
}

export function InstallationLog({ label, logs }: { label: string; logs: string[] }) {
  const viewport = React.useRef<HTMLPreElement>(null);
  const following = React.useRef(true);
  const content = logs.join('\n');
  React.useLayoutEffect(() => {
    if (following.current && viewport.current) viewport.current.scrollTop = viewport.current.scrollHeight;
  }, [content]);
  return <pre ref={viewport} aria-label={label} onScroll={event => {
    const log = event.currentTarget;
    following.current = log.scrollHeight - log.clientHeight - log.scrollTop <= 2;
  }} className="max-h-56 overflow-auto whitespace-pre-wrap break-all rounded-md bg-slate-950 p-3 font-mono text-[11px] leading-5 text-slate-200">{content}</pre>;
}

export default function InstallationOperation({ title, action, status, busy = false, failed = false, expanded, onToggle, children }: {
  title: string; action?: string; status: string; busy?: boolean; failed?: boolean;
  expanded?: boolean; onToggle?: () => void; children: React.ReactNode;
}) {
  const [open, setOpen] = React.useState(true);
  React.useEffect(() => { if (failed) setOpen(true); }, [failed]);
  const visible = expanded ?? open;
  return <div className="installation-operation rounded-lg border border-slate-200 dark:border-slate-700">
    <button type="button" aria-expanded={visible} className="flex w-full flex-wrap items-center gap-2 px-3 py-2 text-left text-xs" onClick={onToggle || (() => setOpen(value => !value))}>
      {visible ? <ChevronDown size={13}/> : <ChevronRight size={13}/>}{busy && <Loader2 size={13} className="animate-spin"/>}
      <span className="font-medium">{title}</span>{action && <span>{action}</span>}
      <span className={`ml-auto ${failed ? 'text-red-600 dark:text-red-300' : 'text-slate-500 dark:text-slate-400'}`}>{status}</span>
    </button>
    {visible && <div className="space-y-3 border-t border-slate-100 px-3 py-3 dark:border-slate-700">{children}</div>}
  </div>;
}
