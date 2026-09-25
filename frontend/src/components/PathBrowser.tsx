import React from 'react';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../api/client';
import { FsListResponse } from '../api/types';
import { ArrowUp, File, Folder, FolderOpen, Loader2, X } from 'lucide-react';
import { formatBytes } from '../utils/format';
import { formatApiError } from '../utils/errors';

// Paths belong to the server, so avoid the browser host's platform/path rules.
function browseDirectory(value: string): string {
  const path = value.trim() || '/';
  if (/^[a-z]:$/i.test(path)) return `${path}\\`;
  if (!/\.(safetensors|ckpt|pt|pth|bin|gguf|onnx|csv|json|toml|yaml|yml|txt|png|jpe?g|webp|bmp|tiff?|zip)$/i.test(path)) return path;
  const separator = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
  if (separator < 0) return '.';
  if (separator === 0) return path[0];
  const parent = path.slice(0, separator);
  return /^[a-z]:$/i.test(parent) ? path.slice(0, separator + 1) : parent;
}

function childPath(parent: string, name: string): string {
  const separator = parent.includes('\\') && !parent.includes('/') ? '\\' : '/';
  return `${parent.replace(/[\\/]$/, '')}${separator}${name}`;
}

export const PathPickerModal: React.FC<{
  isOpen: boolean;
  initialPath?: string;
  onSelect: (path: string) => void;
  onClose: () => void;
}> = ({ isOpen, initialPath = '/', onSelect, onClose }) => {
  const { t } = useTranslation();
  const [address, setAddress] = React.useState(() => browseDirectory(initialPath));
  const [data, setData] = React.useState<FsListResponse | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');
  const request = React.useRef<AbortController | null>(null);
  const addressRef = React.useRef<HTMLInputElement>(null);
  const dialogRef = React.useRef<HTMLDivElement>(null);
  const titleId = React.useId();

  const loadDirectory = React.useCallback(async (value: string) => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    const path = browseDirectory(value);
    setAddress(path); setData(null); setError(''); setLoading(true);
    addressRef.current?.focus();
    try {
      const result = await apiClient.get<FsListResponse>('/fs/list', { params: { path }, signal: controller.signal, silent: true });
      if (controller.signal.aborted) return;
      setData(result); setAddress(result.path);
    } catch (error) {
      if (!controller.signal.aborted) setError(formatApiError(error));
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    if (!isOpen) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    addressRef.current?.focus();
    return () => previousFocus?.focus();
  }, [isOpen]);

  React.useEffect(() => {
    if (!isOpen) return;
    void loadDirectory(initialPath);
    return () => request.current?.abort();
  }, [isOpen, initialPath, loadDirectory]);

  if (!isOpen) return null;
  const canSelect = !loading && !error && data !== null && address === data.path;
  const keyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); onClose(); }
    if (event.key !== 'Tab') return;
    const controls = dialogRef.current?.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled])');
    if (!controls?.length) return;
    const first = controls[0]; const last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  };

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby={titleId} onKeyDown={keyDown}
        className="min-w-0 bg-white dark:bg-slate-800 text-[var(--studio-text)] rounded-xl max-w-xl w-full max-h-[90vh] overflow-y-auto p-5 space-y-4 shadow-xl border border-slate-200 dark:border-slate-700"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex justify-between items-center gap-3 border-b pb-2 dark:border-slate-700">
          <h3 id={titleId} className="font-semibold text-lg">{t('pathBrowser.title')}</h3>
          <button type="button" onClick={onClose} aria-label={t('common.close')} className="ui-btn ui-btn-quiet ui-btn-icon"><X className="h-5 w-5" /></button>
        </div>
        <div className="space-y-2">
          <label htmlFor={`${titleId}-address`} className="text-xs text-[var(--studio-dim)]">{t('pathBrowser.address')}</label>
          <div className="flex min-w-0 gap-2">
            <input ref={addressRef} id={`${titleId}-address`} value={address} onChange={(e) => { request.current?.abort(); setAddress(e.target.value); setData(null); setLoading(false); setError(''); }}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); if (address.trim()) void loadDirectory(address); } }}
              className="min-w-0 flex-1 rounded border border-slate-300 px-3 py-2 text-sm font-mono dark:bg-slate-900 dark:border-slate-600" />
            <button type="button" disabled={!address.trim()} onClick={() => void loadDirectory(address)} className="ui-btn ui-btn-primary">{t('pathBrowser.openDirectory')}</button>
          </div>
          <p className="text-xs text-[var(--studio-dim)]">{t('pathBrowser.serverHint')}</p>
        </div>
        {loading && <div role="status" className="flex items-center gap-2 py-5 text-sm text-[var(--studio-dim)]"><Loader2 className="h-4 w-4 animate-spin" />{t('pathBrowser.loading')}</div>}
        {error && <div role="alert" className="space-y-2 rounded bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300"><p className="whitespace-pre-line break-words">{error}</p><button type="button" onClick={() => void loadDirectory(address)} className="ui-link">{t('common.retry')}</button></div>}
        {!loading && !error && !canSelect && <p className="text-sm text-[var(--studio-dim)]">{t('pathBrowser.openToBrowse')}</p>}
        {canSelect && data && <div className="max-h-60 overflow-y-auto divide-y divide-slate-100 dark:divide-slate-700" aria-label={t('pathBrowser.entries')}>
          {data.parent && <button type="button" onClick={() => void loadDirectory(data.parent!)} className="flex w-full items-center gap-2 p-2 text-left text-sm hover:bg-slate-50 dark:hover:bg-slate-700 font-medium text-[var(--studio-accent)]"><ArrowUp className="h-4 w-4" />{t('pathBrowser.parentDir')}</button>}
          {data.entries.map((entry) => <button type="button" key={entry.name} onClick={() => {
            const path = childPath(data.path, entry.name);
            if (entry.is_dir) void loadDirectory(path);
            else { onSelect(path); onClose(); }
          }} className="flex w-full min-w-0 items-center justify-between gap-3 p-2 text-left text-sm hover:bg-slate-50 dark:hover:bg-slate-700">
            <span className="flex min-w-0 items-center gap-2">{entry.is_dir ? <Folder className="h-4 w-4 shrink-0 text-[var(--studio-accent)]" /> : <File className="h-4 w-4 shrink-0 text-[var(--studio-dim)]" />}<span className="break-all">{entry.name}</span></span>
            <span className="shrink-0 text-xs text-[var(--studio-dim)]">{entry.is_dir ? t('pathBrowser.dir') : formatBytes(entry.size)}</span>
          </button>)}
          {data.entries.length === 0 && <p className="p-4 text-center text-sm text-[var(--studio-dim)]">{t('pathBrowser.empty')}</p>}
        </div>}
        <div className="flex justify-end gap-2 pt-3 border-t dark:border-slate-700">
          <button type="button" onClick={onClose} className="ui-btn">{t('common.cancel')}</button>
          <button type="button" disabled={!canSelect} onClick={() => { if (canSelect && data) { onSelect(data.path); onClose(); } }} className="ui-btn ui-btn-primary">{t('pathBrowser.selectCurrent')}</button>
        </div>
      </div>
    </div>
  );
};

export const PathInput: React.FC<{
  value: string;
  onChange: (val: string) => void;
  placeholder?: string;
  ariaLabel?: string;
  defaultPath?: string;
}> = ({ value = '', onChange, placeholder, ariaLabel, defaultPath }) => {
  const { t } = useTranslation();
  const [modalOpen, setModalOpen] = React.useState(false);
  return (
    <div className="path-input-control flex min-w-0 gap-2">
      <input type="text" aria-label={ariaLabel || t('pathBrowser.pathLabel')} value={value || ''} placeholder={placeholder} onChange={(e) => onChange(e.target.value)}
        className="min-w-0 flex-1 px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600 font-mono" />
      <button type="button" onClick={() => setModalOpen(true)} className="ui-btn path-input-browse">
        <FolderOpen className="w-4 h-4" /><span>{t('common.browse')}</span>
      </button>
      <PathPickerModal isOpen={modalOpen} initialPath={value || defaultPath || '/'} onSelect={onChange} onClose={() => setModalOpen(false)} />
    </div>
  );
};
