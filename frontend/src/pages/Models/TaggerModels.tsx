import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Download, ExternalLink } from 'lucide-react';
import { apiClient } from '../../api/client';
import { formatBytes } from '../../utils/format';
import { formatApiError } from '../../utils/errors';

type Catalog = { id: string; name: string; path: string; ready: boolean; size: number; url: string; license: string };
export default function TaggerModels({ onStarted }: { onStarted: () => Promise<void> }) {
  const location = useLocation();
  const [rows, setRows] = React.useState<Catalog[]>([]);
  const [mirror, setMirror] = React.useState('official');
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState('');
  React.useEffect(() => {
    const refresh = async () => { try { setRows(await apiClient.get<Catalog[]>('/models/catalog', { silent: true })); } catch (e) { setError(formatApiError(e)); } };
    void refresh(); const interval = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(interval);
  }, []);
  return <section id="models-taggers" data-settings-section tabIndex={-1} className="settings-section"><div className="settings-section-heading"><div><h3>本地自动打标模型</h3><p className="settings-note">供数据集自动标注使用，独立于训练模型默认组件。</p></div><Link to="/settings/environment?tab=runtime&package=onnxruntime" replace state={location.state} className="text-xs text-blue-600 dark:text-blue-400">安装 / 检查 ONNX Runtime</Link></div>
    {rows.map(row => <div className="settings-model-component" key={row.id}><div><h3>{row.name}</h3><p className="settings-note">{formatBytes(row.size)} · {row.license}</p></div><div className="space-y-2 min-w-0"><p className="settings-model-path">{row.path}</p><p className="settings-note">model.onnx 与 selected_tags.csv 成套下载并校验后可用。</p><div className="flex flex-wrap items-center gap-2"><select aria-label="打标模型下载来源" value={mirror} onChange={e => setMirror(e.target.value)} className="rounded-md border bg-transparent px-2 py-1.5 text-xs dark:border-slate-600"><option value="official">Hugging Face 官方</option><option value="hf-mirror">HF-Mirror（匿名）</option></select><button disabled={busy || row.ready} className="inline-flex items-center gap-1 rounded-md bg-blue-600 px-3 py-1.5 text-xs text-white disabled:opacity-40" onClick={async () => { setBusy(true); setError(''); try { await apiClient.post(`/models/catalog/${row.id}/download`, { provider: 'huggingface', mirror }); await onStarted(); } catch (e) { setError(formatApiError(e)); } finally { setBusy(false); } }}><Download size={13} />{row.ready ? '已就绪' : busy ? '加入下载中' : '下载打标模型'}</button><a className="inline-flex items-center gap-1 text-xs text-blue-600" href={row.url} target="_blank" rel="noreferrer">发布页<ExternalLink size={12} /></a></div></div></div>)}
    {error && <p role="alert" className="break-words text-xs text-red-600">{error}</p>}
  </section>;
}
