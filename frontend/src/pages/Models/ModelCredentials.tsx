import React from 'react';
import { ExternalLink, Loader2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import { formatApiError } from '../../utils/errors';

type Provider = 'huggingface' | 'modelscope';
type State = Record<Provider, { configured: boolean }>;
const providers: { id: Provider; name: string; url: string }[] = [
  { id: 'huggingface', name: 'Hugging Face', url: 'https://huggingface.co/settings/tokens' },
  { id: 'modelscope', name: '魔搭 ModelScope', url: 'https://modelscope.cn/my/myaccesstoken' },
];

export default function ModelCredentials() {
  const [status, setStatus] = React.useState<State | null>(null);
  const [draft, setDraft] = React.useState<Record<Provider, string>>({ huggingface: '', modelscope: '' });
  const [busy, setBusy] = React.useState<Provider | null>(null);
  const [error, setError] = React.useState('');
  const [notice, setNotice] = React.useState('');
  const refresh = React.useCallback(async () => {
    try { setStatus(await apiClient.get<State>('/models/credentials')); setError(''); }
    catch (e) { setError(formatApiError(e)); }
  }, []);
  React.useEffect(() => { void refresh(); }, [refresh]);
  const save = async (provider: Provider, clear = false) => {
    setBusy(provider); setError(''); setNotice('');
    try {
      const result = clear ? await apiClient.delete<{ configured: boolean }>(`/models/credentials/${provider}`, { silent: true })
        : await apiClient.put<{ configured: boolean }>(`/models/credentials/${provider}`, { token: draft[provider].trim() }, { silent: true });
      setStatus(old => old ? { ...old, [provider]: result } : null);
      setDraft(old => ({ ...old, [provider]: '' }));
      setNotice(clear ? '已清除，此来源后续使用匿名下载。' : '令牌已保存，将用于后续官方来源下载。');
      await refresh();
    } catch (e) {
      // Never render an HTTP response that could echo a submitted secret from an older server.
      const token = draft[provider].trim();
      setError(token ? formatApiError(e).split(token).join('[已隐藏]') : formatApiError(e));
    } finally { setBusy(null); }
  };
  return <section id="models-credentials" data-settings-section tabIndex={-1} className="settings-section" data-testid="model-credentials">
    <div className="settings-section-heading"><div><h3>下载令牌</h3><p className="settings-note">公开模型可匿名下载。受限仓库须先在发布页取得访问权限。</p></div></div>
    {providers.map(provider => <form key={provider.id} className="settings-field" onSubmit={e => { e.preventDefault(); void save(provider.id); }}>
      <label htmlFor={`token-${provider.id}`} className="settings-field-label">{provider.name}<span className="mt-1 block text-xs text-slate-500">{status ? status[provider.id].configured ? '已配置' : '未配置' : '读取状态中'}</span></label>
      <div className="min-w-0 space-y-2"><div className="flex flex-wrap gap-2"><input id={`token-${provider.id}`} type="password" autoComplete="new-password" value={draft[provider.id]} onChange={e => setDraft(old => ({ ...old, [provider.id]: e.target.value }))} placeholder="输入新令牌以替换；已保存值不会回显" maxLength={4096} className="min-w-0 flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-[13px] dark:border-slate-600 dark:bg-slate-900" />
        <button type="submit" disabled={!!busy || !draft[provider.id].trim()} className="rounded-md bg-blue-600 px-3 py-2 text-xs text-white disabled:opacity-40">{busy === provider.id ? <Loader2 size={15} className="animate-spin" /> : '保存'}</button>
        <button type="button" disabled={!!busy || !status?.[provider.id].configured} onClick={() => void save(provider.id, true)} className="rounded-md border px-3 py-2 text-xs disabled:opacity-40 dark:border-slate-600">清除</button></div>
        <a href={provider.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-xs text-blue-600 dark:text-blue-400">获取访问令牌<ExternalLink size={12} /></a>
      </div>
    </form>)}
    <p className="settings-note">令牌保存在运行服务的电脑，仅发送给对应官方来源；HF-Mirror 使用匿名下载。清除后不会继续使用环境变量或 CLI 中的同源令牌。正在进行的下载保留启动时凭证。</p>
    {error && <p role="alert" className="break-words text-xs text-red-600">{error}<button onClick={() => void refresh()} className="ml-2 underline">重试</button></p>}
    {notice && <p role="status" className="text-xs text-emerald-600">{notice}</p>}
  </section>;
}
