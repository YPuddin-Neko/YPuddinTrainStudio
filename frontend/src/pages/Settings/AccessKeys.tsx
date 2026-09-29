import React from 'react';
import { ExternalLink, Loader2 } from 'lucide-react';
import { useLocation } from 'react-router-dom';
import { apiClient } from '../../api/client';
import { useWorkspaceText } from '../../utils/workspaceText';
import { SettingsSections } from './SettingsSections';
import './access-keys.css';

export type CredentialProvider = 'huggingface' | 'modelscope' | 'danbooru' | 'gelbooru' | 'e621' | 'rule34';
export type CredentialStates = Record<CredentialProvider, { configured: boolean }>;
const providers: { id: CredentialProvider; name: string; url: string; account?: 'username' | 'user_id' }[] = [
  { id: 'huggingface', name: 'Hugging Face', url: 'https://huggingface.co/settings/tokens' },
  { id: 'modelscope', name: 'ModelScope', url: 'https://modelscope.cn/my/myaccesstoken' },
  { id: 'danbooru', name: 'Danbooru', url: 'https://danbooru.donmai.us/profile', account: 'username' },
  { id: 'gelbooru', name: 'Gelbooru', url: 'https://gelbooru.com/index.php?page=account&s=options', account: 'user_id' },
  { id: 'e621', name: 'e621', url: 'https://e621.net/users/home', account: 'username' },
  { id: 'rule34', name: 'Rule34', url: 'https://rule34.xxx/index.php?page=account&s=options', account: 'user_id' },
];
const emptyDraft = () => Object.fromEntries(providers.map(p => [p.id, { account: '', secret: '' }])) as Record<CredentialProvider, { account: string; secret: string }>;

export default function AccessKeys() {
  const text = useWorkspaceText();
  const location = useLocation();
  const [status, setStatus] = React.useState<CredentialStates | null>(null);
  const [draft, setDraft] = React.useState(emptyDraft);
  const [busy, setBusy] = React.useState<CredentialProvider | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [statusError, setStatusError] = React.useState('');
  const [feedback, setFeedback] = React.useState<Partial<Record<CredentialProvider, { error?: string; notice?: string }>>>({});
  const mounted = React.useRef(false);
  const loadError = text('无法读取密钥配置状态，请重试。', 'Could not load access-key status. Retry.');
  // Render only locally defined errors: even an older/misconfigured server may echo
  // submitted tokens (including encoded forms) in its error message or details.
  const safeError = (failure: unknown) => {
    const code = failure && typeof failure === 'object' && 'status' in failure ? failure.status : undefined;
    if (code === 422) return text('账号或密钥格式不正确，请检查后重试。', 'Check the account and key format, then retry.');
    if (code === 503) return text('凭据文件暂时不可读写，请检查服务数据目录权限后重试。', 'The credentials file is unavailable. Check service data directory permissions and retry.');
    return text('操作失败，请重试。', 'The request failed. Please retry.');
  };
  const refresh = React.useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    try {
      const result = await apiClient.get<CredentialStates>('/credentials', { silent: true, signal });
      if (mounted.current && !signal?.aborted) { setStatus(result); setStatusError(''); }
    } catch {
      if (mounted.current && !signal?.aborted) setStatusError(loadError);
    } finally { if (mounted.current && !signal?.aborted) setLoading(false); }
  }, [loadError]);
  React.useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    void refresh(controller.signal);
    return () => { mounted.current = false; controller.abort(); };
  }, [refresh]);
  React.useEffect(() => {
    const id = location.hash.slice(1);
    if (providers.some(p => id === `credentials-${p.id}`)) {
      const target = document.getElementById(id);
      target?.scrollIntoView?.({ block: 'start' }); target?.focus({ preventScroll: true });
    }
  }, [location.hash]);
  const save = async (provider: typeof providers[number], clear = false) => {
    setBusy(provider.id); setFeedback(old => ({ ...old, [provider.id]: {} }));
    const value = draft[provider.id];
    const payload = provider.account ? { [provider.account]: value.account.trim(), api_key: value.secret.trim() } : { token: value.secret.trim() };
    try {
      const result = clear
        ? await apiClient.delete<{ configured: boolean }>(`/credentials/${provider.id}`, { silent: true })
        : await apiClient.put<{ configured: boolean }>(`/credentials/${provider.id}`, payload, { silent: true });
      if (mounted.current) {
        setStatus(old => old ? { ...old, [provider.id]: result } : old);
        setDraft(old => ({ ...old, [provider.id]: { account: '', secret: '' } }));
        const notice = clear ? text('已清除', 'Cleared') : text('已保存', 'Saved');
        setFeedback(old => ({ ...old, [provider.id]: { notice } }));
      }
      window.dispatchEvent(new Event('credentials.changed'));
    } catch (failure) { if (mounted.current) setFeedback(old => ({ ...old, [provider.id]: { error: safeError(failure) } })); }
    finally { if (mounted.current) setBusy(null); }
  };
  return <div className="access-keys" data-testid="access-keys-settings"><SettingsSections sections={providers.map(p => ({ id: `credentials-${p.id}`, label: p.name }))}>
    {statusError && <div role="alert" className="settings-alert">{statusError}<button type="button" className="ui-link ml-2" disabled={loading || !!busy} onClick={() => void refresh()}>{text('重试读取', 'Retry status')}</button></div>}
    {providers.map(provider => <section key={provider.id} id={`credentials-${provider.id}`} data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><div><h2>{provider.name}</h2><p className="settings-note">{provider.account ? text('用于从此站点下载训练图和正则图。', 'Downloads training and regularization images from this site.') : text('用于官方模型下载；受限仓库需先取得权限。', 'Official model downloads; gated repositories require access approval.')}</p></div><span role="status" className="settings-note access-key-state">{status ? status[provider.id]?.configured ? text('已配置', 'Configured') : text('未配置', 'Not configured') : loading ? text('读取状态中', 'Loading status') : text('状态不可用', 'Status unavailable')}</span></div>
      <form className="access-key-form" onSubmit={event => { event.preventDefault(); void save(provider); }}>
        {provider.account && <div className="settings-field"><label htmlFor={`account-${provider.id}`}>{provider.account === 'user_id' ? text('用户 ID', 'User ID') : text('用户名', 'Username')}</label><div className="settings-field-control"><input id={`account-${provider.id}`} aria-label={`${provider.name} ${provider.account === 'user_id' ? text('用户 ID', 'user ID') : text('用户名', 'username')}`} className="settings-input" autoComplete="off" value={draft[provider.id].account} maxLength={provider.account === 'user_id' ? 20 : 200} pattern={provider.account === 'user_id' ? '[0-9]+' : undefined} required disabled={!!busy} onChange={event => setDraft(old => ({ ...old, [provider.id]: { ...old[provider.id], account: event.target.value } }))} placeholder={text('保存或更换密钥时填写', 'Enter when saving or replacing a key')} /></div></div>}
        <div className="settings-field"><label htmlFor={`token-${provider.id}`}>{provider.account ? 'API Key' : text('访问令牌', 'Access token')}</label><div className="settings-field-control"><input id={`token-${provider.id}`} aria-label={`${provider.name} ${provider.account ? 'API Key' : text('访问令牌', 'access token')}`} className="settings-input" type="password" autoComplete="new-password" spellCheck={false} required maxLength={4096} disabled={!!busy} value={draft[provider.id].secret} onChange={event => setDraft(old => ({ ...old, [provider.id]: { ...old[provider.id], secret: event.target.value } }))} placeholder={text('输入新密钥；已保存值不会回显', 'Enter a new key; stored values are never displayed')} /></div></div>
        <div className="access-key-actions"><button type="submit" className="ui-btn ui-btn-primary" disabled={!!busy || !status || !draft[provider.id].secret.trim() || (!!provider.account && !draft[provider.id].account.trim())}>{busy === provider.id && <Loader2 size={14} className="animate-spin" />}{text('保存', 'Save')}</button><button type="button" className="ui-btn" disabled={!!busy || !status?.[provider.id]?.configured} onClick={() => void save(provider, true)}>{text('清除', 'Clear')}</button><a href={provider.url} target="_blank" rel="noreferrer" className="ui-link">{text('账号密钥页面', 'Account key page')}<ExternalLink size={12} /></a></div>
        {feedback[provider.id]?.error && <div role="alert" className="settings-alert access-key-feedback">{feedback[provider.id]?.error}</div>}
        {feedback[provider.id]?.notice && <p role="status" className="access-key-feedback access-key-notice">{feedback[provider.id]?.notice}</p>}
      </form>
    </section>)}
    <p className="settings-note">{text('密钥保存在服务器，仅用于对应来源。修改后对新任务生效。', 'Keys are stored on the server and used only for their source. Changes apply to new tasks.')}</p>
  </SettingsSections></div>;
}
