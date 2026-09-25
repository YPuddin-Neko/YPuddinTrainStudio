import StudioSelect from '../../components/StudioSelect';
import { useWorkspaceText } from '../../utils/workspaceText';

export interface NetworkSettings {
  proxy_mode: 'system' | 'direct' | 'custom';
  proxy_url: string;
  proxy_username: string;
  proxy_password_configured: boolean;
}

export default function NetworkPreferences({ value, password, disabled, onChange, onPasswordChange }: {
  value: NetworkSettings; password?: string; disabled: boolean;
  onChange: (value: NetworkSettings) => void; onPasswordChange: (value: string) => void;
}) {
  const text = useWorkspaceText();
  return <section id="preferences-network" data-settings-section tabIndex={-1} className="settings-section">
    <div className="settings-section-heading"><div><h2>{text('网络代理', 'Network proxy')}</h2><p className="settings-note">{text('保存后用于新的下载请求。', 'Applies to new downloads after saving.')}</p></div></div>
    <div className="settings-field"><label htmlFor="preferences-proxy-mode">{text('连接方式', 'Connection mode')}</label><div className="settings-field-control">
      <StudioSelect id="preferences-proxy-mode" aria-label={text('连接方式', 'Connection mode')} disabled={disabled} value={value.proxy_mode} onValueChange={proxy_mode => onChange({ ...value, proxy_mode: proxy_mode as NetworkSettings['proxy_mode'] })} options={[
        { value: 'system', label: text('沿用服务器代理', 'Use server proxy settings') },
        { value: 'direct', label: text('直接连接', 'Direct connection') },
        { value: 'custom', label: text('自定义代理', 'Custom proxy') },
      ]}/>
      {value.proxy_mode === 'custom' && <p className="settings-note">{text('代理地址须能从服务器访问；127.0.0.1 指服务器本机。', 'Use a proxy reachable from the server; 127.0.0.1 refers to the server itself.')}</p>}
    </div></div>
    {value.proxy_mode === 'custom' && <>
      <div className="settings-field"><label htmlFor="preferences-proxy-url">{text('代理地址', 'Proxy address')}</label><div className="settings-field-control"><input id="preferences-proxy-url" className="settings-input font-mono" type="url" placeholder="http://127.0.0.1:7890" value={value.proxy_url} onChange={event => onChange({ ...value, proxy_url: event.target.value })} autoComplete="off"/><p className="settings-note">{text('支持 HTTP / HTTPS。', 'Supports HTTP / HTTPS.')}</p></div></div>
      <div className="settings-field"><label htmlFor="preferences-proxy-user">{text('代理账号（可选）', 'Proxy username (optional)')}</label><div className="settings-field-control"><input id="preferences-proxy-user" className="settings-input" value={value.proxy_username} onChange={event => onChange({ ...value, proxy_username: event.target.value })} autoComplete="off"/></div></div>
      <div className="settings-field"><label htmlFor="preferences-proxy-password">{text('代理密码（可选）', 'Proxy password (optional)')}</label><div className="settings-field-control"><input id="preferences-proxy-password" className="settings-input" type="password" value={password ?? ''} placeholder={value.proxy_password_configured && password === undefined ? text('已保存；不填写则保持原密码', 'Saved; leave unchanged to keep it') : ''} autoComplete="new-password" onChange={event => onPasswordChange(event.target.value)}/>{value.proxy_password_configured && <button type="button" className="ui-link mt-2" onClick={() => onPasswordChange('')}>{password === '' ? text('保存时清除密码', 'Password will be cleared on save') : text('清除已保存密码', 'Clear saved password')}</button>}</div></div>
    </>}
  </section>;
}
