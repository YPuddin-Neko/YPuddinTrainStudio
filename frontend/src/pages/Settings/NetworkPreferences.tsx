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
    <div className="settings-section-heading"><div><h2>{text('网络代理', 'Network proxy')}</h2><p className="settings-note">{text('用于模型下载、环境包下载和正则图片获取。保存后，新发起的请求使用此设置。', 'Used for model downloads, environment packages and regularization images. New requests use these settings after saving.')}</p></div></div>
    <div className="settings-field"><label htmlFor="preferences-proxy-mode">{text('连接方式', 'Connection mode')}</label><div className="settings-field-control">
      <StudioSelect id="preferences-proxy-mode" aria-label={text('连接方式', 'Connection mode')} disabled={disabled} value={value.proxy_mode} onValueChange={proxy_mode => onChange({ ...value, proxy_mode: proxy_mode as NetworkSettings['proxy_mode'] })} options={[
        { value: 'system', label: text('沿用服务器代理', 'Use server proxy settings') },
        { value: 'direct', label: text('直接连接', 'Direct connection') },
        { value: 'custom', label: text('自定义代理', 'Custom proxy') },
      ]}/>
      <p className="settings-note">{value.proxy_mode === 'system' ? text('沿用训练服务器现有的代理设置。需要为所有下载指定同一个代理地址时，请选择“自定义代理”。', 'Use the training server’s existing proxy settings. Choose Custom proxy to use one specified address for all downloads.') : value.proxy_mode === 'direct' ? text('新请求不经过代理。', 'New requests bypass proxies.') : text('请填写训练服务器能连接的代理地址。远程部署时，127.0.0.1 指服务器，不是当前浏览器所在的电脑。', 'Enter a proxy reachable from the training server. With remote deployment, 127.0.0.1 refers to the server, not the computer running this browser.')}</p>
    </div></div>
    {value.proxy_mode === 'custom' && <>
      <div className="settings-field"><label htmlFor="preferences-proxy-url">{text('代理地址', 'Proxy address')}</label><div className="settings-field-control"><input id="preferences-proxy-url" className="settings-input font-mono" type="url" placeholder="http://127.0.0.1:7890" value={value.proxy_url} onChange={event => onChange({ ...value, proxy_url: event.target.value })} autoComplete="off"/><p className="settings-note">{text('支持 HTTP / HTTPS 代理，账号和密码请填写在下方。', 'Supports HTTP / HTTPS proxies. Enter credentials below.')}</p></div></div>
      <div className="settings-field"><label htmlFor="preferences-proxy-user">{text('代理账号（可选）', 'Proxy username (optional)')}</label><div className="settings-field-control"><input id="preferences-proxy-user" className="settings-input" value={value.proxy_username} onChange={event => onChange({ ...value, proxy_username: event.target.value })} autoComplete="off"/></div></div>
      <div className="settings-field"><label htmlFor="preferences-proxy-password">{text('代理密码（可选）', 'Proxy password (optional)')}</label><div className="settings-field-control"><input id="preferences-proxy-password" className="settings-input" type="password" value={password ?? ''} placeholder={value.proxy_password_configured && password === undefined ? text('已保存；不填写则保持原密码', 'Saved; leave unchanged to keep it') : ''} autoComplete="new-password" onChange={event => onPasswordChange(event.target.value)}/>{value.proxy_password_configured && <button type="button" className="settings-link mt-2" onClick={() => onPasswordChange('')}>{password === '' ? text('保存时清除密码', 'Password will be cleared on save') : text('清除已保存密码', 'Clear saved password')}</button>}</div></div>
    </>}
  </section>;
}
