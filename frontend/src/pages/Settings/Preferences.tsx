import DownloadPreferences from './DownloadPreferences';
import NetworkPreferences, { type NetworkSettings } from './NetworkPreferences';
type SettingsType = ApiSettings;

import ServiceControls from '../../components/ServiceControls';
import React from 'react';
import { formatApiError } from '../../utils/errors';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { Settings as ApiSettings } from '../../api/types';
import { PathInput } from '../../components/PathBrowser';
import { Loader2, Save } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import { SettingsSections } from './SettingsSections';
import { ServiceInfo } from './ServiceInfo';
import StudioSelect from '../../components/StudioSelect';
import { useWorkspaceText } from '../../utils/workspaceText';

const defaultNetworkSettings: NetworkSettings = { proxy_mode: 'system', proxy_url: '', proxy_username: '', proxy_password_configured: false };

export default function Preferences() {
  const { t, i18n } = useTranslation();
  const text = useWorkspaceText();
  const [params] = useSearchParams();
  const downloads = params.get('section') === 'downloads';
  const appearance = params.get('section') === 'interface';
  const [settings, setSettings] = React.useState<SettingsType | null>(null);
  const [saving, setSaving] = React.useState(false);
  const savingRef = React.useRef(false);
  const [proxyPassword, setProxyPassword] = React.useState<string | undefined>();
  const [saved, setSaved] = React.useState(false);
  const [serviceRefreshTarget, setServiceRefreshTarget] = React.useState<HTMLSpanElement | null>(null);
  const [serviceRefreshKey, setServiceRefreshKey] = React.useState(0);
  const [error, setError] = React.useState('');

  const load = React.useCallback(async () => {
    setError('');
    try {
      const config = await apiClient.get<SettingsType>('/settings');
      setSettings(config);
    } catch (e) { setError(formatApiError(e)); }
  }, []);

  React.useEffect(() => {
    void load();
    const changed = (event: Event) => {
      const ui = (event as CustomEvent<SettingsType>).detail.ui;
      setSettings(previous => previous ? { ...previous, ui } : previous);
    };
    window.addEventListener('studio.settings.changed', changed);
    return () => window.removeEventListener('studio.settings.changed', changed);
  }, [load]);

  const update = (fn: (s: SettingsType) => SettingsType) => {
    if (savingRef.current) return;
    setSaved(false);
    setSettings((prev) => (prev ? fn({ ...prev, paths: { ...prev.paths }, server: { ...prev.server }, ui: { ...prev.ui } }) : prev));
  };

  const handleSave = () => {
    if (!settings || savingRef.current) return;
    savingRef.current = true;
    setSaving(true); setError('');
    apiClient.put<SettingsType>('/settings', { ...settings, ...(proxyPassword !== undefined ? { network: { ...(settings.network ?? defaultNetworkSettings), proxy_password: proxyPassword } } : {}) })
      .then((res) => {
        setSettings(res);
        setProxyPassword(undefined);
        // 即时生效：语言
        if (res.ui?.language && res.ui.language !== i18n.language) {
          i18n.changeLanguage(res.ui.language);
          localStorage.setItem('i18nextLng', res.ui.language);
        }
        // 即时生效：主题
        document.documentElement.classList.toggle('dark', res.ui.theme === 'dark' || res.ui.theme === 'system' && (window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false));
        window.dispatchEvent(new CustomEvent('studio.settings.changed', { detail: res }));
        setSaved(true);
        setServiceRefreshKey(value=>value+1);
        setTimeout(() => setSaved(false), 2000);
      })
      .catch((e) => setError(formatApiError(e)))
      .finally(() => { savingRef.current = false; setSaving(false); });
  };

  if (!settings) {
    return (
      <div className="flex items-center space-x-2 text-slate-500" data-testid="settings-loading">
        {error ? <div role="alert" className="text-red-600">{error}<button onClick={() => void load()} className="ml-3 underline">{t('common.retry', '重试')}</button></div> : <><Loader2 className="w-4 h-4 animate-spin" /><span>{t('common.loading')}</span></>}
      </div>
    );
  }

  return <div data-testid="settings-page"><SettingsSections sections={downloads ? [{ id: 'preferences-downloads', label: text('软件下载源', 'Package sources') }] : appearance ? [{ id: 'preferences-appearance', label: t('settings.ui') }, { id: 'preferences-service', label: t('settings.server') }, { id: 'preferences-network', label: text('网络代理', 'Network proxy') }] : [{ id: 'preferences-storage', label: t('settings.paths') }]}>
    {error && <div role="alert" className="settings-alert">{error}</div>}
    <fieldset disabled={saving} aria-busy={saving} className="contents">
    {downloads ? <DownloadPreferences value={settings.downloads ?? { pypi: 'ustc', pytorch: 'mirror', fallback: true }} onChange={value => update(s => ({ ...s, downloads: value }))} /> : !appearance ? <section id="preferences-storage" data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><div><h2>{t('settings.paths')}</h2><p className="settings-note">{text('训练数据、采样与产物按项目和版本隔离；更改默认路径仅影响新任务。', 'Training data, samples and outputs are isolated by project and version. Path changes apply to new jobs.')}</p></div></div>
      {([['data_root', t('settings.dataRoot')], ['cache_dir', t('settings.cacheDir')], ['models_dir', t('settings.modelsDir')]] as const).map(([key, label]) => <div className="settings-field" key={key}>
        <label htmlFor={`preferences-${key}`}>{label}</label><div className="settings-field-control">
          {key === 'data_root' ? <><PathInput ariaLabel={label} value={settings.paths[key]} onChange={value => update(s => ({ ...s, paths: { ...s.paths, data_root: value } }))} /><p className="settings-note">{text('保存后写入配置，重启服务后生效；当前服务继续使用现有目录。', 'Saved to configuration and applied after restarting; the current service keeps using its existing directory.')}</p></> : <PathInput ariaLabel={label} value={settings.paths[key]} onChange={value => update(s => ({ ...s, paths: { ...s.paths, [key]: value } }))} />}
          {key === 'cache_dir' && <p className="settings-note">{text('自定义缓存目录用于新生成的训练缓存、扫描索引、缩略图和软件包缓存；已有文件保留在原位置。', 'A custom cache directory applies to new training caches, indexes, thumbnails and package downloads. Existing files stay in place.')}</p>}
        </div>
      </div>)}
      <div className="settings-field"><label>{text('基础环境目录', 'Base environment directory')}</label><div className="settings-field-control">
        <PathInput ariaLabel={text('基础环境目录', 'Base environment directory')} value={settings.paths.bootstrap_env_dir ?? ''} placeholder={text('留空使用源码目录下的默认位置', 'Leave blank for the default source directory location')} onChange={value => update(s => ({...s, paths: {...s.paths, bootstrap_env_dir: value}}))}/>
        <p className="settings-note">{text('按平台分别创建 Python 环境。保存后下次从启动脚本启动时生效；新目录需要安装依赖，旧环境不会搬迁或删除。', 'Creates a Python environment per platform. Applies on the next launcher start; a new location requires installation. Old environments are preserved.')}</p>
      </div></div>
      <div className="settings-field"><label htmlFor="preferences-output-mode">{text('训练产物位置', 'Training output location')}</label><div className="settings-field-control">
        <StudioSelect disabled={saving} id="preferences-output-mode" aria-label={text('训练产物位置', 'Training output location')} value={settings.paths.output_mode === 'custom' ? 'custom' : 'project'}
          options={[{value:'project',label:text('项目版本目录（默认）','Project version directory (default)')},{value:'custom',label:text('自定义输出根目录','Custom output root')}]}
          onValueChange={value => update(s => ({...s,paths:{...s.paths,output_mode:value === 'custom' ? 'custom' : 'project'}}))}/>
        {settings.paths.output_mode === 'custom' ? <><div className="mt-2"><PathInput ariaLabel={t('settings.outputDir')} value={settings.paths.output_dir} onChange={value => update(s => ({...s,paths:{...s.paths,output_dir:value}}))}/></div><p className="settings-note">{text('在此目录下按项目 ID、版本号和任务 ID 分开保存，避免重复训练互相覆盖。','Outputs under this root are separated by project ID, version and job ID.')}</p></>
          : <p className="settings-note font-mono">project/{'<project_id>'}/v1/output/{'<job_id>'}/</p>}
      </div></div>

    </section> : <>
      <section id="preferences-appearance" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-section-heading"><div><h2>{t('settings.ui')}</h2><p className="settings-note">{t('settings.uiDesc', '语言与主题在保存后立即生效。')}</p></div></div>
        <div className="settings-field"><label htmlFor="preferences-language">{t('settings.language')}</label><div className="settings-field-control"><StudioSelect disabled={saving} id="preferences-language" aria-label={t('settings.language')} value={settings.ui.language} onValueChange={value => update(s => ({...s,ui:{...s.ui,language:value as SettingsType['ui']['language']}}))} options={[{value:'zh-CN',label:'中文'},{value:'en',label:'English'}]} data-testid="settings-language"/></div></div>
        <div className="settings-field"><label htmlFor="preferences-theme">{t('settings.theme')}</label><div className="settings-field-control"><StudioSelect disabled={saving} id="preferences-theme" aria-label={t('settings.theme')} value={settings.ui.theme} onValueChange={value => update(s => ({...s,ui:{...s.ui,theme:value as SettingsType['ui']['theme']}}))} options={[{value:'system',label:t('settings.themeSystem')},{value:'light',label:t('settings.themeLight')},{value:'dark',label:t('settings.themeDark')}]} data-testid="settings-theme"/></div></div>
      </section>
      <section id="preferences-service" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-section-heading"><div><h2>{t('settings.server')}</h2><p className="settings-note">{t('settings.serverNote')}</p></div><span className="settings-service-refresh" ref={setServiceRefreshTarget}/></div>
        <div className="settings-field"><span className="settings-field-label">{t('settings.connectedService', '当前连接')}</span><div className="settings-field-control py-1.5 font-mono break-all">{window.location.origin}</div></div>
        <ServiceInfo />
        <div className="settings-field"><label htmlFor="preferences-host">{t('settings.host')}</label><div className="settings-field-control"><input id="preferences-host" type="text" aria-label={t('settings.host')} value={settings.server.host} onChange={event => update(s => ({ ...s, server: { ...s.server, host: event.target.value } }))} className="settings-input font-mono" /></div></div>
        <div className="settings-field"><label htmlFor="preferences-port">{t('settings.port')}</label><div className="settings-field-control"><input id="preferences-port" type="number" min={1} max={65535} aria-label={t('settings.port')} value={settings.server.port} onChange={event => update(s => ({ ...s, server: { ...s.server, port: Number(event.target.value) } }))} className="settings-input font-mono" /></div></div>
      </section>
      <NetworkPreferences value={settings.network ?? defaultNetworkSettings} password={proxyPassword} disabled={saving} onChange={network => update(s => ({ ...s, network }))} onPasswordChange={value => { if (!savingRef.current) { setProxyPassword(value); setSaved(false); } }}/>
    </>}
    </fieldset>
    <div className="settings-save">{appearance && <ServiceControls secondary disabled={saving} refreshTarget={serviceRefreshTarget} refreshKey={serviceRefreshKey}/>}<button onClick={handleSave} disabled={saving} className="settings-action" data-testid="settings-save-btn">{saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}<span>{saved ? text('已保存，请重启服务', 'Saved; restart service to apply') : saving ? t('settings.saving') : t('settings.save')}</span></button></div>
  </SettingsSections></div>;
}
