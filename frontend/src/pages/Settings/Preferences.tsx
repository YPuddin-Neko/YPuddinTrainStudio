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
import Switch from '../../components/Switch';
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
  const loadedSettings = React.useRef<SettingsType | null>(null);
  const initialSettings = React.useRef<SettingsType | null>(null);
  const [serviceRefreshTarget, setServiceRefreshTarget] = React.useState<HTMLSpanElement | null>(null);
  const [serviceRefreshKey, setServiceRefreshKey] = React.useState(0);
  const [error, setError] = React.useState('');

  const load = React.useCallback(async () => {
    setError('');
    try {
      const config = await apiClient.get<SettingsType>('/settings');
      setSettings(config);
      loadedSettings.current = config;
      initialSettings.current = config;
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
        loadedSettings.current = res;
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
        {error ? <div role="alert" className="text-red-600">{error}<button type="button" onClick={() => void load()} className="ui-link ml-3">{t('common.retry', '重试')}</button></div> : <><Loader2 className="w-4 h-4 animate-spin" /><span>{t('common.loading')}</span></>}
      </div>
    );
  }

  const changed = (section: 'paths' | 'server', key: string) => {
    const get = (config: SettingsType | null) => (config?.[section] as Record<string, unknown> | undefined)?.[key] ?? (key === 'open_browser' ? true : '');
    return { edited: get(settings) !== get(initialSettings.current), unsaved: get(settings) !== get(loadedSettings.current) };
  };
  const changeNotice = (section: 'paths' | 'server', key: string, launcher = false) => {
    const state = changed(section, key);
    if (!state.edited && !state.unsaved) return null;
    return <p className="settings-note" role="status">{launcher
      ? state.unsaved ? text('保存后，下次从启动脚本启动时生效。', 'Save to apply on the next launcher start.') : text('已保存，下次从启动脚本启动时生效。', 'Saved; applies on the next launcher start.')
      : state.unsaved ? text('保存后需重启服务生效。', 'Save and restart the service to apply.') : text('已保存，重启服务后生效。', 'Saved; restart the service to apply.')}</p>;
  };
  const restartPending = changed('paths', 'data_root').edited || changed('server', 'host').edited || changed('server', 'port').edited;

  return <div data-testid="settings-page"><SettingsSections sections={downloads ? [{ id: 'preferences-downloads', label: text('软件下载源', 'Package sources') }] : appearance ? [{ id: 'preferences-appearance', label: t('settings.ui') }, { id: 'preferences-service', label: t('settings.server') }, { id: 'preferences-network', label: text('网络代理', 'Network proxy') }] : [{ id: 'preferences-storage', label: t('settings.paths') }]}>
    {error && <div role="alert" className="settings-alert">{error}</div>}
    <fieldset disabled={saving} aria-busy={saving} className="contents">
    {downloads ? <DownloadPreferences value={settings.downloads ?? { pypi: 'ustc', pytorch: 'mirror', fallback: true }} onChange={value => update(s => ({ ...s, downloads: value }))} /> : !appearance ? <section id="preferences-storage" data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><div><h2>{t('settings.paths')}</h2><p className="settings-note">{text('更改路径不会移动已有文件。', 'Changing paths does not move existing files.')}</p></div></div>
      {([['data_root', t('settings.dataRoot')], ['cache_dir', t('settings.cacheDir')], ['models_dir', t('settings.modelsDir')]] as const).map(([key, label]) => <div className="settings-field" key={key}>
        <label htmlFor={`preferences-${key}`}>{label}</label><div className="settings-field-control">
          <PathInput directoryOnly allowMissingDirectory ariaLabel={label} value={settings.paths[key]} onChange={value => update(s => ({ ...s, paths: { ...s.paths, [key]: value } }))} />
          {key === 'data_root' && <><p className="settings-note">{text('保存项目、数据集、任务记录和服务设置。', 'Stores projects, datasets, job history and service settings.')}</p>{changeNotice('paths', key)}</>}
          {key === 'cache_dir' && <p className="settings-note">{text('用于训练、缩略图和安装包缓存。', 'Stores training, thumbnail and package caches.')}</p>}
          {key === 'models_dir' && <p className="settings-note">{text('保存下载的模型及其组件；本地模型也可从其他目录登记。', 'Stores downloaded models and their components. Local models can also be registered from other folders.')}</p>}
        </div>
      </div>)}
      <div className="settings-field"><label>{text('基础环境目录', 'Base environment directory')}</label><div className="settings-field-control">
        <PathInput directoryOnly allowMissingDirectory ariaLabel={text('基础环境目录', 'Base environment directory')} value={settings.paths.bootstrap_env_dir ?? ''} placeholder={text('留空使用源码目录下的默认位置', 'Leave blank for the default source directory location')} onChange={value => update(s => ({...s, paths: {...s.paths, bootstrap_env_dir: value}}))}/>
        <p className="settings-note">{text('存放启动脚本管理的 Python 环境和依赖。', 'Stores Python environments and dependencies managed by the launcher.')}</p>
        {changeNotice('paths', 'bootstrap_env_dir', true)}
        {changed('paths', 'bootstrap_env_dir').edited && <p className="settings-note">{text('新目录需安装依赖，旧环境保留。', 'The new directory needs dependencies; the old environment is retained.')}</p>}
      </div></div>
      <div className="settings-field"><label htmlFor="preferences-output-mode">{text('训练产物位置', 'Training output location')}</label><div className="settings-field-control">
        <StudioSelect disabled={saving} id="preferences-output-mode" aria-label={text('训练产物位置', 'Training output location')} value={settings.paths.output_mode === 'custom' ? 'custom' : 'project'}
          options={[{value:'project',label:text('项目版本目录（默认）','Project version directory (default)')},{value:'custom',label:text('自定义输出根目录','Custom output root')}]}
          onValueChange={value => update(s => ({...s,paths:{...s.paths,output_mode:value === 'custom' ? 'custom' : 'project'}}))}/>
        {settings.paths.output_mode === 'custom' ? <><div className="mt-2"><PathInput directoryOnly allowMissingDirectory ariaLabel={t('settings.outputDir')} value={settings.paths.output_dir} onChange={value => update(s => ({...s,paths:{...s.paths,output_dir:value}}))}/></div><p className="settings-note">{text('按项目、版本和训练任务分别保存。','Outputs are organized by project, version and training run.')}</p></>
          : <p className="settings-note font-mono">project/{'<project_id>'}/v1/output/{'<job_id>'}/</p>}
      </div></div>

      {([
        ['state_dir',text('恢复点目录','Recovery point directory'),text('保存完整训练状态，留空随训练产物保存。','Stores complete training state; blank keeps it with training outputs.')],
        ['samples_dir',text('采样图目录','Sample image directory'),text('保存训练预览图，留空使用各项目版本的 samples 目录。','Stores training previews; blank uses each project version’s samples folder.')],
        ['logs_dir',text('日志目录','Log directory'),text('保存控制台日志、事件记录和 TensorBoard，留空随训练产物保存。','Stores console logs, events and TensorBoard data; blank keeps them with training outputs.')],
      ] as const).map(([key,label,purpose])=><div className="settings-field" key={key}>
        <label>{label}</label><div className="settings-field-control">
          <PathInput directoryOnly allowMissingDirectory ariaLabel={label} value={settings.paths[key] || ''} placeholder={text('使用默认目录','Use default directory')}
            resolveDefaultPath={async()=> (await apiClient.get<{path:string}>('/fs/browse-root',{params:{field:`settings.${key}`},silent:true})).path}
            onChange={value=>update(s=>({...s,paths:{...s.paths,[key]:value}}))}/>
          <p className="settings-note">{purpose}</p>
          {(changed('paths',key).edited || changed('paths',key).unsaved) && <p className="settings-note" role="status">{changed('paths',key).unsaved
            ? text('保存后对新建任务生效；已有文件保留原位置。','Save to apply to new jobs; existing files stay in place.')
            : text('已保存，新建任务使用此目录。','Saved. New jobs use this directory.')}</p>}
        </div>
      </div>)}

    </section> : <>
      <section id="preferences-appearance" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-section-heading"><div><h2>{t('settings.ui')}</h2><p className="settings-note">{t('settings.uiDesc', '语言与主题在保存后立即生效。')}</p></div></div>
        <div className="settings-field"><label htmlFor="preferences-language">{t('settings.language')}</label><div className="settings-field-control"><StudioSelect disabled={saving} id="preferences-language" aria-label={t('settings.language')} value={settings.ui.language} onValueChange={value => update(s => ({...s,ui:{...s.ui,language:value as SettingsType['ui']['language']}}))} options={[{value:'zh-CN',label:'中文'},{value:'en',label:'English'}]} data-testid="settings-language"/></div></div>
        <div className="settings-field"><label htmlFor="preferences-theme">{t('settings.theme')}</label><div className="settings-field-control"><StudioSelect disabled={saving} id="preferences-theme" aria-label={t('settings.theme')} value={settings.ui.theme} onValueChange={value => update(s => ({...s,ui:{...s.ui,theme:value as SettingsType['ui']['theme']}}))} options={[{value:'system',label:t('settings.themeSystem')},{value:'light',label:t('settings.themeLight')},{value:'dark',label:t('settings.themeDark')}]} data-testid="settings-theme"/></div></div>
        <div className="settings-field"><label htmlFor="preferences-open-browser">{text('启动时打开浏览器', 'Open browser on startup')}</label><div className="settings-field-control">
          <Switch id="preferences-open-browser" aria-label={text('启动时打开浏览器', 'Open browser on startup')} checked={settings.server.open_browser !== false} onCheckedChange={open_browser => update(s => ({...s, server: {...s.server, open_browser}}))}>{settings.server.open_browser !== false ? text('已开启', 'Enabled') : text('未开启', 'Disabled')}</Switch>
          {changeNotice('server', 'open_browser', true)}
        </div></div>
      </section>
      <section id="preferences-service" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-section-heading"><div><h2>{t('settings.server')}</h2></div><span className="settings-service-refresh" ref={setServiceRefreshTarget}/></div>
        <div className="settings-field"><span className="settings-field-label">{t('settings.connectedService', '当前连接')}</span><div className="settings-field-control py-1.5 font-mono break-all">{window.location.origin}</div></div>
        <ServiceInfo />
        <div className="settings-field"><label htmlFor="preferences-host">{t('settings.host')}</label><div className="settings-field-control"><input id="preferences-host" type="text" aria-label={t('settings.host')} value={settings.server.host} onChange={event => update(s => ({ ...s, server: { ...s.server, host: event.target.value } }))} className="settings-input font-mono" />{changeNotice('server', 'host')}</div></div>
        <div className="settings-field"><label htmlFor="preferences-port">{t('settings.port')}</label><div className="settings-field-control"><input id="preferences-port" type="number" min={1} max={65535} aria-label={t('settings.port')} value={settings.server.port} onChange={event => update(s => ({ ...s, server: { ...s.server, port: Number(event.target.value) } }))} className="settings-input font-mono" />{changeNotice('server', 'port')}</div></div>
      </section>
      <NetworkPreferences value={settings.network ?? defaultNetworkSettings} password={proxyPassword} disabled={saving} onChange={network => update(s => ({ ...s, network }))} onPasswordChange={value => { if (!savingRef.current) { setProxyPassword(value); setSaved(false); } }}/>
    </>}
    </fieldset>
    <div className="settings-save">{(appearance || !downloads && restartPending) && <ServiceControls secondary applySavedAddress disabled={saving} refreshTarget={serviceRefreshTarget} refreshKey={serviceRefreshKey} onRestarted={() => {
      const active = initialSettings.current, saved = loadedSettings.current;
      if (active && saved) initialSettings.current = {...active, paths:{...active.paths,data_root:saved.paths.data_root},server:{...active.server,host:saved.server.host,port:saved.server.port}};
      setServiceRefreshKey(value => value + 1);
    }}/>}<span role="status" className="settings-note">{saved && text('已保存', 'Saved')}</span><button type="button" onClick={handleSave} disabled={saving} className="ui-btn ui-btn-primary" data-testid="settings-save-btn">{saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}<span>{saving ? t('settings.saving') : t('settings.save')}</span></button></div>
  </SettingsSections></div>;
}
