import React from 'react';
import { formatApiError } from '../../utils/errors';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { Settings as SettingsType } from '../../api/types';
import { PathInput } from '../../components/PathBrowser';
import { Loader2, Save } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import { SettingsSections } from './SettingsSections';
import { ServiceInfo } from './ServiceInfo';
import StudioSelect from '../../components/StudioSelect';
import { useWorkspaceText } from '../../utils/workspaceText';

export default function Preferences() {
  const { t, i18n } = useTranslation();
  const text = useWorkspaceText();
  const [params] = useSearchParams();
  const appearance = params.get('section') === 'interface';
  const [settings, setSettings] = React.useState<SettingsType | null>(null);
  const [saving, setSaving] = React.useState(false);
  const savingRef = React.useRef(false);
  const [saved, setSaved] = React.useState(false);
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
    apiClient.put<SettingsType>('/settings', settings)
      .then((res) => {
        setSettings(res);
        // 即时生效：语言
        if (res.ui?.language && res.ui.language !== i18n.language) {
          i18n.changeLanguage(res.ui.language);
          localStorage.setItem('i18nextLng', res.ui.language);
        }
        // 即时生效：主题
        document.documentElement.classList.toggle('dark', res.ui.theme === 'dark' || res.ui.theme === 'system' && (window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false));
        window.dispatchEvent(new CustomEvent('studio.settings.changed', { detail: res }));
        setSaved(true);
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

  return <div data-testid="settings-page"><SettingsSections sections={appearance ? [{ id: 'preferences-appearance', label: t('settings.ui') }, { id: 'preferences-service', label: t('settings.server') }] : [{ id: 'preferences-storage', label: t('settings.paths') }]}>
    {error && <div role="alert" className="settings-alert">{error}</div>}
    <fieldset disabled={saving} aria-busy={saving} className="contents">
    {!appearance ? <section id="preferences-storage" data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><div><h2>{t('settings.paths')}</h2><p className="settings-note">{text('训练数据、采样与产物按项目和版本隔离；更改默认路径仅影响新任务。', 'Training data, samples and outputs are isolated by project and version. Path changes apply to new jobs.')}</p></div></div>
      {([['data_root', t('settings.dataRoot')], ['cache_dir', t('settings.cacheDir')], ['models_dir', t('settings.modelsDir')]] as const).map(([key, label]) => <div className="settings-field" key={key}>
        <label htmlFor={`preferences-${key}`}>{label}</label><div className="settings-field-control">
          {key === 'data_root' ? <><input id={`preferences-${key}`} aria-label={label} readOnly value={settings.paths[key]} className="settings-input font-mono opacity-70" /><p className="settings-note">{t('settings.dataRootNote')}</p></> : <PathInput ariaLabel={label} value={settings.paths[key]} onChange={value => update(s => ({ ...s, paths: { ...s.paths, [key]: value } }))} />}
        </div>
      </div>)}
      <div className="settings-field"><label htmlFor="preferences-output-mode">{text('训练产物位置', 'Training output location')}</label><div className="settings-field-control">
        <StudioSelect disabled={saving} id="preferences-output-mode" aria-label={text('训练产物位置', 'Training output location')} value={settings.paths.output_mode === 'custom' ? 'custom' : 'project'}
          options={[{value:'project',label:text('项目版本目录（默认）','Project version directory (default)')},{value:'custom',label:text('自定义输出根目录','Custom output root')}]}
          onValueChange={value => update(s => ({...s,paths:{...s.paths,output_mode:value === 'custom' ? 'custom' : 'project'}}))}/>
        {settings.paths.output_mode === 'custom' ? <><div className="mt-2"><PathInput ariaLabel={t('settings.outputDir')} value={settings.paths.output_dir} onChange={value => update(s => ({...s,paths:{...s.paths,output_dir:value}}))}/></div><p className="settings-note">{text('在此目录下按项目 ID、版本号和任务 ID 分开保存，避免重复训练互相覆盖。','Outputs under this root are separated by project ID, version and job ID.')}</p></>
          : <p className="settings-note font-mono">project/{'<project_id>'}/v1/output/{'<job_id>'}/</p>}
      </div></div>
      <details className="settings-inline-details"><summary>{text('新项目目录结构', 'New project directory layout')}</summary>
        <pre className="text-xs leading-6 overflow-auto">{`studio_data/project/<project_id>/v1/\n  traindata/\n  reg/\n  samples/<job_id>/\n  output/<job_id>/`}</pre>
        <p>{text('显示名称可以使用任意语言；目录使用字母、数字和下划线组成的项目 ID。旧项目保留原目录与文件引用。', 'Display names support any language. Directory IDs use letters, digits and underscores. Existing projects retain their original paths.')}</p>
      </details>
    </section> : <>
      <section id="preferences-appearance" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-section-heading"><div><h2>{t('settings.ui')}</h2><p className="settings-note">{t('settings.uiDesc', '语言与主题在保存后立即生效。')}</p></div></div>
        <div className="settings-field"><label htmlFor="preferences-language">{t('settings.language')}</label><div className="settings-field-control"><StudioSelect disabled={saving} id="preferences-language" aria-label={t('settings.language')} value={settings.ui.language} onValueChange={value => update(s => ({...s,ui:{...s.ui,language:value as SettingsType['ui']['language']}}))} options={[{value:'zh-CN',label:'中文'},{value:'en',label:'English'}]} data-testid="settings-language"/></div></div>
        <div className="settings-field"><label htmlFor="preferences-theme">{t('settings.theme')}</label><div className="settings-field-control"><StudioSelect disabled={saving} id="preferences-theme" aria-label={t('settings.theme')} value={settings.ui.theme} onValueChange={value => update(s => ({...s,ui:{...s.ui,theme:value as SettingsType['ui']['theme']}}))} options={[{value:'system',label:t('settings.themeSystem')},{value:'light',label:t('settings.themeLight')},{value:'dark',label:t('settings.themeDark')}]} data-testid="settings-theme"/></div></div>
      </section>
      <section id="preferences-service" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-section-heading"><div><h2>{t('settings.server')}</h2><p className="settings-note">{t('settings.serverNote')}</p></div></div>
        <div className="settings-field"><span className="settings-field-label">{t('settings.connectedService', '当前连接')}</span><div className="settings-field-control py-1.5 font-mono break-all">{window.location.origin}</div></div>
        <ServiceInfo />
        <div className="settings-field"><label htmlFor="preferences-host">{t('settings.host')}</label><div className="settings-field-control"><input id="preferences-host" type="text" aria-label={t('settings.host')} value={settings.server.host} onChange={event => update(s => ({ ...s, server: { ...s.server, host: event.target.value } }))} className="settings-input font-mono" /></div></div>
        <div className="settings-field"><label htmlFor="preferences-port">{t('settings.port')}</label><div className="settings-field-control"><input id="preferences-port" type="number" min={1} max={65535} aria-label={t('settings.port')} value={settings.server.port} onChange={event => update(s => ({ ...s, server: { ...s.server, port: Number(event.target.value) } }))} className="settings-input font-mono" /></div></div>
      </section>
    </>}
    </fieldset>
    <div className="settings-save"><button onClick={handleSave} disabled={saving} className="settings-action" data-testid="settings-save-btn">{saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}<span>{saved ? t('settings.saved') : saving ? t('settings.saving') : t('settings.save')}</span></button></div>
  </SettingsSections></div>;
}
