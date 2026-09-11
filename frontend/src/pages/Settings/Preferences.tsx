import React from 'react';
import { formatApiError } from '../../utils/errors';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { Settings as SettingsType } from '../../api/types';
import { PathInput } from '../../components/PathBrowser';
import { Loader2, Save } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import { SettingsSections } from './SettingsSections';

export default function Preferences() {
  const { t, i18n } = useTranslation();
  const [params] = useSearchParams();
  const appearance = params.get('section') === 'interface';
  const [settings, setSettings] = React.useState<SettingsType | null>(null);
  const [saving, setSaving] = React.useState(false);
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
  }, [load]);

  const update = (fn: (s: SettingsType) => SettingsType) => {
    setSaved(false);
    setSettings((prev) => (prev ? fn({ ...prev, paths: { ...prev.paths }, server: { ...prev.server }, ui: { ...prev.ui } }) : prev));
  };

  const handleSave = () => {
    if (!settings) return;
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
      .finally(() => setSaving(false));
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
    {!appearance ? <section id="preferences-storage" data-settings-section tabIndex={-1} className="settings-section">
      <div className="settings-section-heading"><div><h2>{t('settings.paths')}</h2><p className="settings-note">{t('settings.newJobsPaths')}</p></div></div>
      {([['data_root', t('settings.dataRoot')], ['cache_dir', t('settings.cacheDir')], ['models_dir', t('settings.modelsDir')], ['output_dir', t('settings.outputDir')]] as const).map(([key, label]) => <div className="settings-field" key={key}>
        <label htmlFor={`preferences-${key}`}>{label}</label><div className="settings-field-control">
          {key === 'data_root' ? <><input id={`preferences-${key}`} aria-label={label} readOnly value={settings.paths[key]} className="settings-input font-mono opacity-70" /><p className="settings-note">{t('settings.dataRootNote')}</p></> : <PathInput ariaLabel={label} value={settings.paths[key]} onChange={value => update(s => ({ ...s, paths: { ...s.paths, [key]: value } }))} />}
        </div>
      </div>)}
    </section> : <>
      <section id="preferences-appearance" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-section-heading"><div><h2>{t('settings.ui')}</h2><p className="settings-note">{t('settings.uiDesc', '语言与主题在保存后立即生效。')}</p></div></div>
        <div className="settings-field"><label htmlFor="preferences-language">{t('settings.language')}</label><div className="settings-field-control"><select id="preferences-language" aria-label={t('settings.language')} value={settings.ui.language} onChange={event => update(s => ({ ...s, ui: { ...s.ui, language: event.target.value as SettingsType['ui']['language'] } }))} className="settings-input" data-testid="settings-language"><option value="zh-CN">中文</option><option value="en">English</option></select></div></div>
        <div className="settings-field"><label htmlFor="preferences-theme">{t('settings.theme')}</label><div className="settings-field-control"><select id="preferences-theme" aria-label={t('settings.theme')} value={settings.ui.theme} onChange={event => update(s => ({ ...s, ui: { ...s.ui, theme: event.target.value as SettingsType['ui']['theme'] } }))} className="settings-input" data-testid="settings-theme"><option value="system">{t('settings.themeSystem')}</option><option value="light">{t('settings.themeLight')}</option><option value="dark">{t('settings.themeDark')}</option></select></div></div>
      </section>
      <section id="preferences-service" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-section-heading"><div><h2>{t('settings.server')}</h2><p className="settings-note">{t('settings.serverNote')}</p></div></div>
        <div className="settings-field"><span className="settings-field-label">{t('settings.connectedService', '当前连接')}</span><div className="settings-field-control py-1.5 font-mono break-all">{window.location.origin}</div></div>
        <div className="settings-field"><label htmlFor="preferences-host">{t('settings.host')}</label><div className="settings-field-control"><input id="preferences-host" type="text" aria-label={t('settings.host')} value={settings.server.host} onChange={event => update(s => ({ ...s, server: { ...s.server, host: event.target.value } }))} className="settings-input font-mono" /></div></div>
        <div className="settings-field"><label htmlFor="preferences-port">{t('settings.port')}</label><div className="settings-field-control"><input id="preferences-port" type="number" min={1} max={65535} aria-label={t('settings.port')} value={settings.server.port} onChange={event => update(s => ({ ...s, server: { ...s.server, port: Number(event.target.value) } }))} className="settings-input font-mono" /></div></div>
      </section>
    </>}
    <div className="settings-save"><button onClick={handleSave} disabled={saving} className="settings-action" data-testid="settings-save-btn">{saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}<span>{saved ? t('settings.saved') : saving ? t('settings.saving') : t('settings.save')}</span></button></div>
  </SettingsSections></div>;
}
