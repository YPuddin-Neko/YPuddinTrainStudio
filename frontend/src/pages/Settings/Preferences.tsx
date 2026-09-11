import React from 'react';
import { formatApiError } from '../../utils/errors';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { Settings as SettingsType } from '../../api/types';
import { PathInput } from '../../components/PathBrowser';
import { FolderCog, Loader2, Palette, Save, Server } from 'lucide-react';

export default function Preferences() {
  const { t, i18n } = useTranslation();
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

  return (
    <div className="space-y-4 max-w-4xl" data-testid="settings-page">
      {error && <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950/20 dark:text-red-300">{error}</div>}

      <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 space-y-4">
        <div>
          <h3 className="font-semibold flex items-center space-x-2">
            <FolderCog className="w-4 h-4 text-slate-400" />
            <span>{t('settings.paths')}</span>
          </h3>
          <p className="mt-0.5 text-xs text-slate-400">{t('settings.pathsDesc', '训练数据、缓存、模型与产物的存储位置。')}</p>
        </div>
        {(
          [
            ['data_root', t('settings.dataRoot')],
            ['cache_dir', t('settings.cacheDir')],
            ['models_dir', t('settings.modelsDir')],
            ['output_dir', t('settings.outputDir')],
          ] as const
        ).map(([key, label]) => (
          <div key={key}>
            <label className="text-xs text-slate-400">{label}</label>
            {key === 'data_root' ? <>
              <input aria-label={label} readOnly value={settings.paths[key]} className="w-full rounded border px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600 opacity-70" />
              <p className="mt-1 text-xs text-slate-400">{t('settings.dataRootNote')}</p>
            </> : <PathInput ariaLabel={label} value={settings.paths[key]} onChange={(v) => update((s) => ({ ...s, paths: { ...s.paths, [key]: v } }))} />}
          </div>
        ))}
        <p className="text-xs text-slate-400">{t('settings.newJobsPaths')}</p>
      </section>

      <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 space-y-4">
        <div>
          <h3 className="font-semibold flex items-center space-x-2">
            <Server className="w-4 h-4 text-slate-400" />
            <span>{t('settings.server')}</span>
          </h3>
          <p className="mt-0.5 text-xs text-slate-400">{t('settings.serverDesc', '内置 API 服务的监听配置。')}</p>
        </div>
        <div className="rounded-lg bg-slate-50 p-3 text-xs leading-6 text-slate-500 dark:bg-slate-900">
          <p>{t('settings.connectedService', '当前连接')}：<span className="font-mono">{window.location.origin}</span></p>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs text-slate-400">{t('settings.host')}</label>
            <input
              type="text"
              aria-label={t('settings.host')}
              value={settings.server.host}
              onChange={(e) => update((s) => ({ ...s, server: { ...s.server, host: e.target.value } }))}
              className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600 font-mono"
            />
          </div>
          <div>
            <label className="text-xs text-slate-400">{t('settings.port')}</label>
            <input
              type="number"
              aria-label={t('settings.port')}
              min={1}
              max={65535}
              value={settings.server.port}
              onChange={(e) => update((s) => ({ ...s, server: { ...s.server, port: Number(e.target.value) } }))}
              className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600 font-mono"
            />
          </div>
        </div>
        <p className="text-xs text-slate-400">{t('settings.serverNote')}</p>
      </section>

      <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 space-y-4">
        <div>
          <h3 className="font-semibold flex items-center space-x-2">
            <Palette className="w-4 h-4 text-slate-400" />
            <span>{t('settings.ui')}</span>
          </h3>
          <p className="mt-0.5 text-xs text-slate-400">{t('settings.uiDesc', '语言与主题在保存后立即生效。')}</p>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs text-slate-400">{t('settings.language')}</label>
            <select
              aria-label={t('settings.language')}
              value={settings.ui.language}
              onChange={(e) => update((s) => ({ ...s, ui: { ...s.ui, language: e.target.value as SettingsType['ui']['language'] } }))}
              className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
              data-testid="settings-language"
            >
              <option value="zh-CN">中文</option>
              <option value="en">English</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-400">{t('settings.theme')}</label>
            <select
              aria-label={t('settings.theme')}
              value={settings.ui.theme}
              onChange={(e) => update((s) => ({ ...s, ui: { ...s.ui, theme: e.target.value as SettingsType['ui']['theme'] } }))}
              className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
              data-testid="settings-theme"
            >
              <option value="system">{t('settings.themeSystem')}</option>
              <option value="light">{t('settings.themeLight')}</option>
              <option value="dark">{t('settings.themeDark')}</option>
            </select>
          </div>
        </div>
      </section>

      <div className="sticky bottom-0 flex items-center justify-end border-t border-slate-200 bg-white/95 p-3 dark:border-slate-700 dark:bg-slate-900/95">
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center space-x-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium disabled:opacity-50"
          data-testid="settings-save-btn"
        >
          <Save className="w-4 h-4" />
          <span>{saved ? t('settings.saved') : saving ? t('settings.saving') : t('settings.save')}</span>
        </button>
      </div>
    </div>
  );
}
