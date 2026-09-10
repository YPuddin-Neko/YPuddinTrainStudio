import React from 'react';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { Settings as SettingsType } from '../../api/types';
import { PathInput } from '../../components/PathBrowser';
import { Save, Settings as SettingsIcon } from 'lucide-react';

export default function Settings() {
  const { i18n } = useTranslation();
  const [settings, setSettings] = React.useState<SettingsType | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [saved, setSaved] = React.useState(false);

  React.useEffect(() => {
    apiClient.get<SettingsType>('/settings').then(setSettings).catch(console.error);
  }, []);

  const update = (fn: (s: SettingsType) => SettingsType) => {
    setSettings((prev) => (prev ? fn({ ...prev, paths: { ...prev.paths }, server: { ...prev.server }, ui: { ...prev.ui } }) : prev));
  };

  const handleSave = () => {
    if (!settings) return;
    setSaving(true);
    apiClient.put<SettingsType>('/settings', settings)
      .then((res) => {
        setSettings(res);
        // 即时生效：语言
        if (res.ui?.language && res.ui.language !== i18n.language) {
          i18n.changeLanguage(res.ui.language);
          localStorage.setItem('i18nextLng', res.ui.language);
        }
        // 即时生效：主题
        if (res.ui?.theme === 'dark') document.documentElement.classList.add('dark');
        else if (res.ui?.theme === 'light') document.documentElement.classList.remove('dark');
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
      })
      .catch(console.error)
      .finally(() => setSaving(false));
  };

  if (!settings) return <div className="text-slate-500">Loading…</div>;

  return (
    <div className="space-y-6 max-w-3xl" data-testid="settings-page">
      <h2 className="text-2xl font-bold flex items-center space-x-2">
        <SettingsIcon className="w-6 h-6 text-slate-500" />
        <span>Settings</span>
      </h2>

      <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 space-y-4">
        <h3 className="font-semibold">Paths</h3>
        {(
          [
            ['data_root', 'Data root'],
            ['cache_dir', 'Cache directory'],
            ['models_dir', 'Models directory'],
            ['output_dir', 'Output directory'],
          ] as const
        ).map(([key, label]) => (
          <div key={key}>
            <label className="text-xs text-slate-400">{label}</label>
            <PathInput
              value={settings.paths[key]}
              onChange={(v) => update((s) => ({ ...s, paths: { ...s.paths, [key]: v } }))}
            />
          </div>
        ))}
      </section>

      <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 space-y-4">
        <h3 className="font-semibold">Server</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs text-slate-400">Host</label>
            <input
              type="text"
              value={settings.server.host}
              onChange={(e) => update((s) => ({ ...s, server: { ...s.server, host: e.target.value } }))}
              className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600 font-mono"
            />
          </div>
          <div>
            <label className="text-xs text-slate-400">Port</label>
            <input
              type="number"
              value={settings.server.port}
              onChange={(e) => update((s) => ({ ...s, server: { ...s.server, port: Number(e.target.value) } }))}
              className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600 font-mono"
            />
          </div>
        </div>
        <p className="text-xs text-slate-400">Server changes take effect after restart.</p>
      </section>

      <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 space-y-4">
        <h3 className="font-semibold">Interface</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs text-slate-400">Language</label>
            <select
              value={settings.ui.language}
              onChange={(e) => update((s) => ({ ...s, ui: { ...s.ui, language: e.target.value } }))}
              className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
              data-testid="settings-language"
            >
              <option value="zh-CN">中文</option>
              <option value="en">English</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-slate-400">Theme</label>
            <select
              value={settings.ui.theme}
              onChange={(e) => update((s) => ({ ...s, ui: { ...s.ui, theme: e.target.value } }))}
              className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
              data-testid="settings-theme"
            >
              <option value="system">System</option>
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </div>
        </div>
      </section>

      <div className="flex justify-end">
        <button
          onClick={handleSave}
          disabled={saving}
          className="flex items-center space-x-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium disabled:opacity-50"
          data-testid="settings-save-btn"
        >
          <Save className="w-4 h-4" />
          <span>{saved ? 'Saved!' : saving ? 'Saving…' : 'Save Settings'}</span>
        </button>
      </div>
    </div>
  );
}
