import React from 'react';
import { Link } from 'react-router-dom';
import { formatApiError } from '../../utils/errors';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { ModelAsset, Settings as SettingsType, SystemInfo } from '../../api/types';
import { PathInput } from '../../components/PathBrowser';
import { EnvironmentStatus } from '../../components/EnvironmentStatus';
import { Download, FolderCog, HardDrive, Loader2, Palette, Save, Server, Settings as SettingsIcon } from 'lucide-react';

export default function Settings() {
  const { t, i18n } = useTranslation();
  const [settings, setSettings] = React.useState<SettingsType | null>(null);
  const [saving, setSaving] = React.useState(false);
  const [saved, setSaved] = React.useState(false);
  const [error, setError] = React.useState('');
  const [models, setModels] = React.useState<ModelAsset[]>([]);
  const [modelFamily, setModelFamily] = React.useState('anima');
  const [modelSaving, setModelSaving] = React.useState(false);
  const [info, setInfo] = React.useState<SystemInfo | null>(null);

  const load = React.useCallback(async () => {
    setError('');
    try {
      const [config, assets, system] = await Promise.all([
        apiClient.get<SettingsType>('/settings'), apiClient.get<ModelAsset[]>('/models'), apiClient.get<SystemInfo>('/system/info'),
      ]);
      setSettings(config); setModels(assets); setInfo(system);
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
        document.documentElement.classList.toggle('dark', res.ui.theme === 'dark' || res.ui.theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
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
    <div className="space-y-6 max-w-3xl" data-testid="settings-page">
      <div>
        <h2 className="text-2xl font-bold flex items-center space-x-2"><SettingsIcon className="w-6 h-6 text-slate-500" /><span>{t('settings.title')}</span></h2>
        <p className="mt-1 text-sm text-slate-500">{t('settings.subtitle', '数据目录、服务监听与界面偏好。')}</p>
      </div>

      {error && <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950/20 dark:text-red-300">{error}</div>}

      <section id="models" className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3"><div><h3 className="font-semibold flex items-center gap-2"><HardDrive className="w-4 h-4 text-slate-400" />{t('settings.defaultModels', '默认训练模型')}</h3><p className="mt-1 text-xs leading-6 text-slate-500">{t('settings.defaultModelsHelp', '按模型族分别设置。选择后立即保存，新项目和未填写的训练路径会使用这些默认值；已有显式配置保留。')}</p></div><Link to={`/models?family=${modelFamily}`} className="inline-flex items-center gap-2 rounded-lg bg-blue-50 px-3 py-2 text-sm text-blue-600 dark:bg-blue-950/30 dark:text-blue-400"><Download size={16} />{t('settings.manageModels', '下载 / 添加模型')}</Link></div>
        <div className="flex gap-2">{['anima', 'krea2'].map(f => <button key={f} className={`rounded-lg px-4 py-2 text-sm ${modelFamily === f ? 'bg-blue-600 text-white' : 'bg-slate-100 dark:bg-slate-900'}`} onClick={() => setModelFamily(f)}>{f === 'anima' ? 'Anima' : 'Krea 2'}</button>)}</div>
        {['dit', 'text_encoder', 'vae', 'tokenizer'].map(kind => {
          const choices = models.filter(m => m.family === modelFamily && m.kind === kind);
          const selected = choices.find(m => m.is_default);
          const title = t(`models.kind_${kind}`, { dit: '主模型 / DiT', text_encoder: '文本编码器', vae: 'VAE', tokenizer: '分词器目录' }[kind] || kind);
          return <div key={kind} className="grid gap-2 sm:grid-cols-[140px_1fr]"><label htmlFor={`default-${kind}`} className="pt-2 text-sm text-slate-500">{title}</label><div><select id={`default-${kind}`} disabled={modelSaving} value={selected?.id || ''} className="w-full rounded-lg border px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-900" onChange={async e => {
            const id = e.target.value; setModelSaving(true); setError('');
            try {
              if (id) await apiClient.patch(`/models/${id}`, { is_default: true });
              else if (selected) await apiClient.patch(`/models/${selected.id}`, { is_default: false });
              setModels(await apiClient.get<ModelAsset[]>('/models'));
            } catch (failure) { setError(formatApiError(failure)); }
            finally { setModelSaving(false); }
          }}><option value="">{kind === 'tokenizer' ? t('settings.bundledTokenizer', '使用内置分词器') : t('settings.noDefaultModel', '尚未设置 — 请先添加或下载')}</option>{choices.map(m => <option key={m.id} value={m.id} disabled={!m.exists}>{m.path.split(/[\\/]/).pop()}{!m.exists ? ` (${t('models.missing')})` : ''}</option>)}</select>{selected && <p className="mt-1 break-all text-xs text-slate-400">{selected.path}</p>}</div></div>;
        })}
      </section>
      <EnvironmentStatus />
      <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 space-y-4">
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

      <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 space-y-4">
        <div>
          <h3 className="font-semibold flex items-center space-x-2">
            <Server className="w-4 h-4 text-slate-400" />
            <span>{t('settings.server')}</span>
          </h3>
          <p className="mt-0.5 text-xs text-slate-400">{t('settings.serverDesc', '内置 API 服务的监听配置。')}</p>
        </div>
        <div className="rounded-lg bg-slate-50 p-3 text-xs leading-6 text-slate-500 dark:bg-slate-900">
          <p>{t('settings.connectedService', '当前连接')}：<span className="font-mono">{window.location.origin}</span></p>
          {info && <p>Python {info.python || '—'} · {info.platform || '—'}</p>}
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

      <section className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6 space-y-4">
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

      <div className="flex justify-end">
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
