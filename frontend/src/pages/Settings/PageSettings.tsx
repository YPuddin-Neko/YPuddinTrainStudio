import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Loader2, Save } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { Settings } from '../../api/types';
import StudioSelect from '../../components/StudioSelect';
import { formatApiError } from '../../utils/errors';
import { cloneCharts, DEFAULT_METRIC_CHARTS, isDefaultLayout, layoutKey, type MetricChartSetting } from '../../utils/metricCharts';
import { useWorkspaceText } from '../../utils/workspaceText';
import MetricChartEditor from './MetricChartEditor';
import { SettingsSections } from './SettingsSections';

type Draft = { language: Settings['ui']['language']; theme: Settings['ui']['theme']; charts: MetricChartSetting[] };

const draftOf = (settings: Settings): Draft => ({
  language: settings.ui.language, theme: settings.ui.theme,
  charts: cloneCharts(settings.ui.metric_charts?.length ? settings.ui.metric_charts : DEFAULT_METRIC_CHARTS),
});
const draftKey = (draft: Draft) => JSON.stringify([draft.language, draft.theme, layoutKey(draft.charts)]);

/** How the pages look: language, theme and the job page's metric charts, saved together. */
export default function PageSettings({ focus }: { focus?: 'charts' }) {
  const { t, i18n } = useTranslation();
  const text = useWorkspaceText();
  const root = React.useRef<HTMLDivElement>(null);
  const [draft, setDraft] = React.useState<Draft | null>(null);
  const [saved, setSaved] = React.useState('');
  const [error, setError] = React.useState('');
  const [saving, setSaving] = React.useState(false);
  const [sorting, setSorting] = React.useState(false);
  const [done, setDone] = React.useState(false);

  const load = React.useCallback((signal?: AbortSignal) => {
    setError('');
    return apiClient.get<Settings>('/settings', { signal, silent: true }).then(settings => {
      if (signal?.aborted) return;
      const next = draftOf(settings);
      setDraft(next); setSaved(draftKey(next));
    }).catch(failure => { if (!signal?.aborted) setError(formatApiError(failure)); });
  }, []);
  React.useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);
  const loaded = draft !== null;
  // Opened from a job page's charts: start at the chart section.
  React.useEffect(() => {
    if (!loaded || focus !== 'charts') return;
    const section = root.current?.querySelector<HTMLElement>('#page-charts');
    section?.scrollIntoView?.({ block: 'start' });
    section?.focus({ preventScroll: true });
  }, [loaded, focus]);

  if (!draft) return <div className="settings-main-loading" data-testid="settings-loading">{error
    ? <div role="alert" className="settings-alert">{error}<button type="button" className="ui-link ml-3" onClick={() => void load()}>{t('common.retry', '重试')}</button></div>
    : <Loader2 size={18} className="animate-spin" aria-label={t('common.loading')}/>}</div>;

  const dirty = draftKey(draft) !== saved;
  const change = (patch: Partial<Draft>) => { setDraft({ ...draft, ...patch }); setDone(false); };
  const save = async () => {
    if (sorting) return;
    setSaving(true); setError('');
    try {
      // The built-in chart layout is stored as none, so a later change to the defaults still reaches it.
      const result = await apiClient.put<Settings>('/settings', { ui: { language: draft.language, theme: draft.theme, metric_charts: isDefaultLayout(draft.charts) ? null : draft.charts } }, { silent: true });
      const next = draftOf(result);
      setDraft(next); setSaved(draftKey(next)); setDone(true);
      if (result.ui.language !== i18n.language) {
        void i18n.changeLanguage(result.ui.language);
        try { localStorage.setItem('i18nextLng', result.ui.language); } catch { /* The saved setting still applies at the next start. */ }
      }
      document.documentElement.classList.toggle('dark', result.ui.theme === 'dark' || result.ui.theme === 'system' && (window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false));
      // The app shell follows the language and theme; job pages open behind this dialog redraw their charts.
      window.dispatchEvent(new CustomEvent('studio.settings.changed', { detail: result }));
    } catch (failure) { setError(formatApiError(failure)); }
    finally { setSaving(false); }
  };

  return <div ref={root} data-testid="page-settings"><SettingsSections sections={[{ id: 'page-interface', label: t('settings.ui') }, { id: 'page-charts', label: text('指标图表', 'Metric charts') }]}>
    {error && <div role="alert" className="settings-alert">{error}</div>}
    <fieldset disabled={saving} aria-busy={saving} className="contents">
      <section id="page-interface" data-settings-section tabIndex={-1} className="settings-section">
        <div className="settings-section-heading"><div><h2>{t('settings.ui')}</h2><p className="settings-note">{t('settings.uiDesc', '语言与主题在保存后立即生效。')}</p></div></div>
        <div className="settings-field"><label htmlFor="page-language">{t('settings.language')}</label><div className="settings-field-control">
          <StudioSelect id="page-language" aria-label={t('settings.language')} value={draft.language} onValueChange={value => change({ language: value as Draft['language'] })}
            options={[{ value: 'zh-CN', label: '中文' }, { value: 'en', label: 'English' }]} data-testid="settings-language"/>
        </div></div>
        <div className="settings-field"><label htmlFor="page-theme">{t('settings.theme')}</label><div className="settings-field-control">
          <StudioSelect id="page-theme" aria-label={t('settings.theme')} value={draft.theme} onValueChange={value => change({ theme: value as Draft['theme'] })}
            options={[{ value: 'system', label: t('settings.themeSystem') }, { value: 'light', label: t('settings.themeLight') }, { value: 'dark', label: t('settings.themeDark') }]} data-testid="settings-theme"/>
        </div></div>
        <div className="settings-field"><span>{text('首次设置', 'First-time setup')}</span><div className="settings-field-control"><Link to="/setup" className="ui-btn">{text('重新打开引导', 'Open setup guide')}</Link></div></div>
      </section>
      <MetricChartEditor id="page-charts" onSortingChange={setSorting} charts={draft.charts} onChange={charts => change({ charts })}/>
    </fieldset>
    <div className="settings-save">
      <span role="status" className="settings-note">{done && text('已保存，已立即生效。', 'Saved and applied.')}</span>
      <button type="button" className="ui-btn ui-btn-primary" disabled={saving || sorting || !dirty} onClick={() => void save()} data-testid="page-settings-save">{saving ? <Loader2 size={14} className="animate-spin"/> : <Save size={14}/>}<span>{saving ? t('settings.saving') : t('settings.save')}</span></button>
    </div>
  </SettingsSections></div>;
}
