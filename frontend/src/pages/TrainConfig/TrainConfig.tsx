import React from 'react';
import { Link, useParams, useNavigate, useLocation, useSearchParams, Navigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { SchemaForm, ValidationError } from '../../schema/SchemaForm/SchemaForm';
import { apiClient } from '../../api/client';
import { Job, Plan, Preset, ModelAsset, DatasetInfo } from '../../api/types';
import { useFamilies, familyByName } from '../../api/hooks/useFamilies';
import { mergeConfig } from '../../utils/config';
import { applyTrainingPreset, reusableTrainingPreset } from '../../utils/trainingPresets';
import { formatApiError } from '../../utils/errors';
import { fillDefaultModels, changeModelFamily, matchingTrainingDatasets } from '../../utils/workspaceConfig';
import { useWorkspaceText } from '../../utils/workspaceText';
import ProjectWorkspaceHeader from '../../components/projects/ProjectWorkspaceHeader';
import { useProjectVersions } from '../../components/projects/useProjectVersions';
import { projectUrl, versionConfigUrl, type VersionedProject } from '../../utils/projectVersions';
import '../../styles/project-workspace.css';
import BucketInspector from './BucketInspector';
import StudioSelect from '../../components/StudioSelect';
import { useWorkspaceHeight } from '../../components/projects/useWorkspaceHeight';
import './training-workspace.css';
import { CONFIG_TAB_GROUPS, ConfigTab, ConfigIssue, presentConfigIssues, presentPlanWarning } from '../../utils/configPresentation';
import { AlertCircle, CheckCircle2, ChevronRight, Search, SlidersHorizontal, Play, Settings2, Brush, Database, Box, Sparkles, Loader2 } from 'lucide-react';

const trainingDraftKey = (projectId: string, versionId?: string) => `training-draft:${projectId}:${versionId || 'legacy'}`;
const isConfigObject = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value);

// Restore only locally changed fields, preserving unrelated server changes made while away.
function restoreDraftChanges(current: unknown, base: unknown, draft: unknown): unknown {
  if (JSON.stringify(base) === JSON.stringify(draft)) return current;
  if (!isConfigObject(base) || !isConfigObject(draft)) return draft;
  const next: Record<string, unknown> = isConfigObject(current) ? { ...current } : {};
  for (const key of new Set([...Object.keys(base), ...Object.keys(draft)])) {
    if (['__proto__', 'constructor', 'prototype'].includes(key)) continue;
    if (JSON.stringify(base[key]) === JSON.stringify(draft[key])) continue;
    if (!(key in draft)) delete next[key];
    else next[key] = restoreDraftChanges(next[key], base[key], draft[key]);
  }
  return next;
}

function readTrainingDraft(key: string) {
  try {
    const saved: unknown = JSON.parse(sessionStorage.getItem(key) || 'null');
    if (isConfigObject(saved) && saved.version === 1 && isConfigObject(saved.base) && isConfigObject(saved.draft)) return { version: 1, base: saved.base, draft: saved.draft };
  } catch { /* Storage can be unavailable or contain an obsolete draft. */ }
  return null;
}

function clearSavedTrainingDraft(key: string, encoded: string) {
  try {
    const saved = readTrainingDraft(key);
    // An older request must not erase edits made while it was in flight.
    if (saved && JSON.stringify(saved.draft) === encoded) sessionStorage.removeItem(key);
  } catch { /* The server save remains successful when session storage is unavailable. */ }
}

function rememberTrainingDraft(key: string, config: Record<string, unknown>, base: string) {
  try {
    if (JSON.stringify(config) === base) clearSavedTrainingDraft(key, base);
    else if (base) sessionStorage.setItem(key, JSON.stringify({ version: 1, base: JSON.parse(base), draft: config }));
  } catch { /* The existing save and beforeunload guards still protect the draft. */ }
}

export default function TrainConfig() {
  const { id, versionId } = useParams<{ id: string; versionId: string }>();
  return <TrainConfigContent key={`${id}/${versionId || 'active'}`} projectId={id} versionId={versionId}/>;
}
function TrainConfigContent({ projectId, versionId }: { projectId?: string; versionId?: string }) {
  const { t, i18n } = useTranslation();
  const english = i18n.language.startsWith('en');
  const text = useWorkspaceText();
  const toolbarRef = useWorkspaceHeight('--training-toolbar-height');
  const location = useLocation();
  const navigate = useNavigate();
  const { data: families } = useFamilies();
  const familiesRef = React.useRef(families);
  React.useLayoutEffect(() => { familiesRef.current = families; }, [families]);
  const [tabParams, setTabParams] = useSearchParams();
  const requestedTab = tabParams.get('tab');
  const activeTab: ConfigTab = requestedTab && Object.prototype.hasOwnProperty.call(CONFIG_TAB_GROUPS, requestedTab) ? requestedTab as ConfigTab : 'train';
  const setActiveTab = (tab: ConfigTab) => {
    const next = new URLSearchParams(tabParams); next.set('tab', tab);
    setTabParams(next, { state: location.state });
  };
  const [search, setSearch] = React.useState('');
  const previousTab = React.useRef(activeTab);
  React.useLayoutEffect(() => {
    if (previousTab.current !== activeTab) document.getElementById('training-parameters')?.scrollIntoView?.({block:'start'});
    previousTab.current = activeTab;
  }, [activeTab]);
  const [project, setProject] = React.useState<VersionedProject | null>(null);
  const versions = useProjectVersions(project, versionId);
  const versionStatus = versions.current?.status;
  const archived = !!versions.current?.archived;
  const [datasets, setDatasets] = React.useState<DatasetInfo[]>([]);
  const [revealVersion, setRevealVersion] = React.useState(0);
  const [issuesOpen, setIssuesOpen] = React.useState(false);
  const [config, setConfig] = React.useState<Record<string, any>>({});
  const [schema, setSchema] = React.useState<any>(null);
  const [defaults, setDefaults] = React.useState<Record<string, any>>({});
  const [registeredModels, setRegisteredModels] = React.useState<ModelAsset[]>([]);
  const registeredModelsRef = React.useRef<ModelAsset[]>([]);
  const initialConfigRef = React.useRef<string|null>(null);
  const [auxiliaryErrors,setAuxiliaryErrors] = React.useState<Record<string,string>>({});
  const [auxiliaryReload,setAuxiliaryReload] = React.useState(0);
  const [auxiliaryLoading,setAuxiliaryLoading] = React.useState(false);
  const [loaded, setLoaded] = React.useState(false);
  const [recoveredDraft, setRecoveredDraft] = React.useState(false);
  const [reload, setReload] = React.useState(0);
  const [error, setError] = React.useState('');
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [presets, setPresets] = React.useState<Preset[]>([]);
  const [validatedConfig, setValidatedConfig] = React.useState('');
  const [plan, setPlan] = React.useState<Plan | null>(null);
  const [validationErrors, setValidationErrors] = React.useState<ValidationError[]>([]);
  const [validating, setValidating] = React.useState(true);
  const [isEnqueuing, setIsEnqueuing] = React.useState(false);
  const [enqueueSuccess, setEnqueueSuccess] = React.useState(false);
  const [savedAt, setSavedAt] = React.useState<string | null>(null);
  const [jobName, setJobName] = React.useState('');
  const [priority, setPriority] = React.useState(0);
  const [scheduledAt, setScheduledAt] = React.useState('');
  const [presetName, setPresetName] = React.useState('');
  const [savingPreset, setSavingPreset] = React.useState(false);
  const [importText, setImportText] = React.useState('');
  const [importOpen, setImportOpen] = React.useState(false);
  const [importing, setImporting] = React.useState(false);
  const [savingNavigation, setSavingNavigation] = React.useState(false);
  const navigationPendingRef = React.useRef(false);
  const lastSavedRef = React.useRef('');
  const submittedConfigRef = React.useRef<string | null>(null);
  const saveQueueRef = React.useRef<Promise<unknown>>(Promise.resolve());
  const draftRef = React.useRef({ projectId, versionId, config, loaded, archived });
  React.useLayoutEffect(() => {
    draftRef.current = { projectId, versionId, config, loaded, archived };
    if (projectId && loaded && !archived) rememberTrainingDraft(trainingDraftKey(projectId, versionId), config, submittedConfigRef.current || lastSavedRef.current);
  }, [projectId, versionId, config, loaded, archived]);
  React.useEffect(() => () => {
    const draft = draftRef.current;
    if (!draft.projectId || !draft.loaded || draft.archived || (!submittedConfigRef.current && JSON.stringify(draft.config) === lastSavedRef.current)) return;
    // Route navigation must not discard changes still in the autosave debounce.
    const key = trainingDraftKey(draft.projectId, draft.versionId);
    const encoded = JSON.stringify(draft.config);
    rememberTrainingDraft(key, draft.config, submittedConfigRef.current || lastSavedRef.current);
    saveQueueRef.current = saveQueueRef.current.catch(() => {}).then(async () => {
      if (encoded !== lastSavedRef.current) await apiClient.put(versionConfigUrl(draft.projectId!, draft.versionId), draft.config, { silent: true });
      clearSavedTrainingDraft(key, encoded);
    });
    void saveQueueRef.current.catch(() => {});
  }, [projectId, versionId]);

  const flushDraft = async () => {
    while (true) {
      const draft = draftRef.current;
      if (!draft.projectId || !draft.loaded || draft.archived) break;
      const encoded = JSON.stringify(draft.config);
      const pending = saveQueueRef.current.catch(() => {}).then(async () => {
        const current = draftRef.current;
        const latest = current.projectId === draft.projectId && current.versionId === draft.versionId ? current : draft;
        const submitted = JSON.stringify(latest.config);
        if (submitted === lastSavedRef.current) return;
        submittedConfigRef.current = submitted;
        try { await apiClient.put(versionConfigUrl(draft.projectId!, draft.versionId), latest.config, { silent: true }); }
        finally { submittedConfigRef.current = null; }
        lastSavedRef.current = submitted;
        clearSavedTrainingDraft(trainingDraftKey(draft.projectId!, draft.versionId), submitted);
        setSavedAt(new Date().toLocaleTimeString(undefined, { hour12: false }));
      });
      saveQueueRef.current = pending;
      await pending;
      if (draft.projectId === draftRef.current.projectId && draft.versionId === draftRef.current.versionId && encoded === JSON.stringify(draftRef.current.config)) break;
    }
  };
  const saveDraftNow = async () => {
    if (navigationPendingRef.current) return;
    navigationPendingRef.current = true; setSavingNavigation(true); setError('');
    try { await flushDraft(); }
    catch (err) { setError(`${text('草稿保存失败，修改仍保留在此页面。', 'Draft could not be saved. Your changes remain on this page.')}\n${formatApiError(err)}`); }
    finally { navigationPendingRef.current = false; setSavingNavigation(false); }
  };
  const navigateWithSavedDraft = async (destination: string) => {
    if (navigationPendingRef.current) return;
    navigationPendingRef.current = true; setSavingNavigation(true); setError('');
    try {
      await flushDraft();
      navigate(destination, destination.startsWith('/settings') ? { state: { backgroundLocation: location } } : undefined);
    } catch (err) {
      setError(`${text('草稿保存失败，已留在当前页面。请重试后再离开。', 'Draft could not be saved. Your changes remain here; retry before leaving.')}\n${formatApiError(err)}`);
    } finally { navigationPendingRef.current = false; setSavingNavigation(false); }
  };

  const handleInternalLink = (event: MouseEvent) => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const anchor = event.target instanceof Element ? event.target.closest<HTMLAnchorElement>('a[href]') : null;
    if (!anchor || (anchor.target && anchor.target.toLowerCase() !== '_self') || anchor.hasAttribute('download')) return;
    const destination = new URL(anchor.href, window.location.href);
    if (destination.origin !== window.location.origin) return;
    event.preventDefault();
    void navigateWithSavedDraft(`${destination.pathname}${destination.search}${destination.hash}`);
  };
  const linkGuard = React.useRef(handleInternalLink);
  React.useLayoutEffect(() => { linkGuard.current = handleInternalLink; });
  React.useEffect(() => {
    const guard = (event: MouseEvent) => linkGuard.current(event);
    const beforeUnload = (event: BeforeUnloadEvent) => {
      const draft = draftRef.current;
      if (draft.loaded && !draft.archived && JSON.stringify(draft.config) !== lastSavedRef.current) {
        event.preventDefault(); event.returnValue = '';
      }
    };
    document.addEventListener('click',guard,true);
    window.addEventListener('beforeunload',beforeUnload);
    return () => { document.removeEventListener('click',guard,true); window.removeEventListener('beforeunload',beforeUnload); };
  }, []);


  React.useEffect(() => {
    if (versionId && (versionStatus !== 'ready' || archived)) return;
    let active = true;
    setLoaded(false);
    setError('');
    setSavedAt(null);
    setRecoveredDraft(false);
    Promise.all([
      apiClient.get<any>('/schema/train', { silent: true }),
      apiClient.get<Record<string, any>>('/config/defaults', { silent: true }),
      projectId ? apiClient.get<Record<string, any>>(versionConfigUrl(projectId, versionId), { silent: true }) : Promise.resolve({}),
    ]).then(([nextSchema, nextDefaults, draft]) => {
      if (!active) return;
      const next = fillDefaultModels(mergeConfig(nextDefaults, draft), registeredModelsRef.current);
      setSchema(nextSchema);
      setDefaults(nextDefaults);
      initialConfigRef.current = JSON.stringify(next);
      lastSavedRef.current = JSON.stringify(next);
      const saved = projectId ? readTrainingDraft(trainingDraftKey(projectId, versionId)) : null;
      let restored: Record<string, any> = saved ? restoreDraftChanges(next, saved.base, saved.draft) as Record<string, any> : next;
      // Move drafts from the retired model page into the same version's training draft.
      // A newer model edit in the training draft takes precedence; unrelated fields survive.
      if (projectId) {
        const legacyKey = `model-draft:${projectId}:${versionId || 'legacy'}`;
        try {
          const model = JSON.parse(sessionStorage.getItem(legacyKey) || 'null');
          if (isConfigObject(model) && typeof model.family === 'string') {
            if (!saved || JSON.stringify(saved.base.model) === JSON.stringify(saved.draft.model)) {
              restored = { ...restored, model: { ...next.model, ...model } };
              if (model.family !== next.model?.family) restored = { ...restored, adapter: { ...restored.adapter, preset: familyByName(familiesRef.current, model.family)?.default_preset ?? restored.adapter?.preset }, dataset: { ...restored.dataset, text_encoding: 'auto' } };
            }
            const key = trainingDraftKey(projectId, versionId);
            rememberTrainingDraft(key, restored, JSON.stringify(next));
            if (JSON.stringify(restored) === JSON.stringify(next) || JSON.stringify(readTrainingDraft(key)?.draft) === JSON.stringify(restored)) sessionStorage.removeItem(legacyKey);
          }
        } catch { /* Leave the old draft intact if storage is unavailable. */ }
      }
      setRecoveredDraft(JSON.stringify(restored) !== JSON.stringify(next));
      setConfig(restored);
      setLoaded(true);
    }).catch((err) => { if (active) setError(formatApiError(err)); });
    return () => { active = false; };
  }, [projectId, versionId, reload, versionStatus, archived]);

  React.useEffect(() => {
    let active=true;
    setAuxiliaryLoading(true);
    const clear=(key:string)=>setAuxiliaryErrors(previous=>{const next={...previous};delete next[key];return next;});
    const requests=[
      apiClient.get<Preset[]>('/presets',{silent:true}).then(items=>{if(active){setPresets(items);clear('presets');}}).catch(error=>{if(active)setAuxiliaryErrors(previous=>({...previous,presets:formatApiError(error)}));}),
      apiClient.get<ModelAsset[]>('/models',{silent:true}).then(models=>{
        if(!active)return;
        registeredModelsRef.current=models;setRegisteredModels(models);clear('models');
        // A late registry reply/retry must not replace a draft the user has edited.
        setConfig(current=>JSON.stringify(current)===initialConfigRef.current?fillDefaultModels(current,models):current);
      }).catch(error=>{if(active)setAuxiliaryErrors(previous=>({...previous,models:formatApiError(error)}));}),
    ];
    void Promise.all(requests).finally(()=>{if(active)setAuxiliaryLoading(false);});
    return ()=>{active=false;};
  },[projectId,versionId,auxiliaryReload]);

  React.useEffect(() => {
    if (!projectId) return;
    let active = true;
    void apiClient.get<VersionedProject>(`/projects/${projectId}`, { silent: true }).then(value => { if (active) setProject(value); }).catch(() => {});
    void apiClient.get<DatasetInfo[]>(`/projects/${projectId}/datasets`, { params: { version_id: versionId }, silent: true }).then(value => { if (active) setDatasets(value); }).catch(() => {});
    return () => { active = false; };
  }, [projectId, versionId]);

  // Serialise writes so a slow older save cannot overwrite a newer draft.
  React.useEffect(() => {
    const encoded = JSON.stringify(config);
    if (!projectId || !loaded || archived || savingNavigation || encoded === lastSavedRef.current) return;
    let active = true;
    setSavedAt(null);
    const timer = setTimeout(() => {
      saveQueueRef.current = saveQueueRef.current.catch(() => {}).then(async () => {
        if (!active) return;
        let next = config;
        while (true) {
          const submitted = JSON.stringify(next);
          if (submitted !== lastSavedRef.current) {
            submittedConfigRef.current = submitted;
            try { await apiClient.put(versionConfigUrl(projectId, versionId), next, { silent: true }); }
            finally { submittedConfigRef.current = null; }
            lastSavedRef.current = submitted;
            clearSavedTrainingDraft(trainingDraftKey(projectId, versionId), submitted);
          }
          const latest = draftRef.current;
          if (latest.projectId !== projectId || latest.versionId !== versionId || !latest.loaded || latest.archived || JSON.stringify(latest.config) === submitted) break;
          // Include edits made during this request, even a return to the old saved value.
          next = latest.config;
          rememberTrainingDraft(trainingDraftKey(projectId, versionId), next, lastSavedRef.current);
        }
        setSavedAt(new Date().toLocaleTimeString(undefined, { hour12: false }));
      }).catch((err) => { setError(formatApiError(err)); });
    }, 1000);
    return () => { active = false; clearTimeout(timer); };
  }, [config, projectId, versionId, loaded, savingNavigation, archived]);

  // 族联动副作用：切换 model.family 后，adapter.preset 与 dataset.text_encoding 不合法时自动回退
  React.useEffect(() => {
    const family = familyByName(families, config?.model?.family);
    if (!family) return;
    setConfig((prev) => {
      let changed = false;
      const next = { ...prev, adapter: { ...(prev.adapter || {}) }, dataset: { ...(prev.dataset || {}) } };
      // adapter.preset 不在新族列表 → 回到该族 default_preset
      const presets = (family.presets || []).map((p) => p.name);
      if (next.adapter.preset != null && presets.length > 0 && !presets.includes(next.adapter.preset)) {
        next.adapter.preset = family.default_preset || presets[0];
        changed = true;
      }
      // dataset.text_encoding 不在族 text_modes → 回退 auto（krea2 无 online，后端会 400）
      if (
        next.dataset.text_encoding != null &&
        (family.text_modes || []).length > 0 &&
        !family.text_modes.includes(next.dataset.text_encoding)
      ) {
        next.dataset.text_encoding = 'auto';
        changed = true;
      }
      return changed ? next : prev;
    });
  }, [families, config?.model?.family]);

  React.useEffect(() => {
    if (!loaded) return;
    const controller = new AbortController();
    setValidating(true);
    const timer = setTimeout(() => {
      Promise.all([
        apiClient.post<Plan>('/plan', { config }, { signal: controller.signal, silent: true }),
        apiClient.post<{ errors: ValidationError[] }>('/config/validate', { config }, { signal: controller.signal, silent: true }),
      ]).then(([nextPlan, validation]) => {
        if (controller.signal.aborted) return;
        setPlan(nextPlan);
        setValidatedConfig(JSON.stringify(config));
        setValidationErrors([...validation.errors, ...(nextPlan.errors || [])].filter((item, index, all) => all.findIndex((x) => x.loc === item.loc && x.msg === item.msg) === index));
        setValidating(false);
      }).catch((err) => {
        if (!controller.signal.aborted) { setPlan(null); setError(formatApiError(err)); setValidating(false); }
      });
    }, 500);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [config, loaded]);

  const handleApplyPreset = (preset: Preset) => {
    setError('');
    setConfig((prev) => fillDefaultModels(applyTrainingPreset(prev, preset.config), registeredModels));
  };

  const handleConfigChange = (next: Record<string, any>) => {
    const family = familyByName(families, next.model?.family);
    setConfig(family && next.model?.family !== config.model?.family ? changeModelFamily(next, family, registeredModels) : next);
  };

  const handleSavePreset = async () => {
    if (!presetName.trim()) return;
    setSavingPreset(true);
    setError('');
    try {
      await apiClient.post('/presets', { name: presetName.trim(), config: reusableTrainingPreset(config) }, { silent: true });
      setPresets(await apiClient.get<Preset[]>('/presets'));
      setPresetName('');
    } catch (err: unknown) { setError(formatApiError(err)); }
    finally { setSavingPreset(false); }
  };

  const handleImport = async () => {
    setImporting(true);
    setError('');
    try {
      const next = await apiClient.post<Record<string, any>>('/config/import', { text: importText, format: 'toml' }, { silent: true });
      setConfig(fillDefaultModels(next, registeredModels));
      setImportOpen(false);
    } catch (err: unknown) { setError(formatApiError(err)); }
    finally { setImporting(false); }
  };

  const handleExport = async () => {
    setError('');
    try {
      const { text } = await apiClient.post<{ text: string }>('/config/export', { config, format: 'toml' }, { silent: true });
      const url = URL.createObjectURL(new Blob([text], { type: 'application/toml' }));
      const link = document.createElement('a');
      link.href = url; link.download = 'train-config.toml'; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err: unknown) { setError(formatApiError(err)); }
  };

  const handleEnqueue = async () => {
    if (archived) return;
    if (!Number.isInteger(priority) || (scheduledAt && !Number.isFinite(new Date(scheduledAt).getTime()))) {
      setError(text('请检查排期：优先级必须是整数，开始时间必须有效。', 'Check the schedule: priority must be an integer and the start time must be valid.'));
      return;
    }
    setIsEnqueuing(true);
    setError('');
    try {
      if (projectId) {
        await saveQueueRef.current.catch(() => {});
        await apiClient.put(versionConfigUrl(projectId, versionId), config, { silent: true });
        lastSavedRef.current = JSON.stringify(config);
        clearSavedTrainingDraft(trainingDraftKey(projectId, versionId), lastSavedRef.current);
      }
      const job = await apiClient.post<Job>('/jobs', {
        type: 'train', name: jobName.trim() || project?.name || `${config.model?.family || 'model'} training`, project_id: projectId || null, version_id: versionId || project?.active_version_id || null,
        config, priority, scheduled_at: scheduledAt ? new Date(scheduledAt).getTime() / 1000 : null,
      }, { silent: true });
      setEnqueueSuccess(true);
      if (job.id) navigate(`/jobs/${job.id}`);
    } catch (err: any) {
      setError(formatApiError(err));
      if (Array.isArray(err.details?.errors)) setValidationErrors(presentConfigIssues(err.details.errors, english).filter(issue => Object.values(CONFIG_TAB_GROUPS).flat().includes(issue.path.split('.')[0])).map(issue => ({loc:issue.path,msg:issue.detail})));
    } finally { setIsEnqueuing(false); }
  };

  const issues = presentConfigIssues(validationErrors, english);
  const ready = (!versions.enabled || versions.current?.status === 'ready') && loaded && !validating && validatedConfig === JSON.stringify(config) && plan?.ok === true && issues.length === 0;
  const tabs: {id: ConfigTab; label: string; icon: typeof SlidersHorizontal}[] = [
    { id: 'train', label: text('训练参数', 'Training'), icon: SlidersHorizontal },
    { id: 'data', label: text('数据与分桶', 'Dataset & buckets'), icon: Database },
    { id: 'model', label: text('模型与输出', 'Model & output'), icon: Box },
    { id: 'advanced', label: text('采样与高级', 'Sampling & advanced'), icon: Sparkles },
  ];
  const goToIssue = (issue: ConfigIssue) => {
    setActiveTab(issue.tab); setSearch(''); setShowAdvanced(true); setIssuesOpen(false); setRevealVersion(value => value + 1);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      let path = issue.path;
      let target = document.getElementById(`field-${path}`);
      while (!target && path.includes('.')) { path = path.slice(0, path.lastIndexOf('.')); target = document.getElementById(`field-${path}`); }
      target?.scrollIntoView?.({ behavior: 'smooth', block: 'center' });
      target?.querySelector<HTMLElement>('input,select,textarea,button')?.focus({ preventScroll: true });
    }));
  };
  const trainingDatasets = matchingTrainingDatasets(config, datasets);
  const indexedStats = trainingDatasets.length && trainingDatasets.every(item => item.index_status === 'ready' && Number.isFinite(item.stats?.images) && Number.isFinite(item.stats?.captioned)) ? trainingDatasets.reduce((stats, item) => ({images:stats.images + item.stats.images, captioned:stats.captioned + item.stats.captioned}), {images:0, captioned:0}) : undefined;
  const dataUrl = trainingDatasets.length === 1 ? `/datasets/${trainingDatasets[0].source.id}` : projectUrl(projectId || '', versionId, 'data');
  const dataAction = trainingDatasets.length === 1 ? text('标签与遮罩编辑', 'Captions & masks') : trainingDatasets.length > 1 ? text('选择数据集编辑标签与遮罩', 'Choose dataset for captions & masks') : text('导入数据后编辑标签与遮罩', 'Import dataset for captions & masks');
  const modelUrl = `/settings/environment?tab=models&family=${encodeURIComponent(config.model?.family || 'anima')}${projectId ? `&project=${encodeURIComponent(projectId)}` : ''}`;

  if (!versionId && project?.active_version_id) return <Navigate replace to={`${projectUrl(project.id, project.active_version_id, 'train')}${location.search}${location.hash}`} state={location.state}/>;
  const dirty = loaded && JSON.stringify(config) !== lastSavedRef.current;
  const draftStatus = <span className="draft-indicator" data-testid={savedAt && !dirty ? 'draft-saved' : undefined}>{savingNavigation ? <><Loader2 size={12} className="animate-spin"/>{text('正在保存草稿…', 'Saving draft…')}</> : dirty ? text('有未保存修改', 'Unsaved changes') : savedAt ? <><CheckCircle2 size={12}/>{t('train.draftSaved', { time: savedAt })}</> : loaded ? text('修改自动保存', 'Changes save automatically') : text('正在加载…', 'Loading…')}</span>;
  if (versionId && (versions.current?.status !== 'ready' || archived)) return <div className="training-studio project-workspace">
    {project && <ProjectWorkspaceHeader project={project} versionId={versionId} versions={versions.versions} current={versions.current} active="train" refresh={versions.refresh} error={versions.error}/>}
    {archived && <Link className="workspace-message" to={projectUrl(projectId || '', versionId, 'results')}>{text('查看此版本的训练结果', 'View this version’s training results')}</Link>}
    {!versions.current && <p className="workspace-message">{versions.loading ? t('common.loading') : text('此版本不存在或不可访问。', 'This version does not exist or is unavailable.')}</p>}
  </div>;
  return <div className="training-studio project-workspace" aria-busy={savingNavigation}>
    {project ? <ProjectWorkspaceHeader project={project} versionId={versionId} versions={versions.versions} current={versions.current} active="train" refresh={versions.refresh} beforeAction={flushDraft} status={draftStatus} error={versions.error}/> : <div className="project-heading-placeholder">{text('训练参数', 'Training parameters')}{draftStatus}</div>}
    {error && <div role="alert" className="studio-error">{error}<button type="button" onClick={() => { setError(''); if (!loaded) setReload(v => v + 1); }}>{loaded ? text('关闭', 'Dismiss') : t('common.retry')}</button></div>}
    {recoveredDraft && loaded && <p className="workspace-message" role="status">{text('已恢复此版本上次未保存的草稿。', 'Recovered the unsaved draft for this version.')}</p>}
    {Object.keys(auxiliaryErrors).length>0 && <div role="alert" className="studio-error" data-testid="training-auxiliary-error"><div>{Object.entries(auxiliaryErrors).map(([key,message])=><p key={key}>{key==='presets'?text('预设列表读取失败','Preset list could not be loaded'):text('模型库读取失败','Model registry could not be loaded')}: {message}</p>)}<p>{text('本版本配置仍可编辑；重试不会替换当前草稿。','The version configuration remains editable. Retrying will preserve the current draft.')}</p></div><button type="button" disabled={auxiliaryLoading} onClick={()=>setAuxiliaryReload(value=>value+1)}>{text('重试辅助信息','Retry supporting data')}</button></div>}
    <div className="training-toolbar" ref={toolbarRef}>
      <div className="training-toolbar-title"><h2>{text('训练参数', 'Training parameters')}</h2><span className="family-chip">{config.model?.family || '…'}</span></div>
      <label className="config-search"><Search size={15}/><input aria-label={text('搜索训练参数', 'Search training parameters')} placeholder={text('搜索参数名称或关键字…', 'Search parameters…')} value={search} onChange={event => setSearch(event.target.value)} />{search && <button aria-label={text('清空搜索', 'Clear search')} onClick={() => setSearch('')}>×</button>}</label>
      <div className="toolbar-actions"><label className="advanced-toggle"><input type="checkbox" checked={showAdvanced} onChange={event => setShowAdvanced(event.target.checked)}/>{t('train.advanced')}</label>
        <StudioSelect aria-label={t('train.loadPreset')} disabled={!loaded || savingNavigation} value="" onValueChange={name => {const preset=presets.find(item=>item.name===name);if(preset)handleApplyPreset(preset);}} options={[{value:'',label:t('train.loadPreset'),disabled:true},...presets.map(preset=>({value:preset.name,label:preset.name}))]}/>
        <button className="studio-secondary save-draft" disabled={!loaded || !dirty || savingNavigation} onClick={() => void saveDraftNow()}>{savingNavigation ? text('保存中…','Saving…') : text('保存草稿','Save draft')}</button>
        <details className="config-tools"><summary><Settings2 size={14}/>{text('配置工具', 'Config tools')}</summary><div className="config-tools-menu">
          <div className="preset-save"><input aria-label={t('train.presetName')} placeholder={t('train.presetName')} value={presetName} onChange={event => setPresetName(event.target.value)} /><button disabled={!loaded || savingPreset || !presetName.trim()} onClick={handleSavePreset}>{t('train.savePreset')}</button></div>
          <button disabled={!loaded} onClick={() => setImportOpen(value => !value)}>{t('train.importToml')}</button><button disabled={!loaded} onClick={handleExport}>{t('train.exportToml')}</button><button disabled={!loaded} onClick={() => { if (window.confirm(t('train.resetConfirm'))) setConfig(structuredClone(defaults)); }}>{t('train.resetDefaults')}</button>
        </div></details>
      </div>
    </div>
    {importOpen && <section className="config-import"><div className="flex items-center justify-between"><h2>{t('train.importToml')}</h2><button onClick={() => setImportOpen(false)}>{text('关闭', 'Close')}</button></div><input type="file" accept=".toml,text/plain" aria-label={t('train.importFile')} onChange={event => { const file = event.target.files?.[0]; if (file) file.text().then(setImportText).catch(err => setError(formatApiError(err))); }}/><textarea aria-label={t('train.importContent')} value={importText} onChange={event => setImportText(event.target.value)} /><button className="studio-primary" disabled={importing || !importText.trim()} onClick={handleImport}>{t('train.applyImport')}</button></section>}
    <div className="training-columns">
      <div className="training-editor">
        <div className="config-tabs" role="tablist" aria-label={text('参数分区', 'Parameter sections')}>{tabs.map(tab => <button role="tab" aria-label={tab.label} tabIndex={tab.id === activeTab ? 0 : -1} onKeyDown={event => {
          const index = tabs.findIndex(item => item.id === tab.id);
          const nextIndex = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : -1;
          if (nextIndex >= 0) { event.preventDefault(); setActiveTab(tabs[nextIndex].id); setSearch(''); document.getElementById(`tab-${tabs[nextIndex].id}`)?.focus(); }
        }} id={`tab-${tab.id}`} key={tab.id} aria-selected={!search && tab.id === activeTab} aria-controls="training-parameters" onClick={() => {setActiveTab(tab.id);setSearch('');}}><tab.icon size={15}/>{tab.label}{issues.some(issue => issue.tab === tab.id) && <span className="tab-issue-dot" aria-label={text('有待配置项', 'Needs configuration')}/>}</button>)}</div>
        <div id="training-parameters" role="tabpanel" aria-labelledby={search ? undefined : `tab-${activeTab}`}>
          {search && <p className="section-context">{text('搜索所有分区，包含高级参数', 'Searching every section, including advanced parameters')}</p>}
          {!search && activeTab === 'data' && <div className="config-context-card"><div><strong><Database size={14}/>{text('训练数据与遮罩', 'Dataset and masks')}</strong><p>{text('上传图片与标签，检查分桶；需要局部训练时，在图片编辑器绘制白色训练区域。', 'Upload images and captions, inspect buckets, and paint white training regions in the image editor.')}</p></div><div className="context-actions"><Link to={projectUrl(projectId || '', versionId, 'data')} className="studio-secondary">{text('添加数据', 'Add dataset')}</Link><Link to={dataUrl} className="studio-secondary"><Brush size={13}/>{dataAction}</Link></div>{config.dataset?.masked_loss && <p className="mask-context-note">{text('遮罩已启用：白色参与训练，黑色忽略。未制作遮罩且没有 alpha 通道的图片仍按整张图训练。', 'Masking enabled: white trains, black is ignored. Images without a mask or alpha still train the full image.')}</p>}</div>}
          {!search && activeTab === 'model' && <div className="config-context-card"><div><strong><Box size={14}/>{text('选择训练机上的模型', 'Models on the training machine')}</strong><p>{text('在环境设置中下载或注册模型，这里选择本次训练使用的权重。', 'Download or register models in environment settings, then choose weights for this training run.')}</p></div><Link to={modelUrl} className="studio-secondary">{text('管理与下载模型', 'Manage & download models')}<ChevronRight size={13}/></Link></div>}
          {!loaded ? <p className="p-6 text-sm text-slate-500">{t('common.loading')}</p> : <SchemaForm key={revealVersion} compact schema={schema} value={config} onChange={handleConfigChange} showAdvanced={showAdvanced || !!search} groupFilter={search ? undefined : CONFIG_TAB_GROUPS[activeTab]} search={search} errors={issues.map(issue => ({loc:issue.path,msg:issue.message}))} family={familyByName(families, config?.model?.family)} families={families} />}
        </div>
      </div>
      <aside className="training-inspector"><BucketInspector plan={plan} loading={validating} hasSources={!!config.dataset?.sources?.length} indexed={indexedStats || undefined} onIssues={() => setIssuesOpen(true)} onData={() => {setActiveTab('data');setSearch('');}}/>
        {!!plan?.warnings?.length && <details className="plan-notes"><summary><AlertCircle size={13}/>{text('配置提示', 'Configuration notes')} · {plan.warnings.length}</summary><ul>{plan.warnings.map((warning,index) => <li key={index}>{presentPlanWarning(warning.code,warning.msg,english)}</li>)}</ul></details>}
      </aside>
    </div>
    <footer className="training-launch-bar" id="training-launch">
      {issuesOpen && issues.length > 0 && <section className="readiness-panel" aria-label={text('训练前检查', 'Preflight checks')}><div className="readiness-heading"><h2>{text('完成以下配置即可启动训练', 'Complete these settings to start training')}</h2><button onClick={() => setIssuesOpen(false)}>{text('收起', 'Collapse')}</button></div>{issues.map((issue,index) => <div className="readiness-item" key={`${issue.path}-${index}`}><button aria-label={text(`配置${issue.label}`, `Configure ${issue.label}`)} onClick={() => goToIssue(issue)}><span>{issue.label}</span><span>{issue.message}</span><ChevronRight size={14}/></button>{issue.message !== issue.detail && <details><summary>{text('技术详情', 'Technical details')}</summary><code>{issue.detail}</code></details>}</div>)}</section>}
      <div className="launch-status" role="status">{validating ? <><Loader2 size={15} className="animate-spin"/><span>{text('正在检查配置…', 'Checking configuration…')}</span></> : ready ? <><CheckCircle2 size={16} className="text-emerald-500"/><span>{text('可以开始训练', 'Ready to train')}</span></> : <button onClick={() => setIssuesOpen(value => !value)} className="readiness-toggle"><AlertCircle size={16}/><span>{issues.length ? text(`${issues.length} 项待配置`, `${issues.length} settings to complete`) : text('尚未通过检查', 'Checks incomplete')}</span><ChevronRight size={14}/></button>}</div>
      <span className="launch-estimate">{plan?.total_steps ?? '—'} <span>{text('步', 'steps')}</span></span>
      <input className="launch-name" aria-label={t('train.jobName')} placeholder={text('任务名称（可选）', 'Job name (optional)')} value={jobName} onChange={event => setJobName(event.target.value)}/>
      <details className="launch-schedule"><summary>{scheduledAt ? text('已排期', 'Scheduled') : text('排期', 'Schedule')}</summary><div><label>{t('queue.priority')}<input aria-label={t('queue.priority')} type="number" step="1" aria-invalid={!Number.isInteger(priority)} value={priority} onChange={event => setPriority(Number(event.target.value))}/></label>{!Number.isInteger(priority) && <p className="text-xs text-amber-600">{text('优先级必须是整数', 'Priority must be an integer')}</p>}<label>{t('train.scheduledAt')}<input aria-label={t('train.scheduledAt')} type="datetime-local" value={scheduledAt} onChange={event => setScheduledAt(event.target.value)}/></label></div></details>
      <button className="studio-primary start-training" onClick={handleEnqueue} disabled={!ready || isEnqueuing || savingNavigation || !Number.isInteger(priority)}><Play size={14}/>{enqueueSuccess ? t('train.enqueued') : isEnqueuing ? t('train.enqueuing') : text('开始训练', 'Start training')}</button>
    </footer>
  </div>;
}
