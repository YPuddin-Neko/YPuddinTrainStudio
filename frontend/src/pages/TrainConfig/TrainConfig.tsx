import React from 'react';
import { Link, useParams, useNavigate, useLocation, useSearchParams, Navigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { SchemaForm, ValidationError, SourceRoleInfo, OutputBindingInfo } from '../../schema/SchemaForm/SchemaForm';
import { apiClient } from '../../api/client';
import { Job, Plan, Preset, ModelAsset, DatasetInfo } from '../../api/types';
import { useFamilies, familyByName } from '../../api/hooks/useFamilies';
import { mergeConfig } from '../../utils/config';
import { applyTrainingPreset, reusableTrainingPreset } from '../../utils/trainingPresets';
import { formatApiError } from '../../utils/errors';
import { fillDefaultModels, changeModelFamily, matchingTrainingDatasets } from '../../utils/workspaceConfig';
import { inactiveTrainingReason } from '../../utils/trainingFamilies';
import { currentTrainingComputePolicy } from '../../utils/trainingComputePolicy';
import { useWorkspaceText } from '../../utils/workspaceText';
import ProjectWorkspaceHeader from '../../components/projects/ProjectWorkspaceHeader';
import { useProjectVersions } from '../../components/projects/useProjectVersions';
import { projectUrl, versionConfigUrl, type VersionedProject } from '../../utils/projectVersions';
import '../../styles/project-workspace.css';
import BucketInspector from './BucketInspector';
import StudioSelect from '../../components/StudioSelect';
import GpuDevicePicker from '../../components/GpuDevicePicker';
import { useQueueDevices } from '../../api/hooks/useQueueDevices';
import { gpuDeviceLabel, gpuSelectionValid } from '../../utils/gpuDevices';
import PresetPreview from '../../components/PresetPreview';
import { useWorkspaceHeight } from '../../components/projects/useWorkspaceHeight';
import './training-workspace.css';
import '../../styles/parameter-workspace.css';
import ParameterModeToggle from '../../components/ParameterModeToggle';
import Dialog from '../../components/Dialog';
import ConfigInspection from './ConfigInspection';
import ParameterSections from '../../components/ParameterSections';
import { workflowSchema } from '../../utils/parameterWorkflow';
import { CONFIG_TAB_GROUPS, ConfigTab, ConfigIssue, OPAQUE_CONFIG_ISSUE, presentConfigIssues, presentPlanWarning } from '../../utils/configPresentation';
import { AlertCircle, Check, CheckCircle2, ChevronRight, ChevronDown, Search, Play, Settings2, Brush, Database, Loader2, BarChart3, X, Save, ListChecks } from 'lucide-react';

const presetFamily = (preset: Preset): string | undefined => { const model = preset.config.model; return model && typeof model === 'object' && 'family' in model && typeof model.family === 'string' ? model.family : undefined; };

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
  const location = useLocation();
  const navigate = useNavigate();
  const { data: families } = useFamilies();
  const familiesRef = React.useRef(families);
  React.useLayoutEffect(() => { familiesRef.current = families; }, [families]);
  const [tabParams, setTabParams] = useSearchParams();
  const requestedTab = tabParams.get('tab');
  const activeTab: ConfigTab = requestedTab && Object.prototype.hasOwnProperty.call(CONFIG_TAB_GROUPS, requestedTab) ? requestedTab as ConfigTab : 'model';
  const setActiveTab = (tab: ConfigTab, group?: string) => {
    setInspectorOpen(false);
    const next = new URLSearchParams(tabParams); next.set('tab', tab);
    if (group) next.set('group', group); else next.delete('group');
    setTabParams(next, { state: location.state });
  };
  const [search, setSearch] = React.useState('');
  const searchRef = React.useRef<HTMLInputElement>(null);
  const parameterScrollRef = React.useRef<HTMLDivElement>(null);
  const [inspectorOpen, setInspectorOpen] = React.useState(false);
  const clearSearch = () => { setSearch(''); searchRef.current?.focus(); };
  const [project, setProject] = React.useState<VersionedProject | null>(null);
  const versions = useProjectVersions(project, versionId);
  const versionStatus = versions.current?.status;
  const archived = !!versions.current?.archived;
  const editorVisible = !versionId || (versionStatus === 'ready' && !archived);
  const toolbarRef = useWorkspaceHeight('--training-toolbar-height', editorVisible);
  const launchBarRef = useWorkspaceHeight('--training-launch-height', editorVisible);
  const [datasets, setDatasets] = React.useState<DatasetInfo[]>([]);
  const [sourceRoles, setSourceRoles] = React.useState<SourceRoleInfo[]>([]);
  const [outputBinding, setOutputBinding] = React.useState<OutputBindingInfo | null>(null);
  const [revealVersion, setRevealVersion] = React.useState(0);
  const [issuesOpen, setIssuesOpen] = React.useState(false);
  const [config, setConfig] = React.useState<Record<string, any>>({});
  const [schema, setSchema] = React.useState<any>(null);
  const orderedSchema = React.useMemo(() => schema ? workflowSchema(schema) : schema, [schema]);
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
  const [planError, setPlanError] = React.useState('');
  const planErrorRef = React.useRef('');
  const [validationErrors, setValidationErrors] = React.useState<ValidationError[]>([]);
  const [validating, setValidating] = React.useState(true);
  const [isEnqueuing, setIsEnqueuing] = React.useState(false);
  const [enqueueSuccess, setEnqueueSuccess] = React.useState(false);
  const [savedAt, setSavedAt] = React.useState<string | null>(null);
  const [draftSaveState, setDraftSaveState] = React.useState<'idle' | 'saving' | 'failed'>('idle');
  const saveStatusRevision = React.useRef(0);
  const lastSaveError = React.useRef('');
  const saveStatusMounted = React.useRef(true);
  const [jobName, setJobName] = React.useState('');
  const [gpuDevices, setGpuDevices] = React.useState<string[]>([]);
  const [gpuSelectionNotice, setGpuSelectionNotice] = React.useState('');
  const queueDevices = useQueueDevices();
  const gpuCount = Number(config.loop?.gpu_count ?? 1);
  const gpuValid = gpuSelectionValid(gpuDevices, gpuCount, queueDevices.snapshot);
  const previousGpuCount = React.useRef(gpuCount);
  React.useEffect(() => {
    if (!Number.isInteger(gpuCount) || gpuCount < 1 || gpuCount > 64) return;
    const previous = previousGpuCount.current;
    previousGpuCount.current = gpuCount;
    if (gpuCount < previous && gpuDevices.length > gpuCount) {
      const removed = gpuDevices.slice(gpuCount).map(gpuDeviceLabel).join('、');
      setGpuDevices(gpuDevices.slice(0, gpuCount));
      setGpuSelectionNotice(text(`显卡数量已减少为 ${gpuCount} 张，保留前 ${gpuCount} 张，已移除 ${removed}。`, `GPU count reduced to ${gpuCount}; kept the first ${gpuCount} and removed ${removed}.`));
    } else if (previous !== gpuCount) setGpuSelectionNotice('');
  }, [gpuCount, gpuDevices, text]);
  const [priority, setPriority] = React.useState(0);
  const [scheduledAt, setScheduledAt] = React.useState('');
  const [pendingPreset, setPendingPreset] = React.useState<Preset | null>(null);
  const [presetName, setPresetName] = React.useState('');
  const [savingPreset, setSavingPreset] = React.useState(false);
  const [presetDialog, setPresetDialog] = React.useState(false);
  const [presetError, setPresetError] = React.useState('');
  const [inspectionOpen, setInspectionOpen] = React.useState(false);
  const [importText, setImportText] = React.useState('');
  const [importOpen, setImportOpen] = React.useState(false);
  const [importing, setImporting] = React.useState(false);
  const [importError, setImportError] = React.useState('');
  const [savingNavigation, setSavingNavigation] = React.useState(false);
  const navigationPendingRef = React.useRef(false);
  const lastSavedRef = React.useRef('');
  const submittedConfigRef = React.useRef<string | null>(null);
  const saveQueueRef = React.useRef<Promise<unknown>>(Promise.resolve());
  const draftRef = React.useRef({ projectId, versionId, config, loaded, archived });
  React.useEffect(() => {
    saveStatusMounted.current = true;
    return () => { saveStatusMounted.current = false; };
  }, []);
  const persistDraft = React.useCallback(async (owner: string, version: string | null | undefined, draft: Record<string, any>) => {
    const revision = ++saveStatusRevision.current;
    const current = () => saveStatusMounted.current && revision === saveStatusRevision.current
      && draftRef.current.projectId === owner && draftRef.current.versionId === version && draftRef.current.loaded;
    if (current()) {
      setDraftSaveState('saving');
      const previousError = lastSaveError.current;
      if (previousError) setError(previous => previous === previousError ? '' : previous);
      lastSaveError.current = '';
    }
    try {
      await apiClient.put(versionConfigUrl(owner, version), draft, { silent: true });
      if (current()) setDraftSaveState('idle');
    } catch (err) {
      if (current()) {
        lastSaveError.current = formatApiError(err);
        setDraftSaveState('failed');
        setError(lastSaveError.current);
      }
      throw err;
    }
  }, [setError, setDraftSaveState]);
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
        try { await persistDraft(draft.projectId!, draft.versionId, latest.config); }
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
    setOutputBinding(null);
    setError('');
    setSavedAt(null);
    saveStatusRevision.current += 1;
    setDraftSaveState('idle');
    lastSaveError.current = '';
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
      apiClient.get<Preset[]>('/presets',{silent:true}).then(items=>{if(active){setPresets(items.filter(item=>!item.builtin));clear('presets');}}).catch(error=>{if(active)setAuxiliaryErrors(previous=>({...previous,presets:formatApiError(error)}));}),
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
    void apiClient.get<DatasetInfo[]>(`/projects/${projectId}/datasets`, { params: { version_id: versionId, include_cache: false }, silent: true }).then(value => { if (active) setDatasets(value); }).catch(() => {});
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
            try { await persistDraft(projectId, versionId, next); }
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
      }).catch(() => { /* persistDraft reports only errors belonging to the current draft. */ });
    }, 1000);
    return () => { active = false; clearTimeout(timer); };
  }, [config, projectId, versionId, loaded, savingNavigation, archived, persistDraft]);

  // 族联动副作用：切换 model.family 后，adapter.preset 与 dataset.text_encoding 不合法时自动回退
  React.useEffect(() => {
    const family = familyByName(families, config?.model?.family);
    if (!family) return;
    setConfig((prev) => {
      if (inactiveTrainingReason(prev)) return prev;
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
    setPlanError('');
    const supporting = <T,>(key: string, request: Promise<T>, fallback: T): Promise<T> => request.then(result => {
      if (!controller.signal.aborted) setAuxiliaryErrors(previous => {const next={...previous};delete next[key];return next;});
      return result;
    }).catch(error => {
      if (!controller.signal.aborted) setAuxiliaryErrors(previous => ({...previous,[key]:formatApiError(error)}));
      return fallback;
    });
    const timer = setTimeout(() => {
      Promise.all([
        apiClient.post<Plan>('/plan', { config, project_id: projectId || undefined, version_id: versionId }, { signal: controller.signal, silent: true }),
        apiClient.post<{ errors: ValidationError[] }>('/config/validate', { config, project_id: projectId || undefined, version_id: versionId }, { signal: controller.signal, silent: true }),
        projectId ? supporting('sources', apiClient.post<SourceRoleInfo[]>(`/projects/${projectId}/source-roles`, { config }, { params: {version_id:versionId}, signal: controller.signal, silent: true }), []) : Promise.resolve([]),
        projectId ? supporting<OutputBindingInfo | null>('output', apiClient.post<OutputBindingInfo>(`/projects/${projectId}/output-binding`, { config }, { params: {version_id:versionId}, signal: controller.signal, silent: true }), null) : Promise.resolve(null),
      ]).then(([nextPlan, validation, roles, binding]) => {
        if (controller.signal.aborted) return;
        const previousPlanError = planErrorRef.current;
        if (previousPlanError) setError(previous=>previous===previousPlanError?'':previous);
        planErrorRef.current='';
        setSourceRoles(roles);
        setOutputBinding(binding);
        setConfig(previous => {
          let changed = false;
          const next = {...previous};
          for (const section of ['dataset','validation']) {
            if (!Array.isArray(previous[section]?.sources)) continue;
            const sources = previous[section].sources.map((source:any) => {
              const role = roles.find(item => item.managed && item.section === section && item.path === source.path);
              if (!role || source.is_reg === role.is_reg) return source;
              changed = true;
              return {...source,is_reg:role.is_reg};
            });
            next[section] = {...previous[section],sources};
          }
          return changed ? next : previous;
        });
        setPlan(nextPlan);
        setValidatedConfig(JSON.stringify(config));
        setValidationErrors([...validation.errors, ...(nextPlan.errors || [])].filter((item, index, all) => all.findIndex((x) => x.loc === item.loc && x.msg === item.msg) === index));
        setValidating(false);
      }).catch((err) => {
        if (!controller.signal.aborted) { const message=formatApiError(err); planErrorRef.current=message; setPlanError(message); setValidatedConfig(''); setError(message); setValidating(false); }
      });
    }, 500);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [config, loaded, projectId, versionId, auxiliaryReload]);

  const handleApplyPreset = (preset: Preset) => {
    if (inactiveTrainingReason(config) || inactiveTrainingReason(preset.config)) return;
    if (presetFamily(preset) && presetFamily(preset) !== config.model?.family) return;
    setError('');
    setConfig((prev) => fillDefaultModels(applyTrainingPreset(prev, preset.config), registeredModels));
    setPendingPreset(null);
  };

  const handleConfigChange = (next: Record<string, any>) => {
    if (inactiveTrainingReason(config)) return;
    const family = familyByName(families, next.model?.family);
    setConfig(family && next.model?.family !== config.model?.family ? changeModelFamily(next, family, registeredModels) : next);
  };

  const handleSavePreset = async () => {
    if (inactiveTrainingReason(config) || !presetName.trim() || savingPreset) return;
    setSavingPreset(true); setPresetError('');
    try {
      const created = await apiClient.post<Preset>('/presets', { name: presetName.trim(), config: reusableTrainingPreset(config) }, { silent: true });
      setPresets(previous => [...previous.filter(item=>item.name!==created.name), created]);
      setPresetName(''); setPresetDialog(false);
    } catch (err: unknown) { setPresetError(formatApiError(err)); }
    finally { setSavingPreset(false); }
  };

  const handleImport = async () => {
    setImporting(true);
    setImportError('');
    try {
      const next = await apiClient.post<Record<string, any>>('/config/import', { text: importText, format: 'toml' }, { silent: true });
      setConfig(fillDefaultModels(next, registeredModels));
      setImportOpen(false);
    } catch (err: unknown) { setImportError(formatApiError(err)); }
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
    if (archived || !gpuValid) return;
    const inactiveReason = inactiveTrainingReason(config, english);
    if (inactiveReason) { setError(inactiveReason); return; }
    if (!Number.isInteger(priority) || (scheduledAt && !Number.isFinite(new Date(scheduledAt).getTime()))) {
      setError(text('请检查排期：优先级必须是整数，开始时间必须有效。', 'Check the schedule: priority must be an integer and the start time must be valid.'));
      return;
    }
    setIsEnqueuing(true);
    setError('');
    try {
      if (projectId) {
        await saveQueueRef.current.catch(() => {});
        await persistDraft(projectId, versionId, config);
        lastSavedRef.current = JSON.stringify(config);
        clearSavedTrainingDraft(trainingDraftKey(projectId, versionId), lastSavedRef.current);
      }
      const job = await apiClient.post<Job>('/jobs', {
        type: 'train', name: jobName.trim() || project?.name || `${config.model?.family || 'model'} training`, project_id: projectId || null, version_id: versionId || project?.active_version_id || null,
        config, priority, gpu_devices: gpuDevices, scheduled_at: scheduledAt ? new Date(scheduledAt).getTime() / 1000 : null,
      }, { silent: true });
      setEnqueueSuccess(true);
      if (job.id) navigate(`/jobs/${job.id}`);
    } catch (err: any) {
      setError(formatApiError(err));
      if (Array.isArray(err.details?.errors)) setValidationErrors(presentConfigIssues(err.details.errors, english).filter(issue => Object.values(CONFIG_TAB_GROUPS).flat().includes(issue.path.split('.')[0])).map(issue => ({loc:issue.path,msg:issue.detail})));
    } finally { setIsEnqueuing(false); }
  };

  const issues = presentConfigIssues(validationErrors, english);
  const inactiveReason = inactiveTrainingReason(config, english);
  const checked = loaded && !validating && validatedConfig === JSON.stringify(config) && plan !== null;
  const computePolicy = currentTrainingComputePolicy(plan?.compute_policy, config, validatedConfig, validating);
  const planChecked = checked && (plan?.ok === true || plan?.params != null);
  const ready = !inactiveReason && (!versions.enabled || versions.current?.status === 'ready') && loaded && !validating && validatedConfig === JSON.stringify(config) && plan?.ok === true && issues.length === 0;
  const goToIssue = (issue: ConfigIssue) => {
    setActiveTab(issue.tab); setSearch(''); setShowAdvanced(true); setIssuesOpen(false); setRevealVersion(value => value + 1);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      let path = issue.path === 'checkpoint.save_state_every_epochs' ? 'checkpoint.save_state_every_steps' : issue.path;
      let target = document.getElementById(`field-${path}`);
      while (!target && path.includes('.')) { path = path.slice(0, path.lastIndexOf('.')); target = document.getElementById(`field-${path}`); }
      target?.scrollIntoView?.({ behavior: window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center' });
      const control = target?.querySelector<HTMLElement>('input:not(:disabled):not([readonly]):not([type="hidden"]),select:not(:disabled),textarea:not(:disabled):not([readonly]),[role="combobox"]:not(:disabled):not([aria-disabled="true"])')
        || target?.querySelector<HTMLElement>('button:not(:disabled):not([aria-disabled="true"]):not(.config-help-trigger)');
      control?.focus({ preventScroll: true });
    }));
  };
  const trainingDatasets = matchingTrainingDatasets(config, datasets);
  const indexedStats = trainingDatasets.length && trainingDatasets.every(item => item.index_status === 'ready' && Number.isFinite(item.stats?.images) && Number.isFinite(item.stats?.captioned)) ? trainingDatasets.reduce((stats, item) => ({images:stats.images + item.stats.images, captioned:stats.captioned + item.stats.captioned}), {images:0, captioned:0}) : undefined;
  const dataUrl = projectUrl(projectId || '', versionId, 'data');
  const sourceQuery = trainingDatasets.length === 1 ? `&dataset=${encodeURIComponent(trainingDatasets[0].source.id)}` : '';
  const modelUrl = `/settings/environment?tab=models&family=${encodeURIComponent(config.model?.family || 'anima')}${projectId ? `&project=${encodeURIComponent(projectId)}` : ''}`;

  if (!versionId && project?.active_version_id) return <Navigate replace to={`${projectUrl(project.id, project.active_version_id, 'train')}${location.search}${location.hash}`} state={location.state}/>;
  const dirty = loaded && JSON.stringify(config) !== lastSavedRef.current;
  const familyLabel = inactiveReason ? `${config.model?.family === 'flux' ? 'FLUX.1' : 'FLUX.2 dev'} · ${text('已停用', 'Retired')}` : familyByName(families, config.model?.family)?.label || config.model?.family;
  const familyBadge = familyLabel ? <span className="family-chip" data-testid="training-family-badge" title={familyLabel}>{familyLabel}</span> : null;
  const savingDraft = savingNavigation || draftSaveState === 'saving';
  const draftStatus = loaded && (savingDraft || dirty || savedAt) ? <span className="draft-indicator" role="status" aria-label={text('配置保存状态', 'Configuration save status')} data-testid={savedAt && !dirty && !savingDraft && draftSaveState !== 'failed' ? 'draft-saved' : undefined}>{savingDraft ? <><Loader2 size={12} className="animate-spin"/>{text('保存中…', 'Saving…')}</> : dirty && draftSaveState === 'failed' ? <>{text('自动保存失败', 'Autosave failed')}<button type="button" className="ui-link" onClick={() => void saveDraftNow()}>{text('重试保存', 'Retry save')}</button></> : dirty ? text('等待保存…', 'Waiting to save…') : savedAt ? <><Check size={12} aria-hidden="true"/>{text(`已保存 ${savedAt}`, `Saved ${savedAt}`)}</> : null}</span> : null;
  if (versionId && (versions.current?.status !== 'ready' || archived)) return <div className="training-studio project-workspace">
    {project && <ProjectWorkspaceHeader project={project} versionId={versionId} versions={versions.versions} current={versions.current} active="train" refresh={versions.refresh} titleBadge={familyBadge} error={versions.error}/>}
    {archived && <Link className="workspace-message" to={projectUrl(projectId || '', versionId, 'results')}>{text('查看此版本的训练结果', 'View this version’s training results')}</Link>}
    {!versions.current && <p className="workspace-message">{versions.loading ? t('common.loading') : text('此版本不存在或不可访问。', 'This version does not exist or is unavailable.')}</p>}
  </div>;
  return <div className="training-studio project-workspace parameter-workspace" aria-busy={savingNavigation}>
    <div className="parameter-workspace-header">
    <div className="training-title-actions">
    {project ? <ProjectWorkspaceHeader project={project} versionId={versionId} versions={versions.versions} current={versions.current} active="train" refresh={versions.refresh} beforeAction={flushDraft} status={draftStatus} titleBadge={familyBadge} error={versions.error}/> : <div className="project-heading-placeholder"><h1>{text('训练参数', 'Training parameters')}</h1>{familyBadge}{draftStatus}</div>}
    <div className="training-launch-bar training-header-actions" role="group" aria-label={text('训练启动操作', 'Training launch controls')} id="training-launch" ref={launchBarRef}>
      {issuesOpen && issues.length > 0 && <section className="readiness-panel" aria-label={text('训练前检查', 'Preflight checks')}><div className="readiness-heading"><h2>{text('完成以下配置即可启动训练', 'Complete these settings to start training')}</h2><button type="button" className="ui-btn ui-btn-quiet ui-btn-sm" onClick={() => setIssuesOpen(false)}>{text('收起', 'Collapse')}</button></div>{issues.map((issue,index) => <div className="readiness-item" key={`${issue.path}-${index}`}><button aria-label={text(`配置${issue.label}`, `Configure ${issue.label}`)} onClick={() => goToIssue(issue)}><span>{issue.label}</span><span>{issue.message}</span><ChevronRight size={14}/></button>{issue.message === OPAQUE_CONFIG_ISSUE && <details><summary>{text('技术详情', 'Technical details')}</summary><code>{issue.detail}</code></details>}</div>)}</section>}
      <div className="launch-status" role="status">{validating ? <><Loader2 size={15} className="animate-spin"/><span>{text('正在检查配置…', 'Checking configuration…')}</span></> : ready ? <><CheckCircle2 size={16} className="text-emerald-500"/><span>{text('可以开始训练', 'Ready to train')}</span></> : <button onClick={() => setIssuesOpen(value => !value)} className="readiness-toggle"><AlertCircle size={16}/><span>{issues.length ? text(`${issues.length} 项待配置`, `${issues.length} settings to complete`) : text('尚未通过检查', 'Checks incomplete')}</span><ChevronRight size={14}/></button>}</div>
      <button type="button" className="ui-btn launch-plan-toggle" aria-label={text('训练估算与分桶', 'Estimates and buckets')} aria-expanded={inspectorOpen} aria-controls="training-plan-panel" onClick={() => setInspectorOpen(open => !open)}><BarChart3 size={15}/><span>{text('执行估算', 'Estimates')}</span><ChevronDown size={14}/></button>
      <GpuDevicePicker compact training label={text('训练显卡', 'Training GPUs')} value={gpuDevices} count={gpuCount} deviceState={queueDevices}
        disabled={!loaded || !!inactiveReason || isEnqueuing || savingNavigation} onChange={devices => {setGpuDevices(devices);setGpuSelectionNotice('');}}/>
      <input className="launch-name" aria-label={t('train.jobName')} placeholder={text('任务名称（可选）', 'Job name (optional)')} value={jobName} onChange={event => setJobName(event.target.value)}/>
      <details className="launch-schedule" data-popover><summary className="ui-btn">{scheduledAt ? text('已排期', 'Scheduled') : text('排期', 'Schedule')}</summary><div><label>{t('queue.priority')}<input aria-label={t('queue.priority')} type="number" step="1" aria-invalid={!Number.isInteger(priority)} value={priority} onChange={event => setPriority(Number(event.target.value))}/></label>{!Number.isInteger(priority) && <p className="text-xs text-amber-600">{text('优先级必须是整数', 'Priority must be an integer')}</p>}<label>{t('train.scheduledAt')}<input aria-label={t('train.scheduledAt')} type="datetime-local" value={scheduledAt} onChange={event => setScheduledAt(event.target.value)}/></label></div></details>
      <button type="button" className="ui-btn ui-btn-primary start-training" onClick={handleEnqueue} disabled={!ready || !gpuValid || isEnqueuing || savingNavigation || !Number.isInteger(priority)}><Play size={14}/>{enqueueSuccess ? t('train.enqueued') : isEnqueuing ? t('train.enqueuing') : text('开始训练', 'Start training')}</button>
    </div>
    </div>
    {inactiveReason && <div role="alert" className="studio-error" data-testid="retired-training-config">{inactiveReason}</div>}
    {error && <div role="alert" className="studio-error">{error}<button type="button" className="ui-btn ui-btn-sm" onClick={() => { setError(''); if (!loaded) setReload(v => v + 1); }}>{loaded ? text('关闭', 'Dismiss') : t('common.retry')}</button></div>}
    {gpuSelectionNotice && <p className="workspace-message" role="status">{gpuSelectionNotice}</p>}
    {recoveredDraft && loaded && <p className="workspace-message" role="status">{text('已恢复此版本上次未保存的草稿。', 'Recovered the unsaved draft for this version.')}</p>}
    {Object.keys(auxiliaryErrors).length>0 && <div role="alert" className="studio-error" data-testid="training-auxiliary-error"><div>{Object.entries(auxiliaryErrors).map(([key,message])=><p key={key}>{key==='presets'?text('预设列表读取失败','Preset list could not be loaded'):key==='sources'?text('数据目录用途读取失败','Dataset directory ownership could not be loaded'):key==='output'?text('权重保存位置读取失败','Weight output binding could not be loaded'):text('模型库读取失败','Model registry could not be loaded')}: {message}</p>)}<p>{text('草稿已保留，可继续编辑。','Your draft is retained and editable.')}</p></div><button type="button" className="ui-btn ui-btn-sm" disabled={auxiliaryLoading} onClick={()=>setAuxiliaryReload(value=>value+1)}>{text('重试辅助信息','Retry supporting data')}</button></div>}
    <div className="training-toolbar" ref={toolbarRef} role="group" aria-label={text('训练参数工具栏', 'Training parameter controls')}>
      <div className="training-toolbar-filters">
      <label className="config-search"><Search size={16}/><input ref={searchRef} aria-label={text('搜索训练参数', 'Search training parameters')} placeholder={text('搜索参数名称或关键字…', 'Search parameters…')} value={search} onChange={event => {setSearch(event.target.value);setInspectorOpen(false);}} />{search && <button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" aria-label={text('清空搜索', 'Clear search')} onClick={clearSearch}><X size={15}/></button>}</label>
      <ParameterModeToggle advanced={showAdvanced} onChange={setShowAdvanced}/>
      </div>
      <div className="toolbar-actions">
        <StudioSelect aria-label={t('train.loadPreset')} disabled={!loaded || savingNavigation || !!inactiveReason} value="" onValueChange={name => {const preset=presets.find(item=>item.name===name);if(preset)setPendingPreset(preset);}} placeholder={t('train.loadPreset')} options={presets.map(preset=>({value:preset.name,label:`${preset.name}${inactiveTrainingReason(preset.config) ? ` · ${text('已停用','Retired')}` : presetFamily(preset) && presetFamily(preset)!==config.model?.family ? ` · ${text('适用于','For')} ${presetFamily(preset)}` : ''}`,disabled:!!inactiveTrainingReason(preset.config) || !!presetFamily(preset) && presetFamily(preset)!==config.model?.family}))}/>
        <button type="button" className="ui-btn" disabled={!loaded || savingNavigation || !!inactiveReason} onClick={() => {setPresetName('');setPresetError('');setPresetDialog(true);}}><Save size={14}/>{text('另存为新预设','Save as new preset')}</button>
        <button type="button" className="ui-btn" disabled={!loaded || savingNavigation} onClick={() => setInspectionOpen(true)}><ListChecks size={14}/>{text('参数检查','Check parameters')}</button>
        <details className="config-tools" data-popover><summary className="ui-btn"><Settings2 size={14}/>{text('配置工具', 'Config tools')}</summary><div className="config-tools-menu">
          <Link to="/presets">{text('管理参数预设','Manage parameter presets')}</Link>
          <Link to={modelUrl}>{text('管理与下载模型','Manage & download models')}</Link>
          <button type="button" disabled={!loaded} onClick={() => {setImportError('');setImportOpen(true);}}>{t('train.importToml')}</button><button type="button" disabled={!loaded} onClick={handleExport}>{t('train.exportToml')}</button><button type="button" disabled={!loaded} onClick={() => { if (window.confirm(t('train.resetConfirm'))) setConfig(structuredClone(defaults)); }}>{t('train.resetDefaults')}</button>
        </div></details>
      </div>
    </div>
    {presetDialog && <Dialog title={text('另存为新预设','Save as new preset')} closeDisabled={savingPreset} onClose={()=>setPresetDialog(false)}><form className="preset-create-dialog" onSubmit={event=>{event.preventDefault();void handleSavePreset();}}>
      <label>{text('预设名称','Preset name')}<input autoFocus aria-label={t('train.presetName')} value={presetName} disabled={savingPreset} required onChange={event=>setPresetName(event.target.value)}/></label>
      {presetError && <p role="alert" className="config-field-error">{presetError}</p>}
      <footer><button type="button" className="ui-btn" disabled={savingPreset} onClick={()=>setPresetDialog(false)}>{text('取消','Cancel')}</button><button type="submit" className="ui-btn ui-btn-primary" disabled={savingPreset || !presetName.trim()}>{savingPreset?text('保存中…','Saving…'):text('保存新预设','Save new preset')}</button></footer>
    </form></Dialog>}
    {inspectionOpen && <ConfigInspection config={config} onApply={next=>{setConfig(next);setInspectionOpen(false);}} onClose={()=>setInspectionOpen(false)}/>}
    {pendingPreset && <PresetPreview preset={pendingPreset} current={config} onClose={()=>setPendingPreset(null)} onApply={()=>handleApplyPreset(pendingPreset)}/>}
    {importOpen && <Dialog title={t('train.importToml')} closeDisabled={importing} onClose={() => setImportOpen(false)}><section className="config-import">{importError && <p role="alert" className="studio-error">{importError}</p>}<input disabled={importing} type="file" accept=".toml,text/plain" aria-label={t('train.importFile')} onChange={event => { const file = event.target.files?.[0]; if (file) file.text().then(setImportText).catch(err => setImportError(formatApiError(err))); }}/><textarea disabled={importing} aria-label={t('train.importContent')} value={importText} onChange={event => setImportText(event.target.value)} /><button type="button" className="ui-btn ui-btn-primary" disabled={importing || !importText.trim()} onClick={handleImport}>{t('train.applyImport')}</button></section></Dialog>}


    </div>
    <div className="parameter-workspace-body">
      <ParameterSections rootRef={parameterScrollRef} tab={activeTab} group={tabParams.get('group') || undefined} onTabChange={(tab, group) => {setActiveTab(tab, group);setSearch('');}} issues={issues} checked={checked} planChecked={planChecked} hasTrainingMode={!!schema?.properties?.training} fullTraining={config.training?.mode === 'full'} onRevealAdvanced={() => setShowAdvanced(true)}/>
      <div className="training-editor parameter-scroll-region" ref={parameterScrollRef}>
        <div id="training-parameters" role="region" aria-label={search ? text('训练参数搜索结果', 'Training parameter search results') : text('训练参数内容', 'Training parameter fields')}>
          {search && <p className="section-context">{text('搜索所有分区，包含高级参数', 'Searching every section, including advanced parameters')}</p>}
          {!search && activeTab === 'data' && <div className="config-context-card"><div><strong><Database size={14}/>{text('训练数据与遮罩', 'Dataset and masks')}</strong></div><div className="context-actions"><Link to={`${dataUrl}&data_step=datasets`} className="ui-btn ui-btn-sm">{text('添加数据', 'Add dataset')}</Link><Link to={`${dataUrl}&data_step=captions${sourceQuery}`} className="ui-btn ui-btn-sm">{text('标签编辑', 'Caption editor')}</Link><Link to={`${dataUrl}&data_step=paint${sourceQuery}`} className="ui-btn ui-btn-sm"><Brush size={13}/>{text('涂抹与遮罩', 'Paint & masks')}</Link></div>{config.dataset?.masked_loss && <p className="mask-context-note">{text('遮罩已启用：白色参与训练，黑色忽略。未制作遮罩且没有 alpha 通道的图片仍按整张图训练。', 'Masking enabled: white trains, black is ignored. Images without a mask or alpha still train the full image.')}</p>}</div>}


          {!loaded ? <p className="p-6 text-sm text-slate-500">{t('common.loading')}</p> : <SchemaForm projectId={projectId} versionId={versionId} key={revealVersion} compact readOnly={!!inactiveReason} schema={orderedSchema} value={config} computePolicy={computePolicy} sourceRoles={sourceRoles} outputBinding={outputBinding} versionSources={!!projectId} onChange={handleConfigChange} showAdvanced={showAdvanced || !!search} search={search} onClearSearch={clearSearch} errors={issues.map(issue => ({loc:issue.path,msg:issue.message === OPAQUE_CONFIG_ISSUE ? '' : issue.message}))} family={familyByName(families, config?.model?.family)} families={families} />}
        </div>
      </div>
      <aside id="training-plan-panel" className={`training-inspector ${inspectorOpen ? 'is-open' : ''}`} aria-label={text('训练计划', 'Training plan')}><button type="button" className="ui-btn ui-btn-quiet inspector-return" onClick={() => setInspectorOpen(false)}>{text('返回参数', 'Back to parameters')}</button><BucketInspector plan={plan} loading={validating} dataset={config.dataset} error={planError} onRetry={()=>setAuxiliaryReload(value=>value+1)} hasSources={!!config.dataset?.sources?.length} indexed={indexedStats || undefined} onIssues={() => setIssuesOpen(true)} onData={() => {setActiveTab('data');setSearch('');setInspectorOpen(false);}}/>
        {!!plan?.warnings?.length && <details className="plan-notes"><summary><AlertCircle size={13}/>{text('配置提示', 'Configuration notes')} · {plan.warnings.length}</summary><ul>{plan.warnings.map((warning,index) => <li key={index}>{presentPlanWarning(warning.code,warning.msg,english)}</li>)}</ul></details>}
      </aside>
    </div>

  </div>;
}
