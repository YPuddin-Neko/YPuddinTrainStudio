import { useState } from 'react';
import { Loader2, Scan, Undo2, X } from 'lucide-react';
import type { AutoMaskOptions } from '../../api/types';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { VisionModelState, VisionRuntimeNotice } from './VisionModelField';
import { DeviceField, RangeField, ScopeField } from './VisionPanelParts';
import { ACTIVE_DOWNLOAD, useRememberedSettings, useScopeOptions, useVisionModels } from './visionHooks';
import type { PipelineOperation } from './DatasetPipelinePanel';
import Dialog from '../Dialog';
import CheckboxSelect from '../CheckboxSelect';
import { MASK_KINDS, MASK_MODEL_IDS, maskKindLabel, modelMaskKinds, type MaskKind } from './maskPresentation';
import './dataset-vision.css';

type Settings = Required<Pick<AutoMaskOptions, 'confidence' | 'padding' | 'feather' | 'device'>> & { kind: MaskKind; kinds?: MaskKind[] };
const DEFAULTS: Settings = { kind: 'head', confidence: 0.413, padding: 0.10, feather: 0.03, device: 'auto' };

export default function AutoMaskPanel({ projectId, versionId, locked, latest, running, onStart, onUndo, onCancel, beforeStart }: {
  projectId: string; versionId: string; locked: boolean; latest?: PipelineOperation; running?: PipelineOperation;
  onStart: (body: Record<string, unknown>) => Promise<void>; onUndo: (id: string) => void | Promise<void>;
  onCancel?: () => void | Promise<void>; beforeStart?: () => Promise<boolean>;
}) {
  const text = useWorkspaceText();
  const english = text('zh', 'en') === 'en';
  const catalog = useVisionModels();
  const scopes = useScopeOptions(projectId, versionId);
  const [settings, update] = useRememberedSettings('studio.automask.settings', DEFAULTS);
  const [scope, setScope] = useState('');
  const [open, setOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const chosenScope = scopes.options.some(option => option.value === scope) ? scope : scopes.first;
  const kinds = (settings.kinds ?? [settings.kind]).filter(kind => MASK_KINDS.includes(kind));
  const modelFor = (kind: MaskKind) => catalog.data?.models.find(model => model.id === MASK_MODEL_IDS[kind] && modelMaskKinds(model).includes(kind));
  const selected = kinds.map(kind => ({ kind, model: modelFor(kind) }));
  const models = [...new Map(selected.flatMap(item => item.model ? [[item.model.id, item.model] as const] : [])).values()];
  const missing = models.filter(model => !model.ready);
  const detectors = models.map(model => ({ model: model.id, kinds: selected.filter(item => item.model?.id === model.id).map(item => item.kind) }));
  const cuda = !!catalog.data?.runtime.providers.includes('cuda');
  const ready = !!kinds.length && selected.every(item => item.model?.ready) && !!catalog.data?.runtime.available;
  const legacyHead = selected.length === 1 && selected[0].kind === 'head' && selected[0].model?.role === 'head_detector';
  const options = { ...(legacyHead ? { model: selected[0].model!.id } : { detectors }), confidence: settings.confidence, padding: settings.padding, feather: settings.feather, device: cuda ? settings.device : 'cpu' };
  const start = async () => {
    if (!ready || locked || submitting || !chosenScope) return;
    setSubmitting(true); setError('');
    try {
      if (beforeStart && !await beforeStart()) return;
      await onStart({ action: 'automask', dataset_ids: scopes.ids(chosenScope), automask: options });
      setOpen(false);
    } catch (failure) { setError(formatApiError(failure)); }
    finally { setSubmitting(false); }
  };
  const undo = async () => {
    if (!latest || locked || submitting) return;
    if (beforeStart && !await beforeStart()) return;
    await onUndo(latest.id);
  };
  return <span className="automask-tools">
    <button type="button" className="ui-btn ui-btn-sm ui-btn-icon automask-launcher" aria-label={text('自动遮罩', 'Automatic masks')} title={text('自动遮罩', 'Automatic masks')} disabled={locked || submitting} onClick={() => { setError(''); setOpen(true); }}>
      {running ? <Loader2 size={15} className="animate-spin"/> : <Scan size={15}/>}
    </button>
    {running?.can_cancel && onCancel && <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" aria-label={text('取消自动遮罩', 'Cancel automatic masks')} title={text('取消自动遮罩', 'Cancel automatic masks')} onClick={() => void onCancel()}><X size={15}/></button>}
    {!running && latest?.status === 'completed' && latest.can_undo && <button type="button" className="ui-btn ui-btn-quiet" disabled={locked || submitting} onClick={() => void undo()}><Undo2 size={14}/>{text('撤销自动遮罩', 'Undo automatic masks')}</button>}
    {open && <Dialog title={text('自动遮罩', 'Automatic masks')} onClose={() => setOpen(false)} closeDisabled={submitting}>
      <div className="vision-panel automask-settings" data-testid="automask-panel">
        <VisionRuntimeNotice catalog={catalog}/>
        <div className="vision-panel-body">
          <div className="vision-row">
            <div className="vision-field"><span className="vision-field-label">{text('遮罩内容', 'Mask')}</span>
              <CheckboxSelect aria-label={text('遮罩内容', 'Mask')} disabled={locked || submitting} values={kinds} onValuesChange={next => update({ kinds: next as MaskKind[] })}
                options={MASK_KINDS.map(kind => {
                  const model = modelFor(kind), label = maskKindLabel(kind, english);
                  return { value: kind, displayLabel: label, label: model?.ready ? label : `${label} · ${model ? ACTIVE_DOWNLOAD.includes(model.download?.status || '') ? text('下载中', 'downloading') : text('需下载', 'download required') : text('不可用', 'unavailable')}`, disabled: !model };
                })}
                footer={missing.length ? <div className="mask-part-downloads" aria-label={text('所选部位下载', 'Downloads for selected parts')}>{missing.map(model => <div key={model.id} className="mask-part-download"><span>{selected.filter(item => item.model?.id === model.id).map(item => maskKindLabel(item.kind, english)).join(text('、', ', '))}</span><VisionModelState model={model} disabled={locked || submitting}/></div>)}</div> : undefined}/>
              <span className="vision-field-hint">{text('检测框区域不参与训练。', 'Detected box areas are left out of training.')}</span>
            </div>
            <ScopeField scopes={scopes} value={chosenScope} onChange={setScope} disabled={locked || submitting} label={text('处理范围', 'Images')} hint={text('已有遮罩会保留，检测到的区域叠加进去。', 'Existing masks stay; detected areas are added to them.')}/>
          </div>
          <div className="vision-row">
            <RangeField label={text('置信度', 'Confidence')} hint={text('越高误检越少，但可能漏掉较小的目标，常用 0.41。', 'Higher gives fewer false hits but can miss small ones; 0.41 is typical.')} value={settings.confidence} min={0.05} max={0.95} step={0.01} disabled={locked || submitting} onChange={value => update({ confidence: value })}/>
            <RangeField label={text('扩展比例', 'Padding')} hint={text('检测框向外扩大的比例。', 'How far each detection box grows.')} value={settings.padding} min={0} max={1} step={0.01} disabled={locked || submitting} onChange={value => update({ padding: value })}/>
            <RangeField label={text('羽化比例', 'Feather')} hint={text('边缘渐变的宽度，0 为硬边。', 'Width of the soft edge; 0 is a hard edge.')} value={settings.feather} min={0} max={0.5} step={0.01} disabled={locked || submitting} onChange={value => update({ feather: value })}/>
            {cuda && <DeviceField value={settings.device} disabled={locked || submitting} onChange={device => update({ device })}/>}
          </div>
        </div>
        {error && <p role="alert" className="vision-error">{error}</p>}
      </div>
      <div className="workspace-confirm-actions"><button type="button" className="ui-btn" disabled={submitting} onClick={() => setOpen(false)}>{text('取消', 'Cancel')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={locked || submitting || !ready || !chosenScope} onClick={() => void start()}>{submitting ? <Loader2 size={15} className="animate-spin"/> : <Scan size={15}/>} {text('开始处理', 'Run')}</button></div>
    </Dialog>}
  </span>;
}
