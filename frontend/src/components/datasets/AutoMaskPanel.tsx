import { useState, type ReactNode } from 'react';
import { ScanFace } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { AutoMaskOptions, HeadSelection } from '../../api/types';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import VisionModelField, { VisionRuntimeNotice } from './VisionModelField';
import { DeviceField, OperationResult, RangeField, ScopeField } from './VisionPanelParts';
import { useRememberedSettings, useScopeOptions, useVisionModels } from './visionHooks';
import type { PipelineOperation } from './DatasetPipelinePanel';
import HeadMaskReview from './HeadMaskReview';
import './dataset-vision.css';

type Settings = Required<Pick<AutoMaskOptions, 'confidence' | 'padding' | 'feather' | 'device'>>;
const DEFAULTS: Settings = { confidence: 0.413, padding: 0.10, feather: 0.03, device: 'auto' };

export default function AutoMaskPanel({ projectId, versionId, locked, latest, detection, running, onStart, onUndo, onRefresh }: {
  projectId: string; versionId: string; locked: boolean; latest?: PipelineOperation; detection?: PipelineOperation; running?: ReactNode;
  onStart: (body: Record<string, unknown>) => Promise<void>; onUndo: (id: string) => void; onRefresh: () => void;
}) {
  const text = useWorkspaceText();
  const catalog = useVisionModels();
  const scopes = useScopeOptions(projectId, versionId);
  const [settings, update] = useRememberedSettings('studio.automask.settings', DEFAULTS);
  const [scope, setScope] = useState('');
  const [error, setError] = useState('');
  const chosenScope = scopes.options.some(option => option.value === scope) ? scope : scopes.first;
  const model = catalog.data?.models.find(item => item.role === 'head_detector');
  const cuda = !!catalog.data?.runtime.providers.includes('cuda');
  const ready = !!model?.ready && !!catalog.data?.runtime.available;
  const changed = latest?.result.changed_files ?? 0;
  const stale = latest?.result.stale_images ?? 0;
  const options = { model: model?.id, confidence: settings.confidence, padding: settings.padding, feather: settings.feather, device: cuda ? settings.device : 'cpu' };
  // Heads are detected first and written only after review; the newer of the two runs is the one shown.
  const newest = detection && (!latest || (detection.created_at ?? 0) > (latest.created_at ?? 0)) ? detection : latest;
  const reviewing = !!detection && newest === detection && detection.status === 'completed' && !detection.result.applied_by && !detection.result.dismissed && (detection.result.heads ?? 0) > 0;
  const detect = () => onStart({ action: 'detectheads', dataset_ids: scopes.ids(chosenScope), automask: options });
  const start = async () => {
    setError('');
    try { await detect(); } catch (e) { setError(formatApiError(e)); }
  };
  const write = (selections: HeadSelection[]) => onStart({ action: 'automask', automask: { ...options, proposal_id: detection!.id, selections } });
  const dismiss = async () => {
    await apiClient.post(`/dataset-pipeline/operations/${detection!.id}/dismiss`, undefined, { silent: true });
    onRefresh();
  };
  const written = changed ? text(`已为 ${changed} 张图片写入遮罩。`, `Masks written for ${changed} images.`)
    : latest?.result.proposal_id ? text('选中的头部已在遮罩中。', 'The chosen heads were already masked.') : text('没有检测到需要遮罩的头部。', 'No heads needed masking.');
  return <section className="vision-panel" aria-label={text('自动遮罩', 'Automatic masks')} data-testid="automask-panel">
    <header className="vision-panel-head"><h3><ScanFace size={16}/>{text('自动遮罩', 'Automatic masks')}</h3></header>
    <VisionRuntimeNotice catalog={catalog}/>
    <div className="vision-panel-body">
      <VisionModelField role="head_detector" catalog={catalog} value={model?.id || ''} disabled={locked}
        label={text('检测模型', 'Detector')} hint={text('找出每张图片中的头部，头部区域不参与训练。', 'Finds every head; head areas are left out of training.')}/>
      <div className="vision-row">
        <ScopeField scopes={scopes} value={chosenScope} onChange={setScope} disabled={locked} label={text('处理范围', 'Images')} hint={text('已有遮罩会保留，头部区域叠加进去。', 'Existing masks stay; head areas are added to them.')}/>
        <RangeField label={text('置信度', 'Confidence')} hint={text('越高误检越少，但可能漏掉小的头部，常用 0.41。', 'Higher gives fewer false hits but can miss small heads; 0.41 is typical.')}
          value={settings.confidence} min={0.05} max={0.95} step={0.01} disabled={locked} onChange={value => update({ confidence: value })}/>
        <RangeField label={text('扩展比例', 'Padding')} hint={text('头部框向外扩大的比例，覆盖头发与头饰。', 'How far each head box grows to cover hair and accessories.')}
          value={settings.padding} min={0} max={1} step={0.01} disabled={locked} onChange={value => update({ padding: value })}/>
        <RangeField label={text('羽化比例', 'Feather')} hint={text('边缘渐变的宽度，0 为硬边。', 'Width of the soft edge; 0 is a hard edge.')}
          value={settings.feather} min={0} max={0.5} step={0.01} disabled={locked} onChange={value => update({ feather: value })}/>
        {cuda && <DeviceField value={settings.device} disabled={locked} onChange={device => update({ device })}/>}
      </div>
    </div>
    {reviewing && !running ? <HeadMaskReview key={detection!.id} operation={detection!} locked={locked || !ready} onWrite={write} onDetectAgain={detect} onDismiss={dismiss}/>
      : <footer className="vision-panel-actions">{running ? <div className="vision-running">{running}</div> : <>
        <button type="button" className="ui-btn ui-btn-primary" disabled={locked || !ready || !chosenScope} onClick={() => void start()}><ScanFace size={15}/>{text('检测头部', 'Detect heads')}</button>
        {error && <p role="alert" className="vision-error">{error}</p>}
        {newest === detection && detection?.status === 'completed' && !detection.result.heads && !detection.result.dismissed
          ? <p role="status" className="vision-result">{text('没有检测到头部。', 'No heads were found.')}{detection.result.unreadable ? text(` ${detection.result.unreadable} 张图片无法读取。`, ` ${detection.result.unreadable} images could not be read.`) : ''}</p>
          : newest === latest && <OperationResult operation={latest} locked={locked} onUndo={onUndo} undoLabel={text('撤销本次遮罩', 'Undo these masks')}
            done={written + (stale ? text(` ${stale} 张图片在检测后有改动，没有写入。`, ` ${stale} images changed after detection and were skipped.`) : '')}/>}
      </>}</footer>}
  </section>;
}
