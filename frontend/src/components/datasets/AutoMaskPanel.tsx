import { useState } from 'react';
import { ScanFace } from 'lucide-react';
import type { AutoMaskOptions } from '../../api/types';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import VisionModelField, { VisionRuntimeNotice } from './VisionModelField';
import { DeviceField, OperationResult, RangeField, ScopeField } from './VisionPanelParts';
import { useRememberedSettings, useScopeOptions, useVisionModels } from './visionHooks';
import type { PipelineOperation } from './DatasetPipelinePanel';
import './dataset-vision.css';

type Settings = Required<Pick<AutoMaskOptions, 'confidence' | 'padding' | 'feather' | 'device'>>;
const DEFAULTS: Settings = { confidence: 0.413, padding: 0.10, feather: 0.03, device: 'auto' };

export default function AutoMaskPanel({ projectId, versionId, locked, latest, onStart, onUndo }: {
  projectId: string; versionId: string; locked: boolean; latest?: PipelineOperation;
  onStart: (body: Record<string, unknown>) => Promise<void>; onUndo: (id: string) => void;
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
  const start = async () => {
    setError('');
    try {
      await onStart({ action: 'automask', dataset_ids: scopes.ids(chosenScope), automask: { model: model?.id, confidence: settings.confidence, padding: settings.padding, feather: settings.feather, device: cuda ? settings.device : 'cpu' } });
    } catch (e) { setError(formatApiError(e)); }
  };
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
    <footer className="vision-panel-actions">
      <button type="button" className="ui-btn ui-btn-primary" disabled={locked || !ready || !chosenScope} onClick={() => void start()}><ScanFace size={15}/>{text('生成遮罩', 'Create masks')}</button>
      {error && <p role="alert" className="vision-error">{error}</p>}
      <OperationResult operation={latest} locked={locked} onUndo={onUndo} undoLabel={text('撤销本次遮罩', 'Undo these masks')}
        done={changed ? text(`已为 ${changed} 张图片写入遮罩。`, `Masks written for ${changed} images.`) : text('没有检测到需要遮罩的头部。', 'No heads needed masking.')}/>
    </footer>
  </section>;
}
