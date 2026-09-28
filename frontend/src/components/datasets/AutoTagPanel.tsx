import { useState, type ReactNode } from 'react';
import { Tags } from 'lucide-react';
import type { TaggingOptions } from '../../api/types';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import StudioSelect from '../StudioSelect';
import VisionModelField, { VisionRuntimeNotice } from './VisionModelField';
import { DeviceField, OperationResult, RangeField, ScopeField } from './VisionPanelParts';
import { useRememberedSettings, useScopeOptions, useVisionModels } from './visionHooks';
import type { PipelineOperation } from './DatasetPipelinePanel';
import './dataset-vision.css';

type Settings = Required<Pick<TaggingOptions, 'model' | 'general_threshold' | 'character_threshold' | 'existing' | 'device'>> & { exclude: string; trigger: string };
const DEFAULTS: Settings = { model: 'wd-eva02-large-tagger-v3', general_threshold: 0.35, character_threshold: 0.85, existing: 'skip', device: 'auto', exclude: '', trigger: '' };

export default function AutoTagPanel({ projectId, versionId, locked, latest, header, running, onStart, onUndo, onReview }: {
  projectId: string; versionId: string; locked: boolean; latest?: PipelineOperation; header?: ReactNode; running?: ReactNode;
  onStart: (body: Record<string, unknown>) => Promise<void>; onUndo: (id: string) => void; onReview: () => void;
}) {
  const text = useWorkspaceText();
  const catalog = useVisionModels();
  const scopes = useScopeOptions(projectId, versionId);
  const [settings, update] = useRememberedSettings('studio.autotag.settings', DEFAULTS);
  const [scope, setScope] = useState('');
  const [error, setError] = useState('');
  const chosenScope = scopes.options.some(option => option.value === scope) ? scope : scopes.first;
  const model = catalog.data?.models.find(item => item.id === settings.model && item.role === 'tagger') || catalog.data?.models.find(item => item.role === 'tagger');
  const cuda = !!catalog.data?.runtime.providers.includes('cuda');
  const ready = !!model?.ready && !!catalog.data?.runtime.available;
  const changed = latest?.result.changed_files ?? 0;
  const start = async () => {
    setError('');
    try {
      await onStart({
        action: 'autotag',
        dataset_ids: scopes.ids(chosenScope),
        tagging: {
          model: model?.id, general_threshold: settings.general_threshold, character_threshold: settings.character_threshold,
          existing: settings.existing, device: cuda ? settings.device : 'cpu', trigger_word: settings.trigger.trim() || null,
          exclude_tags: settings.exclude.split(',').map(tag => tag.trim()).filter(Boolean),
        },
      });
    } catch (e) { setError(formatApiError(e)); }
  };
  return <section className="vision-panel" aria-label={text('Tagger 模型打标', 'Tagger model')} data-testid="autotag-panel">
    <header className="vision-panel-head">{header ?? <h3><Tags size={16}/>{text('Tagger 模型打标', 'Tagger model')}</h3>}</header>
    <VisionRuntimeNotice catalog={catalog}/>
    <div className="vision-panel-body">
      <VisionModelField role="tagger" catalog={catalog} value={model?.id || ''} disabled={locked} onChange={id => update({ model: id })}
        label={text('打标模型', 'Tagger model')} hint={text('识别画面内容与角色，输出 Danbooru 标签。', 'Reads the image and its characters as Danbooru tags.')}/>
      <div className="vision-row">
        <ScopeField scopes={scopes} value={chosenScope} onChange={setScope} disabled={locked} label={text('打标范围', 'Images')} hint={text('暂不训练的图片不会打标。', 'Images held out of training are skipped.')}/>
        <div className="vision-field"><span className="vision-field-label">{text('已有标签', 'Existing captions')}</span>
          <StudioSelect aria-label={text('已有标签', 'Existing captions')} value={settings.existing} disabled={locked} onValueChange={value => update({ existing: value as Settings['existing'] })} options={[
            { value: 'skip', label: text('跳过已有标签的图片', 'Skip captioned images') },
            { value: 'append', label: text('追加到已有标签后', 'Add after existing tags') },
            { value: 'prepend', label: text('添加到已有标签前', 'Add before existing tags') },
            { value: 'overwrite', label: text('覆盖已有标签', 'Replace existing tags') },
          ]}/>
          <span className="vision-field-hint">{text('重复的标签只保留一个。', 'A tag already present is not added twice.')}</span></div>
        <label className="vision-field"><span className="vision-field-label">{text('触发词', 'Trigger word')}</span>
          <input type="text" aria-label={text('触发词', 'Trigger word')} value={settings.trigger} maxLength={200} disabled={locked} placeholder={text('可选，例如 mychar', 'Optional, e.g. mychar')} onChange={event => update({ trigger: event.target.value })}/>
          <span className="vision-field-hint">{text('写在每条标签的最前面。', 'Placed first in every caption.')}</span></label>
        <label className="vision-field"><span className="vision-field-label">{text('排除标签', 'Excluded tags')}</span>
          <input type="text" aria-label={text('排除标签', 'Excluded tags')} value={settings.exclude} disabled={locked} placeholder={text('例如 simple background', 'e.g. simple background')} onChange={event => update({ exclude: event.target.value })}/>
          <span className="vision-field-hint">{text('这些标签不会写入，用逗号分隔。', 'Never written; separate with commas.')}</span></label>
      </div>
      <div className="vision-row">
        <RangeField label={text('通用标签阈值', 'General tag threshold')} hint={text('越低标签越多，也越容易出错，常用 0.35。', 'Lower adds more tags and more mistakes; 0.35 is typical.')}
          value={settings.general_threshold} min={0.05} max={0.95} step={0.01} disabled={locked} onChange={value => update({ general_threshold: value })}/>
        <RangeField label={text('角色标签阈值', 'Character tag threshold')} hint={text('角色名的把握要求，常用 0.85。', 'Confidence needed for character names; 0.85 is typical.')}
          value={settings.character_threshold} min={0.05} max={0.95} step={0.01} disabled={locked} onChange={value => update({ character_threshold: value })}/>
        {cuda && <DeviceField value={settings.device} disabled={locked} onChange={device => update({ device })}/>}
      </div>
    </div>
    <footer className="vision-panel-actions">{running ? <div className="vision-running">{running}</div> : <>
      <button type="button" className="ui-btn ui-btn-primary" disabled={locked || !ready || !chosenScope} onClick={() => void start()}><Tags size={15}/>{text('开始打标', 'Start tagging')}</button>
      {error && <p role="alert" className="vision-error">{error}</p>}
      <OperationResult operation={latest} locked={locked} onUndo={onUndo} undoLabel={text('撤销本次打标', 'Undo this run')}
        done={changed ? text(`已写入 ${changed} 个标签文件。`, `Wrote ${changed} caption files.`) : text('没有需要写入的标签。', 'No captions needed writing.')}>
        {!!changed && <button type="button" className="ui-link" onClick={onReview}>{text('查看标签', 'Review captions')}</button>}
      </OperationResult>
    </>}</footer>
  </section>;
}
