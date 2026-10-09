import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTtsInstalledModels } from '../../api/hooks/useTtsModels';
import { Download, FolderSearch, RefreshCw } from 'lucide-react';
import { ttsModelMatches, ttsModelSelectable, ttsModelsUrl, type TtsInstalledModel, type TtsModelEngine, type TtsModelVariant } from '../../api/ttsModels';
import StudioSelect from '../../components/StudioSelect';
import Dialog from '../../components/Dialog';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import './models.css';

export interface TtsModelScope {
  engine: TtsModelEngine;
  variant?: TtsModelVariant;
  value?: string;
}
export interface TtsModelPickerProps extends TtsModelScope {
  disabled?: boolean;
  getScope?: () => TtsModelScope | null;
  onSelect: (installation: TtsInstalledModel) => void;
}

export default function TtsModelPicker({ engine, variant, value, disabled, getScope, onSelect }: TtsModelPickerProps) {
  const text = useWorkspaceText(), location = useLocation();
  const [scope, setScope] = React.useState<TtsModelScope | null>(null), [chosen, setChosen] = React.useState('');
  const models = useTtsInstalledModels(!!scope);
  const compatible = (models.data || []).filter(model => scope && ttsModelMatches(model, scope.engine, scope.variant));
  const names = new Map<string, number>();
  compatible.forEach(model => names.set(model.name, (names.get(model.name) || 0) + 1));
  const selected = compatible.find(model => model.id === chosen || !chosen && model.bindings.model_path === scope?.value);
  const selectable = !!scope && !!selected && ttsModelSelectable(selected, scope.engine, scope.variant);
  const status = (model: TtsInstalledModel) => model.status === 'missing' ? text('文件缺失', 'Missing files') : model.status === 'changed' ? text('文件已变化', 'Files changed') : text('不可用', 'Unavailable');
  const open = () => {
    if (disabled) return;
    const next = getScope ? getScope() : { engine, variant, value };
    if (!next) return;
    setChosen(''); setScope({ ...next });
  };
  return <><button type="button" className="ui-btn ui-btn-sm" disabled={disabled} onClick={open}><FolderSearch size={14}/>{text('选择模型', 'Choose model')}</button>{scope && <Dialog title={text('选择语音模型', 'Choose speech model')} onClose={() => setScope(null)}>
    <div className="tts-model-picker" data-testid="tts-model-picker"><div className="tts-model-picker-controls">
      <StudioSelect aria-label={text('选择已下载的语音模型', 'Choose a downloaded speech model')} value={selected?.id || ''} disabled={disabled || models.isFetching || !!models.error || !compatible.length}
        placeholder={models.isPending ? text('正在读取模型…', 'Loading models…') : text('选择已下载模型', 'Choose a downloaded model')} searchable
        options={compatible.map(model => ({ value: model.id, label: `${model.name}${(names.get(model.name) || 0) > 1 ? ` · ${model.path}` : ''}${ttsModelSelectable(model, scope.engine, scope.variant) ? '' : ` · ${status(model)}`}`, disabled: !ttsModelSelectable(model, scope.engine, scope.variant) }))}
        onValueChange={setChosen}/>
      <button type="button" className="ui-btn ui-btn-sm ui-btn-icon" disabled={models.isFetching} onClick={() => void models.refetch()} aria-label={text('刷新可选模型', 'Refresh available models')} title={text('刷新可选模型', 'Refresh available models')}><RefreshCw size={14} className={models.isFetching ? 'animate-spin' : undefined}/></button>
    </div>
    {selected && <code className="tts-model-picker-path">{selected.bindings.model_path}</code>}
    {models.error ? <p role="alert" className="tts-model-picker-error">{formatApiError(models.error)}</p> : !models.isPending && !compatible.some(model => ttsModelSelectable(model, scope.engine, scope.variant)) ? <p className="model-help-text">{text('暂无可用的匹配模型，可下载模型包或填写本地模型路径。', 'No matching model is ready. Download a package or enter a local model path.')}</p> : null}
    <div className="tts-model-picker-actions"><Link className="ui-btn ui-btn-sm" to={ttsModelsUrl(scope.engine, scope.variant)} state={{ backgroundLocation: location }} onClick={() => setScope(null)}><Download size={14}/>{text('下载模型', 'Download models')}</Link><span/><button type="button" className="ui-btn ui-btn-sm" onClick={() => setScope(null)}>{text('取消', 'Cancel')}</button><button type="button" className="ui-btn ui-btn-sm ui-btn-primary" disabled={disabled || models.isFetching || !!models.error || !selectable} onClick={() => { if (selected && selectable && !disabled && !models.error && !models.isFetching) { onSelect(selected); setScope(null); } }}>{text('使用模型', 'Use model')}</button></div>
    </div></Dialog>}</>;
}
