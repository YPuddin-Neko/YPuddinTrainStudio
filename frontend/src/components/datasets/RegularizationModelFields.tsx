import { Link, useLocation } from 'react-router-dom';
import { RefreshCw } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';
import StudioSelect from '../StudioSelect';
import ConfigHelp from '../ConfigHelp';

export type GenerationModelOption = {
  model_id: string | null; family: string; name: string; variant: string | null;
  purpose: string | null; available: boolean; reason: string | null;
};
export type GenerationModels = { current: GenerationModelOption; models: GenerationModelOption[] };

export default function RegularizationModelFields({ options, value, disabled, refreshing, onChange, onRefresh }: {
  options: GenerationModels; value: string; disabled: boolean; refreshing: boolean;
  onChange: (id: string) => void; onRefresh: () => void;
}) {
  const text = useWorkspaceText();
  const location = useLocation();
  const selected = value ? options.models.find(model => model.model_id === value) : options.current;
  return <div className="reg-model-selection">
    <div className="reg-model-heading"><label htmlFor="regularization-model">{text('生成模型', 'Generation model')}<ConfigHelp label={text('生成模型说明', 'Generation model help')}>{text('使用所选底模生成正则图，不加载 LoRA。', 'Generate regularization images with the selected base model, without LoRA.')}</ConfigHelp></label>
      <Link className="ui-link" to={`/settings/environment?tab=models&family=${encodeURIComponent(selected?.family || options.current.family)}`} state={{ backgroundLocation: location.state?.backgroundLocation ?? location }}>{text('管理与下载模型', 'Manage & download models')}</Link>
    </div>
    <div className="reg-model-control"><StudioSelect id="regularization-model" aria-label={text('生成模型', 'Generation model')} searchable value={value} disabled={disabled}
      fallbackLabel={text('所选模型已不可用', 'The selected model is unavailable')} onValueChange={onChange}
      options={[{ value: '', label: text(`当前训练模型 · ${options.current.name}`, `Current training model · ${options.current.name}`) }, ...options.models.map(model => ({ value: model.model_id!, label: model.available ? model.name : `${model.name} · ${text('不可用', 'unavailable')}`, displayLabel: model.name }))]}/>
      <button type="button" className="ui-btn ui-btn-sm ui-btn-icon" disabled={disabled || refreshing} onClick={onRefresh} aria-label={text('刷新可选模型', 'Refresh available models')} title={text('刷新可选模型', 'Refresh available models')}><RefreshCw size={14} className={refreshing ? 'animate-spin' : undefined}/></button>
    </div>
    {(!selected || !selected.available) && <p className="reg-model-error" role="alert">{selected?.reason || text('所选模型已不可用，请重新选择或在模型管理中补齐。', 'The selected model is unavailable. Choose another or complete it in model management.')}</p>}
  </div>;
}
