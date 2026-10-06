import type { FamilyInfo, ModelAsset } from '../../api/types';
import type { components } from '../../api/generated';
import { apiClient } from '../../api/client';
import { fillDefaultModels, MODEL_PATH_FIELDS } from '../../utils/workspaceConfig';
import { modelAssetUnsupportedReason, modelFamilyWeights, trainingFamilyOptions } from '../../utils/trainingFamilies';
import { useWorkspaceText } from '../../utils/workspaceText';
import { PathInput } from '../PathBrowser';
import StudioSelect from '../StudioSelect';
import ConfigHelp from '../ConfigHelp';
import Switch from '../Switch';

export type GenerationModel = Partial<components['schemas']['ModelConfig']> & { family: components['schemas']['ModelConfig']['family'] };

export default function RegularizationModelFields({ model, families, assets, disabled, onChange }: {
  model: GenerationModel; families: FamilyInfo[]; assets: ModelAsset[]; disabled: boolean; onChange: (model: GenerationModel) => void;
}) {
  const text = useWorkspaceText();
  const family = families.find(item => item.name === model.family);
  const weights = modelFamilyWeights(family);
  const setPath = (field: string, value: string) => {
    const next: GenerationModel = { ...model, [field]: value || null };
    if (field === 'dit_path' && model.family === 'krea2') {
      const asset = assets.find(item => item.family === 'krea2' && item.kind === 'dit' && item.path === value);
      next.krea2_variant = asset?.variant === 'raw' || asset?.variant === 'turbo' ? asset.variant : 'auto';
    }
    if (field === 'dit_path' && model.family === 'sdxl') {
      delete next.prediction_type;
      delete next.zero_terminal_snr;
    }
    onChange(next);
  };
  return <fieldset className="reg-model-fields" disabled={disabled}>
    <legend>{text('生成模型', 'Generation model')}<ConfigHelp label={text('生成模型说明', 'Generation model help')}>{text('使用所选底模生成正则图，不加载 LoRA。', 'Generate regularization images with the selected base model, without LoRA.')}</ConfigHelp></legend>
    <div className="reg-form-grid reg-model-grid">
      <label>{text('模型类型', 'Model type')}<StudioSelect aria-label={text('生成模型类型', 'Generation model type')} value={model.family} disabled={disabled} options={trainingFamilyOptions(families, text('zh', 'en') === 'en', model.family)} onValueChange={value => {
        const next = { family: value as GenerationModel['family'], dtype: 'auto' as const };
        onChange(fillDefaultModels({ model: next }, assets).model);
      }}/></label>
      {weights.map(weight => {
        const field = weight.field as keyof typeof MODEL_PATH_FIELDS;
        const value = model[field] || '';
        const matches = assets.filter(asset => asset.exists && asset.kind === weight.kind && (asset.family === model.family || asset.compatible_families?.includes(model.family)) && !modelAssetUnsupportedReason(asset));
        return <label key={field}><span className="reg-model-label">{weight.label}{weight.hint && <ConfigHelp label={text(`${weight.label}说明`, `${weight.label} help`)}>{weight.hint}</ConfigHelp>}</span><div className="reg-model-path">
          <PathInput ariaLabel={weight.label} value={value} placeholder={weight.required ? undefined : text('可选', 'Optional')} onChange={next => setPath(field, next)} resolveDefaultPath={async () => (await apiClient.get<{path:string}>('/models/browse-root', { params: { kind: weight.kind }, silent: true })).path}/>
          {!!matches.length && <StudioSelect aria-label={`${weight.label} · ${text('从已注册模型选择', 'Select a registered model')}`} disabled={disabled} value={matches.some(asset => asset.path === value) ? value : ''} placeholder={text('从已注册模型选择', 'Select a registered model')} options={matches.map(asset => ({ value: asset.path, label: asset.path.split(/[\\/]/).pop() || asset.path }))} onValueChange={next => setPath(field, next)}/>}
        </div></label>;
      })}
      {model.family === 'krea2' && <label>{text('Krea 2 类型', 'Krea 2 variant')}<StudioSelect aria-label={text('Krea 2 类型', 'Krea 2 variant')} value={model.krea2_variant || 'auto'} disabled={disabled} options={[{value:'auto',label:text('读取模型记录', 'Read model record')},{value:'raw',label:'Raw'},{value:'turbo',label:'Turbo'}]} onValueChange={value=>onChange({...model,krea2_variant:value as GenerationModel['krea2_variant']})}/></label>}
      {model.family === 'sdxl' && <>
        <label><span className="reg-model-label">{text('SDXL 预测方式', 'SDXL prediction type')}<ConfigHelp label={text('SDXL 预测方式说明', 'SDXL prediction type help')}>{text('自动读取底模：读取模型中的预测方式与 Zero SNR 标记；没有标记时使用噪声预测。\n噪声预测：用于常规 SDXL。\n速度预测：用于 v-prediction 模型；模型未保存标记时可手动选择。', 'Read from base model: use its prediction and Zero SNR markers; use epsilon when unmarked.\nEpsilon: for standard SDXL.\nv-prediction: select manually for v-prediction models without markers.')}</ConfigHelp></span><StudioSelect aria-label={text('SDXL 预测方式', 'SDXL prediction type')} value={model.prediction_type || 'auto'} disabled={disabled} options={[{value:'auto',label:text('自动读取底模', 'Read from base model')},{value:'epsilon',label:text('噪声预测', 'Epsilon')},{value:'v_prediction',label:text('速度预测', 'v-prediction')}]} onValueChange={value=>{
          const next = {...model};
          if(value === 'auto') {delete next.prediction_type;delete next.zero_terminal_snr;}
          else {next.prediction_type=value as GenerationModel['prediction_type'];next.zero_terminal_snr=value === 'v_prediction' && !!model.zero_terminal_snr;}
          onChange(next);
        }}/></label>
        {model.prediction_type && <div className="reg-model-switch"><Switch checked={!!model.zero_terminal_snr} disabled={disabled || model.prediction_type !== 'v_prediction'} onCheckedChange={value=>onChange({...model,zero_terminal_snr:value})}>{text('零终点信噪比（Zero SNR）', 'Zero terminal SNR')}</Switch><ConfigHelp label={text('零终点信噪比说明', 'Zero terminal SNR help')}>{text('仅在所选 v-prediction 模型要求 Zero SNR 时开启。', 'Enable only when the selected v-prediction model requires zero terminal SNR.')}</ConfigHelp></div>}
      </>}
    </div>
  </fieldset>;
}
