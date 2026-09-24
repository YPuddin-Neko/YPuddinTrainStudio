import type {Preset} from '../api/types';
import {useTranslation} from 'react-i18next';
import Dialog from './Dialog';
import {applyTrainingPreset} from '../utils/trainingPresets';
import {configFieldLabel} from '../utils/configPresentation';
import {useWorkspaceText} from '../utils/workspaceText';

function changedFields(current: Record<string, any>, next: Record<string, any>, prefix=''): Array<{path:string;before:unknown;after:unknown}> {
  return Object.entries(next).flatMap(([key,after])=>{
    const before=current?.[key],path=prefix?`${prefix}.${key}`:key;
    if(JSON.stringify(before)===JSON.stringify(after))return [];
    if(after && typeof after==='object' && !Array.isArray(after))return changedFields(before || {},after,path);
    return [{path,before,after}];
  });
}

export default function PresetPreview({preset,current,onClose,onApply}: {preset:Preset;current:Record<string,any>;onClose:()=>void;onApply:()=>void}) {
  const text=useWorkspaceText(),{i18n}=useTranslation();
  const changes=changedFields(current,applyTrainingPreset(current,preset.config));
  const display=(value:unknown)=>value===undefined||value===null?text('自动 / 未设置','Auto / unset'):typeof value==='boolean'?value?text('开','On'):text('关','Off'):typeof value==='object'?JSON.stringify(value):String(value);
  return <Dialog title={text('加载预设前确认参数','Review preset changes')} onClose={onClose} wide>
    <div className="preset-preview-body"><h3>{preset.name}</h3>{preset.description && <p>{preset.description}</p>}
      <p>{text('数据源和输出位置保持不变。','Data sources and output locations stay unchanged.')}</p>
      <p>{text(`将修改 ${changes.length} 个参数`,`${changes.length} parameters will change`)}</p>
      {changes.length>0 ? <div className="preset-preview-table"><table><thead><tr><th>{text('参数','Parameter')}</th><th>{text('当前','Current')}</th><th>{text('预设','Preset')}</th></tr></thead><tbody>{changes.map(change=><tr key={change.path}><th>{configFieldLabel(change.path,change.path,i18n.language.startsWith('en'))}<small>{change.path}</small></th><td>{display(change.before)}</td><td>{display(change.after)}</td></tr>)}</tbody></table></div> : <p>{text('当前参数已与此预设一致。','Current parameters already match this preset.')}</p>}
    </div><footer className="preset-preview-actions"><button type="button" onClick={onClose}>{text('取消','Cancel')}</button><button type="button" className="studio-primary" disabled={!changes.length} onClick={onApply}>{text('应用到当前版本','Apply to this version')}</button></footer>
  </Dialog>;
}
