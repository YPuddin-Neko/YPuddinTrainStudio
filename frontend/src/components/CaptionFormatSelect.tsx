import {useWorkspaceText} from '../utils/workspaceText';
import StudioSelect from './StudioSelect';

export default function CaptionFormatSelect({value='auto',onChange,disabled=false,label,formats=['txt','json']}: {value?:string;onChange:(value:string)=>void;disabled?:boolean;label?:string;formats?:readonly string[]}) {
  const text=useWorkspaceText();
  const known=['auto','.txt','.json'];
  const normalized=value.toLowerCase();
  const custom=!known.includes(normalized);
  const supportsJson=formats.includes('json');
  const unsupported=normalized === '.json' && !supportsJson;
  return <div className="caption-format-control">
    <StudioSelect aria-label={label || text('标签格式','Caption format')} aria-invalid={unsupported || undefined} disabled={disabled} value={unsupported?'':custom?'custom':normalized} placeholder={text('请选择受支持的格式','Choose a supported format')} onValueChange={next=>onChange(next==='custom'?'.caption':next)} options={[
      {value:'auto',label:supportsJson ? text('自动 · JSON 优先，其次 TXT','Auto · JSON before TXT') : text('自动 · 仅查找 TXT','Auto · TXT only')},
      {value:'.txt',label:text('仅 TXT','TXT only')},...(supportsJson ? [{value:'.json',label:text('仅 JSON','JSON only')}] : []),
      {value:'custom',label:text('自定义扩展名…','Custom extension…')},
    ]}/>
    {unsupported && <p role="alert">{text('当前模型不支持 JSON 标签，请选择 TXT 或其他纯文本格式。','This model does not support JSON captions. Choose TXT or another plain-text format.')}</p>}
    {custom && <input aria-label={text('自定义标签扩展名','Custom caption extension')} value={value} disabled={disabled} onChange={event=>onChange(event.target.value)} placeholder=".caption"/>}
  </div>;
}
