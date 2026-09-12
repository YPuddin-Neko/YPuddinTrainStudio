import {useWorkspaceText} from '../utils/workspaceText';
import StudioSelect from './StudioSelect';

export default function CaptionFormatSelect({value='auto',onChange,disabled=false,label}: {value?:string;onChange:(value:string)=>void;disabled?:boolean;label?:string}) {
  const text=useWorkspaceText();
  const known=['auto','.txt','.json'];
  const custom=!known.includes(value);
  return <div className="caption-format-control">
    <StudioSelect aria-label={label || text('标签格式','Caption format')} disabled={disabled} value={custom?'custom':value} onValueChange={next=>onChange(next==='custom'?'.caption':next)} options={[
      {value:'auto',label:text('自动 · JSON 优先，其次 TXT','Auto · JSON before TXT')},
      {value:'.txt',label:text('仅 TXT','TXT only')},{value:'.json',label:text('仅 JSON','JSON only')},
      {value:'custom',label:text('自定义扩展名…','Custom extension…')},
    ]}/>
    {custom && <input aria-label={text('自定义标签扩展名','Custom caption extension')} value={value} disabled={disabled} onChange={event=>onChange(event.target.value)} placeholder=".caption"/>}
  </div>;
}
