import React from 'react';
import { Loader2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import { PathInput } from '../../components/PathBrowser';
import StudioSelect from '../../components/StudioSelect';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';

export interface ModelInspection {
  path: string; family: string | null; family_candidates: string[]; kind: string | null;
  dtype: string | null; dtypes: Record<string, number>; confidence: 'high' | 'partial' | 'unknown';
  evidence: string[]; warnings: string[]; files_inspected: number;
}

export default function LocalModelRegistration({ initialFamily, onClose, onRegistered, onBusyChange }: {initialFamily:string;onClose:()=>void;onRegistered:(family:string)=>Promise<void>;onBusyChange:(busy:boolean)=>void}) {
  const text = useWorkspaceText();
  const [path,setPath] = React.useState('');
  const [family,setFamily] = React.useState(['anima','krea2'].includes(initialFamily) ? initialFamily : '');
  const [kind,setKind] = React.useState('');
  const [dtype,setDtype] = React.useState('');
  const [detected,setDetected] = React.useState<ModelInspection|null>(null);
  const [detecting,setDetecting] = React.useState(false);
  const [saving,setSaving] = React.useState(false);
  const [isDefault,setIsDefault] = React.useState(false);
  const [error,setError] = React.useState('');
  const [reload,setReload] = React.useState(0);
  const inspectedPath = React.useRef('');
  React.useEffect(() => {
    const controller=new AbortController();
    setDetected(null);setKind('');setDtype('');setError('');inspectedPath.current='';
    if (!path.trim()) {setDetecting(false);return;}
    setDetecting(true);
    const timer=setTimeout(()=>{
      void apiClient.post<ModelInspection>('/models/inspect',{path:path.trim()},{signal:controller.signal,silent:true}).then(result=>{
        if(controller.signal.aborted)return;
        setDetected(result);setKind(result.kind||'');setDtype(result.dtype||'');
        setFamily(current=>result.family||(['anima','krea2'].includes(current)&&(!result.family_candidates.length||result.family_candidates.includes(current))?current:result.family_candidates[0]||''));
        inspectedPath.current=path.trim();setDetecting(false);
      }).catch(error=>{if(!controller.signal.aborted){setError(formatApiError(error));setDetecting(false);}});
    },300);
    return()=>{controller.abort();clearTimeout(timer);};
  },[path,reload]);
  const ready=!!detected && inspectedPath.current===path.trim();
  const labels:Record<string,string>={dit:text('主模型 / DiT','Base model / DiT'),vae:'VAE',text_encoder:text('文本编码器','Text encoder'),tokenizer:text('分词器目录','Tokenizer directory')};
  return <form className="model-source-form" data-testid="add-model-modal" onSubmit={event=>{
    event.preventDefault();if(!ready||!family||!kind||saving)return;
    setSaving(true);onBusyChange(true);setError('');
    void apiClient.post('/models',{family,kind,path:detected.path,dtype:dtype||null,is_default:isDefault},{silent:true}).then(()=>onRegistered(family)).catch(error=>setError(formatApiError(error))).finally(()=>{setSaving(false);onBusyChange(false);});
  }}>
    <p className="model-help-text">{text('选择本机权重文件或完整编码器目录后自动检测；只读文件头和配置，不加载整个模型。','Select a local weight file or complete encoder directory to inspect headers and configuration without loading the model.')}</p>
    <fieldset disabled={saving}>
      <label>{text('文件路径','File path')}<PathInput ariaLabel={text('文件路径','File path')} value={path} onChange={value=>{setPath(value);setDetected(null);inspectedPath.current='';}}/></label>
      {detecting&&<p role="status"><Loader2 size={14} className="animate-spin"/>{text('正在检测模型…','Inspecting model…')}</p>}
      {ready&&<p role="status" data-testid="model-inspection-status">{detected.confidence==='high'?text('已识别模型结构','Model structure identified'):text('部分信息未识别，请确认下方选项','Some information is unknown; confirm the choices below')}{text('；这不代替实际模型加载验收。','; this does not replace a real model-load validation.')}</p>}
      <div className="model-form-grid">
        <label>{text('模型系列','Model family')}<StudioSelect aria-label={text('登记模型系列','Registration model family')} value={family} disabled={!ready||!!detected?.family} onValueChange={setFamily} options={[{value:'',label:text('请选择','Choose')},...['anima','krea2'].filter(name=>!detected?.family_candidates.length||detected.family_candidates.includes(name)).map(value=>({value,label:value==='anima'?'Anima':'Krea 2'}))]}/></label>
        <label>{text('组件','Component')}<StudioSelect aria-label={text('组件','Component')} value={kind} disabled={!ready||!!detected?.kind} onValueChange={setKind} options={[{value:'',label:text('未识别，请选择','Unknown; choose')},...Object.keys(labels).map(value=>({value,label:labels[value]}))]}/></label>
        <label>{text('权重精度','Weight precision')}<StudioSelect aria-label={text('权重精度','Weight precision')} value={dtype} disabled={!ready||!!detected?.dtype||kind==='tokenizer'} onValueChange={setDtype} options={['','bf16','fp16','fp32','fp8','mixed'].map(value=>({value,label:value.toUpperCase()||text('未知','Unknown')}))}/></label>
      </div>
      {ready&&detected.warnings.map(message=><p className="model-help-text" key={message}>{message}</p>)}
      {ready&&<details><summary>{text('检测依据','Inspection evidence')}</summary><p>{detected.evidence.join(' · ')||text('没有匹配到已知结构。','No known structure matched.')}</p><p>{Object.entries(detected.dtypes).map(([type,count])=>`${type}: ${count.toLocaleString()}`).join(' · ')}</p></details>}
      <label className="model-checkbox"><input type="checkbox" checked={isDefault} onChange={event=>setIsDefault(event.target.checked)}/>{text('设为本系列默认组件','Set as this family’s default component')}</label>
    </fieldset>
    {error&&<div role="alert" className="settings-alert">{error}<button className="model-button" type="button" disabled={saving||detecting} onClick={()=>setReload(value=>value+1)}>{text('重新检测','Inspect again')}</button></div>}
    <footer><button className="model-button" type="button" disabled={saving} onClick={onClose}>{text('取消','Cancel')}</button><button className="model-button model-button-primary" type="submit" data-testid="add-model-submit" disabled={saving||detecting||!ready||!family||!kind}>{saving&&<Loader2 size={14} className="animate-spin"/>}{text('添加模型','Add model')}</button></footer>
  </form>;
}
