import React from 'react';
import { apiClient } from '../api/client';
import { useWorkspaceText } from '../utils/workspaceText';
import { formatApiError } from '../utils/errors';
import StudioSelect from './StudioSelect';
import ServiceControls from './ServiceControls';

type Operation={plan?:{name:string;from_version?:string|null;version?:string|null;minimum_free_bytes?:number;not_copied?:string[]}[];id:string;build_id:string;status:string;phase:string;logs:string[];error:string|null;environment_id:string|null;dismissed_at:number|null};
type Snapshot={builds:{id:string;label:string;supported:boolean;reason:string|null;recommended:boolean;backend:string}[];operations:Operation[];current_python:string;selected_environment:string|null;disk_free_bytes:number;minimum_free_bytes?:number;optional_extensions?:string[]};
const active=(op:Operation)=>['planning','installing','verifying'].includes(op.status);
export default function TorchEnvironmentPanel({disabled=false}:{disabled?:boolean}){
  const text=useWorkspaceText();
  const [state,setState]=React.useState<Snapshot|null>(null);
  const [choice,setChoice]=React.useState('');
  const [busy,setBusy]=React.useState(false);
  const [error,setError]=React.useState('');
  const [history,setHistory]=React.useState(false);
  const refresh=React.useCallback(async()=>{const data=await apiClient.get<Snapshot>('/environment/torch',{silent:true});setState(data);setError('');setChoice(old=>old||data.builds.find(b=>b.supported&&b.recommended)?.id||data.builds.find(b=>b.supported)?.id||'');},[]);
  React.useEffect(()=>{void refresh().catch(e=>setError(formatApiError(e)));},[refresh]);
  React.useEffect(()=>{if(!state?.operations.some(active))return;const timer=window.setInterval(()=>void refresh().catch(e=>setError(formatApiError(e))),1500);return()=>window.clearInterval(timer);},[state,refresh]);
  const act=async(url:string,body={})=>{setBusy(true);setError('');try{await apiClient.post(url,body,{silent:true});await refresh();}catch(e){setError(formatApiError(e));}finally{setBusy(false);}};
  const locked=disabled||busy||!!state?.operations.some(active);
  const phases:Record<string,[string,string]>={creating_environment:['创建独立环境','Creating an isolated environment'],installing_pytorch:['下载并安装 PyTorch','Downloading and installing PyTorch'],installing_dependencies:['安装训练依赖','Installing training dependencies'],verifying:['验证计算与服务','Verifying compute and service'],ready_to_restart:['已验证，等待切换','Verified; ready to activate'],failed:['准备失败','Preparation failed'],cancelled:['已取消','Cancelled'],review:['等待确认安装','Review installation']};
  return <section id="environment-torch" data-settings-section tabIndex={-1} className="settings-section">
    <div className="settings-section-heading"><h2>{text('PyTorch 版本','PyTorch version')}</h2></div>
    <p className="settings-note">{text('在项目内准备独立环境，验证成功后重启切换。原环境保留；xFormers、FlashAttention 等扩展需为新版本重新安装。','Prepare an isolated environment inside this project, verify it, then restart to activate. The original environment is retained; compiled extensions need matching installations.')}</p>
    <div className="settings-field"><label>{text('选择版本与计算后端','Version and compute backend')}</label><div className="settings-field-control"><StudioSelect aria-label={text('选择 PyTorch 版本','Choose PyTorch version')} disabled={locked||!state} value={choice} onValueChange={setChoice} options={(state?.builds||[]).filter(b=>b.supported||b.reason?.startsWith('requires_driver')).map(b=>({value:b.id,label:b.label+(b.reason?.startsWith('requires_driver')?text(` · 需 NVIDIA ${b.reason.split('_').at(-1)}+ 驱动`,` · needs NVIDIA ${b.reason.split('_').at(-1)}+ driver`):''),disabled:!b.supported}))}/></div></div>
    {state&&<p className="settings-note">{text(`可用空间 ${(state.disk_free_bytes/1024**3).toFixed(1)} GiB；CUDA 环境至少预留 12 GiB，其他环境至少 8 GiB。`,`Available space: ${(state.disk_free_bytes/1024**3).toFixed(1)} GiB. Reserve at least 12 GiB for CUDA or 8 GiB for other environments.`)}</p>}
    {disabled&&<p className="settings-note">{text('当前任务完成后可准备运行环境。','Finish the current task before preparing an environment.')}</p>}
    <div className="flex flex-wrap gap-2"><button type="button" className="settings-action" disabled={locked||!choice} onClick={()=>void act('/environment/torch/operations',{build_id:choice})}>{text('检查切换计划','Review change plan')}</button><button type="button" className="settings-input" disabled={busy} onClick={()=>void refresh().catch(e=>setError(formatApiError(e)))}>{text('刷新版本状态','Refresh version status')}</button>{!!state?.operations.some(op=>op.dismissed_at)&&<button type="button" className="settings-input" aria-expanded={history} onClick={()=>setHistory(!history)}>{history?text('收起历史','Hide history'):text('历史记录','History')}</button>}</div>
    {error&&<p role="alert" className="settings-alert">{error}</p>}
    {state?.operations.filter(op=>history||!op.dismissed_at).map(op=><div key={op.id} className="torch-operation">
      <div className="settings-section-heading"><strong>{op.build_id}</strong><span className="settings-note">{phases[op.phase]?text(...phases[op.phase]):op.status}</span></div>
      {active(op)&&<progress aria-label={text('环境准备中','Preparing environment')}/>}
      {op.status==='ready'&&op.plan&&<dl className="torch-plan">{op.plan.filter(item=>['torch','torchvision'].includes(item.name)).map(item=><div key={item.name} className="settings-field"><dt>{item.name==='torch'?'PyTorch':'TorchVision'}</dt><dd className="settings-note">{item.from_version||text('未安装','Not installed')} → {item.version}</dd></div>)}</dl>}
      {op.status==='ready'&&<div className="flex flex-wrap gap-2"><button type="button" className="settings-action" disabled={locked} onClick={()=>void act(`/environment/torch/operations/${op.id}/apply`)}>{text('下载并准备此环境','Download and prepare this environment')}</button><button type="button" className="settings-input" disabled={busy} onClick={()=>void act(`/environment/torch/operations/${op.id}/cancel`)}>{text('取消','Cancel')}</button></div>}
      {op.error&&<p role="alert" className="settings-alert">{op.error}</p>}
      {op.logs.length>0&&<details className="settings-inline-details"><summary>{text('安装日志','Installation log')}</summary><pre className="max-h-56 overflow-auto whitespace-pre-wrap break-all">{op.logs.join('\n')}</pre></details>}
      {op.status==='completed'&&op.environment_id&&state.selected_environment!==op.environment_id&&<ServiceControls environmentId={op.environment_id} onRestarted={()=>void refresh()}/>}
      {op.status==='completed'&&state.selected_environment===op.environment_id&&<p className="settings-note">{text('当前正在使用','Currently active')}</p>}
      {active(op)&&<button type="button" className="settings-input" disabled={busy} onClick={()=>void act(`/environment/torch/operations/${op.id}/cancel`)}>{text('取消准备','Cancel preparation')}</button>}
      {!op.dismissed_at&&['completed','failed','cancelled'].includes(op.status)&&<button type="button" className="settings-input" disabled={busy} onClick={()=>void act(`/environment/torch/operations/${op.id}/dismiss`)}>{text('关闭结果','Dismiss result')}</button>}
    </div>)}
  </section>;
}
