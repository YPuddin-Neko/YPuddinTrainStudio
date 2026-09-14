import React from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import { apiClient } from '../api/client';
import { useWorkspaceText } from '../utils/workspaceText';
import { formatApiError } from '../utils/errors';

export interface ServiceRuntime {
  worker_id: number; managed: boolean; can_restart: boolean; reason: string | null;
  current_host: string; current_port: number; saved_host: string; saved_port: number;
  current_python: string; can_restore_original: boolean; selected_environment: string | null;
}
const reasons: Record<string, [string,string]> = {
  start_with_studio_launcher:['请用项目启动脚本运行服务后重启。','Start the service through the project launcher to enable restart.'],
  restart_in_progress:['服务正在重启。','The service is restarting.'],
  training_or_data_worker_running:['训练或数据任务运行中，请完成或停止后重启。','Finish or stop the training/data task before restarting.'],
  extension_operation_running:['等待扩展安装完成。','Wait for the extension operation to finish.'],
  torch_operation_running:['等待 PyTorch 环境准备完成。','Wait for the PyTorch environment to finish preparing.'],
  model_download_running:['等待模型下载完成或取消下载。','Finish or cancel model downloads first.'],
  data_operation_running:['等待数据处理完成。','Wait for data processing to finish.'],
  version_operation_running:['等待版本操作完成。','Wait for the version operation to finish.'],
};

async function requestWithTimeout<T>(request: (signal: AbortSignal) => Promise<T>, timeoutMs: number, controllers: Set<AbortController>): Promise<T> {
  const controller = new AbortController();
  controllers.add(controller);
  let timer: number | undefined;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = window.setTimeout(() => {
      controller.abort();
      reject(new Error('Service request timed out'));
    }, Math.max(1, timeoutMs));
  });
  try { return await Promise.race([request(controller.signal), timeout]); }
  finally { window.clearTimeout(timer); controllers.delete(controller); }
}

export default function ServiceControls({environmentId, onRestarted}: {environmentId?: string; onRestarted?:()=>void}) {
  const text = useWorkspaceText();
  const [runtime, setRuntime] = React.useState<ServiceRuntime | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState('');
  const [notice,setNotice] = React.useState('');
  const [nextAddress,setNextAddress] = React.useState('');
  const cancelled=React.useRef(false);
  const controllers=React.useRef(new Set<AbortController>());
  const refresh=React.useCallback(async(timeoutMs=3000)=> {const value=await requestWithTimeout(signal=>apiClient.get<ServiceRuntime>('/service/runtime',{silent:true,signal}),timeoutMs,controllers.current);if(!cancelled.current)setRuntime(value);return value;},[]);
  React.useEffect(()=>{const activeControllers=controllers.current;cancelled.current=false;void refresh().catch(e=>{if(!cancelled.current)setError(formatApiError(e));});return()=>{cancelled.current=true;for(const controller of activeControllers)controller.abort();};},[refresh]);
  const restart=async(options:Record<string,unknown>)=>{
    setBusy(true);setError('');setNotice('');setNextAddress('');
    const deadline=Date.now()+60000;
    try {
      const before=await refresh();
      if(cancelled.current)return;
      const result=await requestWithTimeout(signal=>apiClient.post<{address_changed:boolean;host:string;port:number;reconnect_url:string}>('/service/restart',options,{silent:true,signal}),Math.min(10000,deadline-Date.now()),controllers.current);
      if(cancelled.current)return;
      if(result.address_changed){
        const address=new URL(window.location.href); address.port=String(result.port); if(!['0.0.0.0','::'].includes(result.host))address.hostname=result.host;
        setNextAddress(address.toString());setNotice(text('服务正在切换地址，使用下方入口重新连接。','The service is changing its address. Reconnect using the link below.'));return;
      }
      setNotice(text('正在重启并重新连接…','Restarting and reconnecting…'));
      while(Date.now()<deadline){
        await new Promise(resolve=>window.setTimeout(resolve,Math.min(1000,deadline-Date.now())));
        if(cancelled.current)return;
        if(Date.now()>=deadline)break;
        try {const current=await refresh(Math.min(3000,deadline-Date.now()));if(cancelled.current)return;if(current.worker_id!==before.worker_id){if(typeof options.environment_id==='string'&&current.selected_environment!==options.environment_id){setError(text('新环境启动失败，服务已恢复原环境。请查看启动日志后重试。','The prepared environment could not start. The original environment was restored; check the launcher log.'));setNotice('');onRestarted?.();return;}setNotice(text('服务已重启。','Service restarted.'));onRestarted?.();return;}}catch{/* Expected while the owned worker restarts. */}
      }
      setNotice('');
      setError(text('暂未重新连接。请查看启动窗口；恢复后可点击重新检测。','Reconnection timed out. Check the launcher window, then refresh status.'));
    }catch(e){if(!cancelled.current){setNotice('');setError(formatApiError(e));}}finally{if(!cancelled.current)setBusy(false);}
  };
  return <div className="service-controls">
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" className="settings-action" disabled={busy||!runtime?.can_restart} onClick={()=>void restart(environmentId?{environment_id:environmentId}:{})}>{busy?<Loader2 size={14} className="animate-spin"/>:<RefreshCw size={14}/>} {environmentId?text('重启并切换到此环境','Restart in this environment'):text('重启服务','Restart service')}</button>
      {!environmentId && runtime?.can_restore_original && <button type="button" className="settings-input" disabled={busy||!runtime.can_restart} onClick={()=>void restart({restore_original_environment:true})}>{text('恢复原环境并重启','Restore original environment')}</button>}
      {!environmentId && runtime && (runtime.saved_host!==runtime.current_host||runtime.saved_port!==runtime.current_port) && <button type="button" className="settings-input" disabled={busy||!runtime.can_restart} onClick={()=>void restart({apply_saved_address:true})}>{text(`应用已保存地址并重启（${runtime.saved_host}:${runtime.saved_port}）`,`Restart at saved address (${runtime.saved_host}:${runtime.saved_port})`)}</button>}
      <button type="button" className="settings-input" disabled={busy} onClick={()=>void refresh().then(()=>setError('')).catch(e=>setError(formatApiError(e)))}>{text('重新检测','Refresh status')}</button>
    </div>
    {runtime?.reason&&<p className="settings-note">{reasons[runtime.reason]?text(...reasons[runtime.reason]):text('当前有操作占用服务，请稍后重试。','The service is busy. Try again later.')}</p>}
    {notice&&<p role="status" className="settings-note">{notice}</p>}
    {nextAddress&&<a className="text-[var(--studio-accent)]" href={nextAddress}>{text('打开新的服务地址','Open the new service address')}</a>}
    {error&&<p role="alert" className="settings-alert">{error==='Service request timed out'?text('服务暂未响应，请重新检测。','The service did not respond. Refresh status to retry.'):error}</p>}
  </div>;
}
