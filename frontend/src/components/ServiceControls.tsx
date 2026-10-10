import React from 'react';
import { createPortal } from 'react-dom';
import { Loader2, RefreshCw } from 'lucide-react';
import { apiClient } from '../api/client';
import { ApiError } from '../api/types';
import { useWorkspaceText } from '../utils/workspaceText';
import { formatApiError } from '../utils/errors';
import { RestartRequiredContext } from './restartRequiredContext';
import { usePageVisible } from '../api/resourcePolicy';

export interface ServiceRuntime {
  worker_id: number; instance_id?: string | null; managed: boolean; can_restart: boolean; reason: string | null;
  current_host: string; current_port: number; saved_host: string; saved_port: number; restart_required?: boolean;
  current_python: string; can_restore_original: boolean; selected_environment: string | null;
}
const reasons: Record<string, [string,string]> = {
  start_with_studio_launcher:['请用项目启动脚本运行服务后重启。','Start the service through the project launcher to enable restart.'],
  restart_in_progress:['服务正在重启。','The service is restarting.'],
  training_or_data_worker_running:['训练或数据任务运行中，请完成或停止后重启。','Finish or stop the training/data task before restarting.'],
  extension_operation_running:['等待扩展安装完成。','Wait for the extension operation to finish.'],
  torch_operation_running:['等待 PyTorch 安装完成。','Wait for the PyTorch environment to finish preparing.'],
  model_download_running:['等待模型下载完成或取消下载。','Finish or cancel model downloads first.'],
  data_operation_running:['等待数据处理完成。','Wait for data processing to finish.'],
  version_operation_running:['等待版本操作完成。','Wait for the version operation to finish.'],
};

async function requestWithTimeout<T>(request: (signal: AbortSignal) => Promise<T>, timeoutMs: number, controllers: Set<AbortController>, controller = new AbortController()): Promise<T> {
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

export default function ServiceControls({environmentId, onRestarted, refreshTarget, secondary = false, disabled = false, refreshKey = 0, applySavedAddress = false, pending = false}: {environmentId?: string; onRestarted?:()=>void; refreshTarget?: HTMLElement | null; secondary?: boolean; disabled?: boolean; refreshKey?: number; applySavedAddress?: boolean; pending?: boolean}) {
  const text = useWorkspaceText();
  const visible = usePageVisible();
  const reportRestart = React.useContext(RestartRequiredContext);
  const [runtime, setRuntime] = React.useState<ServiceRuntime | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState('');
  const [notice,setNotice] = React.useState('');
  const [nextAddress,setNextAddress] = React.useState('');
  const cancelled=React.useRef(false);
  const restarting=React.useRef(false);
  const controllers=React.useRef(new Set<AbortController>());
  const statusRead=React.useRef<{controller:AbortController;promise:Promise<ServiceRuntime>}|null>(null);
  const refresh=React.useCallback((timeoutMs=3000,replace=false):Promise<ServiceRuntime>=> {
    if(statusRead.current&&!replace)return statusRead.current.promise;
    statusRead.current?.controller.abort();
    const controller=new AbortController();
    const current=()=>!cancelled.current&&statusRead.current?.controller===controller;
    const promise=requestWithTimeout(signal=>apiClient.get<ServiceRuntime>('/service/runtime',{silent:true,signal}),timeoutMs,controllers.current,controller)
      .then(value=>{
        if(!current()||controller.signal.aborted)throw Object.assign(new Error('Request replaced'),{name:'AbortError'});
        setRuntime(value);reportRestart(!!value.restart_required);return value;
      }).catch(failure=>{
        if(!current())throw Object.assign(new Error('Request replaced'),{name:'AbortError'});
        throw failure;
      }).finally(()=>{if(statusRead.current?.controller===controller)statusRead.current=null;});
    statusRead.current={controller,promise};
    return promise;
  },[reportRestart]);
  const showReadError=React.useCallback((failure:unknown)=>{
    if(!cancelled.current&&!(failure instanceof Error&&failure.name==='AbortError'))setError(formatApiError(failure));
  },[]);
  React.useEffect(()=>{const activeControllers=controllers.current;cancelled.current=false;return()=>{cancelled.current=true;statusRead.current=null;for(const controller of activeControllers)controller.abort();};},[]);
  React.useEffect(()=>{
    if(!visible||restarting.current)return;
    void refresh(3000,true).catch(showReadError);
    const passiveRead=statusRead.current;
    return()=>{
      // A restart can take over this read for its worker-identity precheck.
      if(!restarting.current&&statusRead.current===passiveRead){statusRead.current=null;passiveRead?.controller.abort();}
    };
  },[visible,refreshKey,refresh,showReadError]);
  const restart=async(options:Record<string,unknown>)=>{
    if(restarting.current)return;
    restarting.current=true;
    setBusy(true);setError('');setNotice('');setNextAddress('');
    const deadline=Date.now()+60000;
    try {
      const before=await refresh();
      if(cancelled.current)return;
      if(typeof options.environment_id==='string'&&before.selected_environment===options.environment_id){
        setNotice(text('当前正在使用此环境。','This environment is currently active.'));onRestarted?.();return;
      }
      let result: {address_changed:boolean;host:string;port:number;reconnect_url:string} | undefined;
      try {
        result=await requestWithTimeout(signal=>apiClient.post<typeof result>('/service/restart',options,{silent:true,signal}),Math.min(10000,deadline-Date.now()),controllers.current);
      } catch (e) {
        // A restart may be accepted even if its response is lost. Keep checking
        // the worker identity within the same deadline; never submit it twice.
        const connectionInterrupted=e instanceof TypeError || e instanceof Error && (e.message==='Service request timed out'||e.name==='AbortError');
        if(e instanceof ApiError || !connectionInterrupted) throw e;
      }
      if(cancelled.current)return;
      if(result?.address_changed){
        const address=new URL(window.location.href); address.port=String(result.port); if(!['0.0.0.0','::'].includes(result.host))address.hostname=result.host;
        setNextAddress(address.toString());setNotice(text('服务正在切换地址，使用下方入口重新连接。','The service is changing its address. Reconnect using the link below.'));return;
      }
      setNotice(text('正在重启并重新连接…','Restarting and reconnecting…'));
      while(Date.now()<deadline){
        await new Promise(resolve=>window.setTimeout(resolve,Math.min(1000,deadline-Date.now())));
        if(cancelled.current)return;
        if(Date.now()>=deadline)break;
        try {
          const current=await refresh(Math.min(3000,deadline-Date.now()));
          if(cancelled.current)return;
          const changed=before.instance_id&&current.instance_id ? before.instance_id!==current.instance_id : current.worker_id!==before.worker_id;
          if(changed){
            if(typeof options.environment_id==='string'&&current.selected_environment!==options.environment_id){setError(text('新环境启动失败，服务已恢复原环境。请查看启动日志后重试。','The prepared environment could not start. The original environment was restored; check the launcher log.'));setNotice('');onRestarted?.();return;}
            setNotice(text('服务已重启。','Service restarted.'));onRestarted?.();return;
          }
        }catch{/* Expected while the owned worker restarts. */}
      }
      setNotice('');
      setError(text('暂未重新连接。请查看启动窗口；恢复后可点击刷新状态。','Reconnection timed out. Check the launcher window, then refresh status.'));
    }catch(e){if(!cancelled.current){setNotice('');setError(formatApiError(e));}}finally{restarting.current=false;if(!cancelled.current)setBusy(false);}
  };
  // A saved setting or an environment change the running service has not applied yet. The saved and
  // running addresses are not compared: a launcher --port override would look pending forever.
  const restartHint = !environmentId && (pending || !!runtime?.restart_required);
  const refreshButton = <button type="button" className="ui-btn service-refresh" disabled={busy || disabled} onClick={()=>void refresh(3000,true).then(current=>{setError('');if(environmentId&&current.selected_environment===environmentId){setNotice(text('当前正在使用此环境。','This environment is currently active.'));onRestarted?.();}}).catch(showReadError)}><RefreshCw size={14}/>{text('刷新状态','Refresh status')}</button>;
  return <div className={`service-controls${secondary ? ' service-controls-secondary' : ''}`}>
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" className={secondary ? "ui-btn" : "ui-btn ui-btn-primary"} disabled={busy||disabled||!runtime?.can_restart} onClick={()=>void restart(environmentId?{environment_id:environmentId}:applySavedAddress?{apply_saved_address:true}:{})}>{busy?<Loader2 size={14} className="animate-spin"/>:<RefreshCw size={14}/>} {environmentId?text('重启并切换到此环境','Restart in this environment'):text('重启服务','Restart service')}</button>
      {!environmentId && runtime?.can_restore_original && <button type="button" className="ui-btn" disabled={busy||disabled||!runtime.can_restart} onClick={()=>void restart({restore_original_environment:true})}>{text('恢复原环境并重启','Restore original environment')}</button>}
      {restartHint && <span className="service-restart-hint">{text('部分设置需要重启才能生效','Some settings take effect after a restart')}</span>}
      {refreshTarget ? createPortal(refreshButton, refreshTarget) : refreshButton}
    </div>
    {runtime?.reason&&<p className="settings-note">{reasons[runtime.reason]?text(...reasons[runtime.reason]):text('当前有操作占用服务，请稍后重试。','The service is busy. Try again later.')}</p>}
    {notice&&<p role="status" className="settings-note">{notice}</p>}
    {nextAddress&&<a className="ui-link" href={nextAddress}>{text('打开新的服务地址','Open the new service address')}</a>}
    {error&&<p role="alert" className="settings-alert">{error==='Service request timed out'?text('服务暂未响应，请刷新状态。','The service did not respond. Refresh status to retry.'):error}</p>}
  </div>;
}
