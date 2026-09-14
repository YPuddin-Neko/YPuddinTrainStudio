import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../src/api/client';
import TorchEnvironmentPanel from '../src/components/TorchEnvironmentPanel';
import ServiceControls from '../src/components/ServiceControls';
import i18n from '../src/i18n';

const build={id:'2.11.0-cu128',label:'PyTorch 2.11.0 · CU128',supported:true,reason:null,recommended:true,backend:'cu128'};
const operation=(status='ready')=>({id:'torch_test',build_id:build.id,status,phase:status==='failed'?'failed':'review',logs:[] as string[],error:null as string|null,environment_id:null as string|null,dismissed_at:null as number|null,plan:[{name:'torch',from_version:'2.10.0',version:'2.11.0'},{name:'torchvision',from_version:'0.25.0',version:'0.26.0'}]});
const snapshot=()=>({builds:[build],operations:[] as ReturnType<typeof operation>[],current_python:'project/venv/python',selected_environment:null, disk_free_bytes:100*1024**3});
const runtime=()=>({worker_id:11,managed:true,can_restart:true,reason:null as string|null,current_host:'127.0.0.1',current_port:8877,saved_host:'127.0.0.1',saved_port:8765,current_python:'project/venv/python',can_restore_original:false,selected_environment:null as string|null});
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.useRealTimers();});
beforeEach(async()=>{await i18n.changeLanguage('zh-CN');});

describe('isolated PyTorch environment controls',()=>{
  it('reviews actual old/new framework versions before preparing the environment',async()=>{
    const data=snapshot();
    const get=vi.spyOn(apiClient,'get').mockImplementation(async()=>structuredClone(data));
    const post=vi.spyOn(apiClient,'post').mockImplementation(async(url,body)=>{
      if(url.endsWith('/apply')) data.operations=[{...operation('installing'),phase:'installing_pytorch'}];
      else {expect(body).toEqual({build_id:build.id});data.operations=[operation()];}
      return structuredClone(data.operations[0]);
    });
    render(<TorchEnvironmentPanel/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'检查切换计划'})).toBeEnabled());
    fireEvent.click(screen.getByRole('button',{name:'检查切换计划'}));
    expect(await screen.findByText('2.10.0 → 2.11.0')).toBeInTheDocument();
    expect(post).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button',{name:'下载并准备此环境'}));
    expect(await screen.findByRole('progressbar',{name:'环境准备中'})).toBeInTheDocument();
    expect(post).toHaveBeenLastCalledWith('/environment/torch/operations/torch_test/apply',{}, {silent:true});
    expect(get).toHaveBeenCalledWith('/environment/torch',{silent:true});
  });
  it('keeps failed logs in explicit history after a persisted dismissal and remount',async()=>{
    const data=snapshot();data.operations=[{...operation('failed'),error:'Network unavailable',logs:['Current environment unchanged']}];
    vi.spyOn(apiClient,'get').mockImplementation(async()=>structuredClone(data));
    const post=vi.spyOn(apiClient,'post').mockImplementation(async()=>{data.operations[0].dismissed_at=123;return structuredClone(data.operations[0]);});
    const first=render(<TorchEnvironmentPanel/>);
    expect(await screen.findByRole('alert')).toHaveTextContent('Network unavailable');
    fireEvent.click(screen.getByRole('button',{name:'关闭结果'}));
    await waitFor(()=>expect(screen.queryByRole('alert')).not.toBeInTheDocument());
    expect(post).toHaveBeenCalledWith('/environment/torch/operations/torch_test/dismiss',{}, {silent:true});
    first.unmount();render(<TorchEnvironmentPanel/>);
    fireEvent.click(await screen.findByRole('button',{name:'历史记录'}));
    expect(await screen.findByRole('alert')).toHaveTextContent('Network unavailable');
    expect(screen.getByText('Current environment unchanged')).toBeInTheDocument();
    expect(data.operations[0].status).toBe('failed');
  });
  it('disables environment preparation while an existing task is running',async()=>{
    vi.spyOn(apiClient,'get').mockResolvedValue(snapshot());
    const post=vi.spyOn(apiClient,'post');
    render(<TorchEnvironmentPanel disabled/>);
    expect(await screen.findByRole('button',{name:'检查切换计划'})).toBeDisabled();
    expect(screen.getByRole('combobox',{name:'选择 PyTorch 版本'})).toBeDisabled();
    expect(post).not.toHaveBeenCalled();
  });
});

describe('service restart controls',()=>{
  it('does not send a restart when the status check resolves after unmount',async()=>{
    let reads=0;
    let release!:(value:ReturnType<typeof runtime>)=>void;
    const delayed=new Promise<ReturnType<typeof runtime>>(resolve=>{release=resolve;});
    vi.spyOn(apiClient,'get').mockImplementation(async()=>++reads===1?runtime():await delayed);
    const post=vi.spyOn(apiClient,'post');
    const onRestarted=vi.fn();
    const view=render(<ServiceControls onRestarted={onRestarted}/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启服务'})).toBeEnabled());
    fireEvent.click(screen.getByRole('button',{name:'重启服务'}));
    await waitFor(()=>expect(reads).toBe(2));
    view.unmount();
    await act(async()=>{release(runtime());await delayed;});
    expect(post).not.toHaveBeenCalled();
    expect(onRestarted).not.toHaveBeenCalled();
  });
  it('does not start reconnect probes when the restart acknowledgement arrives after unmount',async()=>{
    const get=vi.spyOn(apiClient,'get').mockResolvedValue(runtime());
    let release!:(value:{address_changed:boolean;host:string;port:number})=>void;
    const delayed=new Promise<{address_changed:boolean;host:string;port:number}>(resolve=>{release=resolve;});
    const post=vi.spyOn(apiClient,'post').mockImplementation(async()=>await delayed);
    const onRestarted=vi.fn();
    const view=render(<ServiceControls onRestarted={onRestarted}/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启服务'})).toBeEnabled());
    fireEvent.click(screen.getByRole('button',{name:'重启服务'}));
    await waitFor(()=>expect(post).toHaveBeenCalledOnce());
    view.unmount();
    await act(async()=>{release({address_changed:false,host:'127.0.0.1',port:8877});await delayed;});
    expect(get).toHaveBeenCalledTimes(2);
    expect(onRestarted).not.toHaveBeenCalled();
  });
  it('does not emit completion when a new worker response arrives after unmount',async()=>{
    let reads=0;
    let release!:(value:ReturnType<typeof runtime>)=>void;
    const delayed=new Promise<ReturnType<typeof runtime>>(resolve=>{release=resolve;});
    vi.spyOn(apiClient,'get').mockImplementation(async()=>++reads<=2?runtime():await delayed);
    vi.spyOn(apiClient,'post').mockResolvedValue({address_changed:false,host:'127.0.0.1',port:8877});
    const onRestarted=vi.fn();
    const view=render(<ServiceControls onRestarted={onRestarted}/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启服务'})).toBeEnabled());
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole('button',{name:'重启服务'}));
    await act(async()=>{await vi.advanceTimersByTimeAsync(1000);});
    expect(reads).toBe(3);
    view.unmount();
    await act(async()=>{release({...runtime(),worker_id:12});await delayed;});
    expect(onRestarted).not.toHaveBeenCalled();
    expect(reads).toBe(3);
  });
  it('shows a readable localised timeout when the initial status probe hangs',async()=>{
    vi.useFakeTimers();
    vi.spyOn(apiClient,'get').mockImplementation(async()=>await new Promise(()=>{}));
    render(<ServiceControls/>);
    await act(async()=>{await vi.advanceTimersByTimeAsync(3000);});
    expect(screen.getByRole('alert')).toHaveTextContent('服务暂未响应，请重新检测。');
    expect(screen.queryByText('Service request timed out')).not.toBeInTheDocument();
  });
  it('bounds a restart acknowledgement that never returns and releases the controls',async()=>{
    vi.spyOn(apiClient,'get').mockResolvedValue(runtime());
    let pending:AbortSignal|undefined;
    vi.spyOn(apiClient,'post').mockImplementation(async(_endpoint,_body,options)=>{
      pending=options?.signal as AbortSignal;
      return await new Promise(()=>{});
    });
    render(<ServiceControls/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启服务'})).toBeEnabled());
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole('button',{name:'重启服务'}));
    await act(async()=>{await vi.advanceTimersByTimeAsync(10000);});
    expect(pending?.aborted).toBe(true);
    expect(screen.getByRole('alert')).toHaveTextContent('服务暂未响应');
    expect(screen.getByRole('button',{name:'重新检测'})).toBeEnabled();
  });
  it('shows the actual busy reason and does not send a restart',async()=>{
    vi.spyOn(apiClient,'get').mockResolvedValue({...runtime(),can_restart:false,reason:'training_or_data_worker_running'});
    const post=vi.spyOn(apiClient,'post');
    render(<ServiceControls/>);
    expect(await screen.findByText('训练或数据任务运行中，请完成或停止后重启。')).toBeInTheDocument();
    expect(screen.getByRole('button',{name:'重启服务'})).toBeDisabled();
    expect(post).not.toHaveBeenCalled();
  });
  it('keeps current address by default and waits for a new worker before reporting success',async()=>{
    let restarting=false,probes=0;
    vi.spyOn(apiClient,'get').mockImplementation(async()=>({...runtime(),worker_id:restarting&&++probes>1?12:11}));
    const post=vi.spyOn(apiClient,'post').mockImplementation(async()=>{restarting=true;return {address_changed:false,port:8877,host:'127.0.0.1',reconnect_url:'http://127.0.0.1:8877/'};});
    const onRestarted=vi.fn();render(<ServiceControls onRestarted={onRestarted}/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启服务'})).toBeEnabled());
    fireEvent.click(screen.getByRole('button',{name:'重启服务'}));
    expect(await screen.findByText('服务已重启。', {}, { timeout: 4000 })).toBeInTheDocument();
    expect(probes).toBeGreaterThanOrEqual(2);
    expect(post).toHaveBeenCalledWith('/service/restart',{}, {silent:true,signal:expect.any(AbortSignal)});
    expect(onRestarted).toHaveBeenCalledOnce();
  });
  it('does not report a failed environment switch as successful after launcher fallback',async()=>{
    let restarting=false;
    vi.spyOn(apiClient,'get').mockImplementation(async()=>({...runtime(),worker_id:restarting?12:11}));
    vi.spyOn(apiClient,'post').mockImplementation(async()=>{restarting=true;return {address_changed:false,port:8877,host:'127.0.0.1'};});
    render(<ServiceControls environmentId="torch_new"/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启并切换到此环境'})).toBeEnabled());
    fireEvent.click(screen.getByRole('button',{name:'重启并切换到此环境'}));
    expect(await screen.findByRole('alert', {}, { timeout: 4000 })).toHaveTextContent('新环境启动失败，服务已恢复原环境');
    expect(screen.queryByText('服务已重启。')).not.toBeInTheDocument();
  });
  it('changes the address only from the explicit saved-address action',async()=>{
    vi.spyOn(apiClient,'get').mockResolvedValue({...runtime(),saved_host:'10.10.10.16',saved_port:9000});
    const post=vi.spyOn(apiClient,'post').mockResolvedValue({address_changed:true,port:9000,host:'10.10.10.16',reconnect_url:'http://10.10.10.16:9000/'});
    render(<ServiceControls/>);
    fireEvent.click(await screen.findByRole('button',{name:'应用已保存地址并重启（10.10.10.16:9000）'}));
    expect(await screen.findByRole('link',{name:'打开新的服务地址'})).toHaveAttribute('href','http://10.10.10.16:9000/');
    expect(post).toHaveBeenCalledWith('/service/restart',{apply_saved_address:true},{silent:true,signal:expect.any(AbortSignal)});
  });
  it('aborts a hanging reconnect probe after three seconds and can then reconnect',async()=>{
    let restarting=false,probes=0;
    let stalled:AbortSignal|undefined;
    vi.spyOn(apiClient,'get').mockImplementation(async(_endpoint,options)=>{
      if(restarting&&++probes===1){stalled=options?.signal as AbortSignal;return await new Promise(()=>{});}
      return {...runtime(),worker_id:restarting?12:11};
    });
    vi.spyOn(apiClient,'post').mockImplementation(async()=>{restarting=true;return {address_changed:false,port:8877,host:'127.0.0.1'};});
    render(<ServiceControls/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启服务'})).toBeEnabled());
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole('button',{name:'重启服务'}));
    await act(async()=>{await vi.advanceTimersByTimeAsync(1000);});
    expect(stalled?.aborted).toBe(false);
    await act(async()=>{await vi.advanceTimersByTimeAsync(3000);});
    expect(stalled?.aborted).toBe(true);
    await act(async()=>{await vi.advanceTimersByTimeAsync(1000);});
    expect(screen.getByText('服务已重启。')).toBeInTheDocument();
  });
  it('ends reconnect attempts within sixty seconds even when requests never settle',async()=>{
    let restarting=false;
    const signals:AbortSignal[]=[];
    vi.spyOn(apiClient,'get').mockImplementation(async(_endpoint,options)=>{
      if(restarting){signals.push(options?.signal as AbortSignal);return await new Promise(()=>{});}
      return runtime();
    });
    vi.spyOn(apiClient,'post').mockImplementation(async()=>{restarting=true;return {address_changed:false,port:8877,host:'127.0.0.1'};});
    render(<ServiceControls/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启服务'})).toBeEnabled());
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole('button',{name:'重启服务'}));
    await act(async()=>{await vi.advanceTimersByTimeAsync(60000);});
    expect(signals.length).toBeGreaterThan(1);
    expect(signals.every(signal=>signal.aborted)).toBe(true);
    expect(screen.getByRole('alert')).toHaveTextContent('暂未重新连接');
    expect(screen.queryByText('正在重启并重新连接…')).not.toBeInTheDocument();
    expect(screen.getByRole('button',{name:'重新检测'})).toBeEnabled();
  });
});
