import { StrictMode } from 'react';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../../../frontend/src/api/client';
import { ApiError } from '../../../frontend/src/api/types';
import TorchEnvironmentPanel from '../../../frontend/src/components/TorchEnvironmentPanel';
import ServiceControls from '../../../frontend/src/components/ServiceControls';
import i18n from '../../../frontend/src/i18n';

const build={id:'2.11.0-cu128',label:'PyTorch 2.11.0 · CU128',supported:true,reason:null,recommended:true,backend:'cu128'};
const operation=(status='ready')=>({id:'torch_test',build_id:build.id,status,phase:status==='failed'?'failed':'review',logs:[] as string[],error:null as string|null,environment_id:null as string|null,dismissed_at:null as number|null,plan:[{name:'torch',from_version:'2.10.0',version:'2.11.0'},{name:'torchvision',from_version:'0.25.0',version:'0.26.0'}]});
const snapshot=()=>({builds:[build],operations:[] as ReturnType<typeof operation>[],current_python:'project/venv/python',selected_environment:null as string|null, disk_free_bytes:100*1024**3});
const runtime=()=>({worker_id:11,managed:true,can_restart:true,reason:null as string|null,current_host:'127.0.0.1',current_port:8877,saved_host:'127.0.0.1',saved_port:8765,current_python:'project/venv/python',can_restore_original:false,selected_environment:null as string|null});
afterEach(()=>{cleanup();vi.restoreAllMocks();vi.useRealTimers();});
beforeEach(async()=>{await i18n.changeLanguage('zh-CN');});

describe('isolated PyTorch environment controls',()=>{
  it('does not show a stale abort error after StrictMode remounts the initial service probe',async()=>{
    let calls=0;
    vi.spyOn(apiClient,'get').mockImplementation(async(_url,options)=>{
      calls+=1;
      if(calls===1)return await new Promise((_resolve,reject)=>options?.signal?.addEventListener('abort',()=>reject(new DOMException('signal is aborted without reason','AbortError'))));
      return runtime();
    });
    render(<StrictMode><ServiceControls/></StrictMode>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启服务'})).toBeEnabled());
    expect(calls).toBe(2);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('reviews actual old/new framework versions before preparing the environment',async()=>{
    const data=snapshot();
    const get=vi.spyOn(apiClient,'get').mockImplementation(async()=>structuredClone(data));
    const post=vi.spyOn(apiClient,'post').mockImplementation(async(url,body)=>{
      if(url.endsWith('/apply')) data.operations=[{...operation('installing'),phase:'installing_pytorch'}];
      else {expect(body).toEqual({build_id:build.id});data.operations=[operation()];}
      return structuredClone(data.operations[0]);
    });
    render(<TorchEnvironmentPanel/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'检查 PyTorch 安装条件'})).toBeEnabled());
    fireEvent.click(screen.getByRole('button',{name:'检查 PyTorch 安装条件'}));
    expect(await screen.findByText('2.10.0 → 2.11.0')).toBeInTheDocument();
    expect(post).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button',{name:'安装到独立环境'}));
    expect(await screen.findByRole('progressbar',{name:'PyTorch 安装进度'})).toBeInTheDocument();
    expect(post).toHaveBeenLastCalledWith('/environment/torch/operations/torch_test/apply',{}, {silent:true});
    expect(get).toHaveBeenCalledWith('/environment/torch',{silent:true});
  });
  it('keeps a current failure visible but does not restore old logs or history after reopening', async () => {
    const data=snapshot();data.operations=[operation('ready')];
    vi.spyOn(apiClient,'get').mockImplementation(async()=>structuredClone(data));
    const post=vi.spyOn(apiClient,'post');
    const first=render(<TorchEnvironmentPanel/>);
    await screen.findByRole('button',{name:'安装到独立环境'});
    data.operations=[{...operation('failed'),error:'Network unavailable',logs:['Current environment unchanged']}];
    fireEvent.click(screen.getByRole('button',{name:'刷新状态'}));
    expect(await screen.findByRole('alert')).toHaveTextContent('Network unavailable');
    expect(screen.getByLabelText('PyTorch 安装日志')).toHaveClass('bg-slate-950');
    expect(screen.queryByRole('button',{name:'关闭结果'})).not.toBeInTheDocument();
    first.unmount();render(<TorchEnvironmentPanel/>);
    await screen.findByRole('button',{name:'检查 PyTorch 安装条件'});
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByText('Current environment unchanged')).not.toBeInTheDocument();
    expect(screen.queryByRole('button',{name:'历史记录'})).not.toBeInTheDocument();
    expect(post).not.toHaveBeenCalled();
  });
  it('places progress and cancellation in the shared installation log target', async () => {
    const data=snapshot();data.operations=[{...operation('installing'),phase:'installing_pytorch',logs:['Installing the selected PyTorch build']}];
    vi.spyOn(apiClient,'get').mockImplementation(async()=>structuredClone(data));
    const post=vi.spyOn(apiClient,'post').mockImplementation(async()=>{data.operations=[{...operation('cancelled'),phase:'cancelled'}];return data.operations[0];});
    const target=document.createElement('div');document.body.appendChild(target);
    const view=render(<TorchEnvironmentPanel operationsTarget={target}/>);
    const progress=await screen.findByRole('progressbar',{name:'PyTorch 安装进度'});
    expect(target).toContainElement(progress);
    expect(progress.tagName).toBe('DIV');
    expect(progress).not.toHaveAttribute('aria-valuenow');
    expect(view.container).not.toContainElement(progress);
    const cancel=screen.getByRole('button',{name:'取消安装'});
    expect(target).toContainElement(cancel);
    fireEvent.click(cancel);
    await waitFor(()=>expect(post).toHaveBeenCalledWith('/environment/torch/operations/torch_test/cancel',{}, {silent:true}));
    expect(await screen.findByText('已取消')).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    view.unmount();target.remove();
  });
  it('offers an installed environment without reopening its historical installation log', async () => {
    const data=snapshot();data.operations=[{...operation('completed'),phase:'ready_to_restart',environment_id:'new-env',logs:['Old completed installation']}];
    vi.spyOn(apiClient,'get').mockImplementation(async(url)=>url==='/service/runtime'?runtime():structuredClone(data));
    render(<TorchEnvironmentPanel/>);
    expect(await screen.findByText('此版本已安装，可直接切换，无需重新下载。')).toBeInTheDocument();
    expect(await screen.findByRole('button',{name:'重启并切换到此环境'})).toBeInTheDocument();
    expect(screen.queryByText('Old completed installation')).not.toBeInTheDocument();
    expect(screen.queryByRole('heading',{name:'安装日志'})).not.toBeInTheDocument();
    expect(screen.queryByRole('button',{name:'关闭结果'})).not.toBeInTheDocument();
  });
  it('opens on the active build and omits completed logs across repeated visits', async () => {
    const currentBuild={...build,id:'2.13.0-cu130',label:'PyTorch 2.13.0 · CU130',recommended:false};
    const data=snapshot();data.builds.push(currentBuild);data.selected_environment='active-env';
    data.operations=[{...operation('completed'),id:'older',environment_id:'old-env',logs:['Old environment log']},{...operation('completed'),id:'current',build_id:currentBuild.id,environment_id:'active-env',logs:['Active environment log']}];
    const get=vi.spyOn(apiClient,'get').mockResolvedValue(data);
    const visible=vi.fn();
    const first=render(<TorchEnvironmentPanel onOperationsVisible={visible}/>);
    expect(await screen.findByRole('combobox',{name:'选择 PyTorch 版本'})).toHaveTextContent('PyTorch 2.13.0 · CU130 · 当前使用');
    expect(screen.queryByTestId('installed-torch-environment')).not.toBeInTheDocument();
    first.unmount();render(<TorchEnvironmentPanel onOperationsVisible={visible}/>);
    expect(await screen.findByRole('combobox',{name:'选择 PyTorch 版本'})).toHaveTextContent('PyTorch 2.13.0 · CU130 · 当前使用');
    expect(screen.queryByRole('heading',{name:'安装日志'})).not.toBeInTheDocument();
    expect(screen.queryByLabelText('PyTorch 安装日志')).not.toBeInTheDocument();
    expect(screen.queryByRole('button',{name:'重启并切换到此环境'})).not.toBeInTheDocument();
    expect(visible).not.toHaveBeenCalledWith(true);
    expect(get).not.toHaveBeenCalledWith('/service/runtime',expect.anything());
  });
  it('labels an activated environment correctly even when its completed phase remains ready_to_restart', async () => {
    const data=snapshot();data.operations=[operation('ready')];
    let selectedEnvironment:string|null=null;
    vi.spyOn(apiClient,'get').mockImplementation(async(url)=>url==='/service/runtime'?runtime():structuredClone({...data,selected_environment:selectedEnvironment}));
    vi.spyOn(apiClient,'post').mockImplementation(async()=>{
      data.operations=[{...operation('completed'),phase:'ready_to_restart',environment_id:'new-env'}];
      return structuredClone(data.operations[0]);
    });
    render(<TorchEnvironmentPanel/>);
    fireEvent.click(await screen.findByRole('button',{name:'安装到独立环境'}));
    expect(await screen.findByText('安装完成，重启后可使用')).toBeInTheDocument();
    expect(screen.getByRole('button',{name:'重启并切换到此环境'})).toBeInTheDocument();
    selectedEnvironment='new-env';
    const section=screen.getByRole('heading',{name:'PyTorch 版本'}).closest('section')!;
    fireEvent.click(within(section).getByRole('button',{name:'刷新状态'}));
    expect(await screen.findByText('已启用')).toBeInTheDocument();
    expect(screen.getByRole('combobox',{name:'选择 PyTorch 版本'})).toHaveTextContent('当前使用');
    expect(screen.queryByText('安装完成，重启后可使用')).not.toBeInTheDocument();
    expect(screen.queryByRole('button',{name:'重启并切换到此环境'})).not.toBeInTheDocument();
  });
  it('disables environment preparation while an existing task is running',async()=>{
    vi.spyOn(apiClient,'get').mockResolvedValue(snapshot());
    const post=vi.spyOn(apiClient,'post');
    render(<TorchEnvironmentPanel disabled/>);
    expect(await screen.findByRole('button',{name:'检查 PyTorch 安装条件'})).toBeDisabled();
    expect(screen.getByRole('combobox',{name:'选择 PyTorch 版本'})).toBeDisabled();
    expect(post).not.toHaveBeenCalled();
  });
  it.each(['mps','cpu'])('keeps the %s environment guidance free of CUDA extension and storage requirements',async backend=>{
    const data=snapshot();
    data.builds=[{...build,id:`2.11.0-${backend}`,label:`PyTorch 2.11.0 · ${backend.toUpperCase()}`,backend}];
    vi.spyOn(apiClient,'get').mockResolvedValue(data);
    render(<TorchEnvironmentPanel showAttentionExtensions={false}/>);
    const space=await screen.findByText(/可用空间/);
    expect(space).toHaveTextContent('8 GiB');
    expect(space).not.toHaveTextContent('12 GiB');
    const section=screen.getByRole('heading',{name:'PyTorch 版本'}).closest('section')!;
    expect(section).not.toHaveTextContent(/xFormers|FlashAttention|CUDA/);
    expect(screen.getByRole('button',{name:'检查 PyTorch 安装条件'})).toBeEnabled();
  });
  it('keeps CUDA extension guidance and updates storage requirements when the selected build changes',async()=>{
    const data=snapshot();
    data.builds.push({...build,id:'2.11.0-cpu',label:'PyTorch 2.11.0 · CPU',backend:'cpu',recommended:false});
    vi.spyOn(apiClient,'get').mockResolvedValue(data);
    render(<TorchEnvironmentPanel/>);
    expect(await screen.findByText(/可用空间/)).toHaveTextContent('12 GiB');
    const section=screen.getByRole('heading',{name:'PyTorch 版本'}).closest('section')!;
    expect(section).toHaveTextContent('xFormers');
    expect(section).toHaveTextContent('FlashAttention');
    fireEvent.click(screen.getByRole('combobox',{name:'选择 PyTorch 版本'}));
    fireEvent.click(screen.getByRole('option',{name:'PyTorch 2.11.0 · CPU'}));
    expect(screen.getByText(/可用空间/)).toHaveTextContent('8 GiB');
    expect(screen.getByText(/可用空间/)).not.toHaveTextContent('12 GiB');
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
    expect(screen.getByRole('alert')).toHaveTextContent('服务暂未响应，请刷新状态。');
    expect(screen.queryByText('Service request timed out')).not.toBeInTheDocument();
  });
  it('keeps checking after a lost restart acknowledgement and releases controls at the deadline',async()=>{
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
    expect(screen.getByText('正在重启并重新连接…')).toBeInTheDocument();
    expect(screen.getByRole('button',{name:'刷新状态'})).toBeDisabled();
    await act(async()=>{await vi.advanceTimersByTimeAsync(50000);});
    expect(screen.getByRole('alert')).toHaveTextContent('暂未重新连接');
    expect(screen.getByRole('button',{name:'刷新状态'})).toBeEnabled();
  });
  it('recognises a new service instance even if Windows reuses the process id',async()=>{
    let restarting=false;
    vi.spyOn(apiClient,'get').mockImplementation(async()=>({...runtime(),instance_id:restarting?'after':'before',selected_environment:restarting?'new-env':null}));
    vi.spyOn(apiClient,'post').mockImplementation(async()=>{restarting=true;return {address_changed:false,host:'127.0.0.1',port:8877};});
    const complete=vi.fn();render(<ServiceControls environmentId="new-env" onRestarted={complete}/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启并切换到此环境'})).toBeEnabled());
    vi.useFakeTimers();fireEvent.click(screen.getByRole('button',{name:'重启并切换到此环境'}));
    await act(async()=>{await vi.advanceTimersByTimeAsync(1000);});
    expect(screen.getByText('服务已重启。')).toBeInTheDocument();
    expect(complete).toHaveBeenCalledOnce();
  });
  it('does not treat an unchanged instance as a completed restart',async()=>{
    let restarting=false;
    vi.spyOn(apiClient,'get').mockImplementation(async()=>({...runtime(),instance_id:'same-instance',worker_id:restarting?12:11}));
    vi.spyOn(apiClient,'post').mockImplementation(async()=>{restarting=true;return {address_changed:false,host:'127.0.0.1',port:8877};});
    render(<ServiceControls/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启服务'})).toBeEnabled());
    vi.useFakeTimers();fireEvent.click(screen.getByRole('button',{name:'重启服务'}));
    await act(async()=>{await vi.advanceTimersByTimeAsync(1000);});
    expect(screen.getByText('正在重启并重新连接…')).toBeInTheDocument();
    expect(screen.queryByText('服务已重启。')).not.toBeInTheDocument();
  });
  it('recovers a restart whose acknowledgement was lost without posting twice',async()=>{
    let restarting=false;
    vi.spyOn(apiClient,'get').mockImplementation(async()=>({...runtime(),instance_id:restarting?'after':'before'}));
    const post=vi.spyOn(apiClient,'post').mockImplementation(async()=>{restarting=true;throw new TypeError('Failed to fetch');});
    render(<ServiceControls/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启服务'})).toBeEnabled());
    vi.useFakeTimers();fireEvent.click(screen.getByRole('button',{name:'重启服务'}));
    await act(async()=>{await vi.advanceTimersByTimeAsync(1000);});
    expect(screen.getByText('服务已重启。')).toBeInTheDocument();
    expect(post).toHaveBeenCalledOnce();
  });
  it('shows a rejected restart immediately instead of polling it as a lost acknowledgement',async()=>{
    vi.spyOn(apiClient,'get').mockResolvedValue(runtime());
    vi.spyOn(apiClient,'post').mockRejectedValue(new ApiError(409,{code:'service_busy',message:'训练任务运行中'}));
    render(<ServiceControls/>);
    fireEvent.click(await screen.findByRole('button',{name:'重启服务'}));
    expect(await screen.findByRole('alert')).toHaveTextContent('训练任务运行中');
    expect(screen.queryByText('正在重启并重新连接…')).not.toBeInTheDocument();
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
    const post=vi.spyOn(apiClient,'post').mockImplementation(async()=>{restarting=true;return {address_changed:false,port:8877,host:'127.0.0.1'};});
    render(<ServiceControls environmentId="torch_new" applySavedAddress/>);
    await waitFor(()=>expect(screen.getByRole('button',{name:'重启并切换到此环境'})).toBeEnabled());
    fireEvent.click(screen.getByRole('button',{name:'重启并切换到此环境'}));
    expect(await screen.findByRole('alert', {}, { timeout: 4000 })).toHaveTextContent('新环境启动失败，服务已恢复原环境');
    expect(post).toHaveBeenCalledWith('/service/restart',{environment_id:'torch_new'},{silent:true,signal:expect.any(AbortSignal)});
    expect(screen.queryByText('服务已重启。')).not.toBeInTheDocument();
  });
  it('restores the original environment without also applying a saved address',async()=>{
    vi.spyOn(apiClient,'get').mockResolvedValue({...runtime(),can_restore_original:true});
    const post=vi.spyOn(apiClient,'post').mockResolvedValue({address_changed:false,port:8877,host:'127.0.0.1'});
    render(<ServiceControls applySavedAddress/>);
    fireEvent.click(await screen.findByRole('button',{name:'恢复原环境并重启'}));
    await waitFor(()=>expect(post).toHaveBeenCalledWith('/service/restart',{restore_original_environment:true},{silent:true,signal:expect.any(AbortSignal)}));
  });
  it.each([
    ['10.10.10.16','10.10.10.16'],
    ['127.0.0.1','127.0.0.1'],
    ['0.0.0.0',null],
    ['::',null],
  ] as const)('applies saved address %s:9000 through the single restart action and reconnects to a usable host',async(savedHost,reconnectHost)=>{
    vi.spyOn(apiClient,'get').mockResolvedValue({...runtime(),saved_host:savedHost,saved_port:9000});
    const post=vi.spyOn(apiClient,'post').mockResolvedValue({address_changed:true,port:9000,host:savedHost,reconnect_url:'http://127.0.0.1:9000/'});
    render(<ServiceControls applySavedAddress/>);
    const restart=screen.getByRole('button',{name:'重启服务'});
    await waitFor(()=>expect(restart).toBeEnabled());
    expect(screen.getAllByRole('button',{name:/重启/})).toEqual([restart]);
    expect(post).not.toHaveBeenCalled();
    fireEvent.click(restart);
    const link=await screen.findByRole('link',{name:'打开新的服务地址'});
    const address=new URL(link.getAttribute('href')!);
    expect(address.hostname).toBe(reconnectHost??window.location.hostname);
    expect(address.port).toBe('9000');
    expect(post).toHaveBeenCalledOnce();
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
    expect(screen.getByRole('button',{name:'刷新状态'})).toBeEnabled();
  });
});
