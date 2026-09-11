import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import TaggingPanel from '../src/components/datasets/TaggingPanel';
import { apiClient } from '../src/api/client';
import i18n from '../src/i18n';

const model = {name:'WD SwinV2 v3',path:'D:/models/wd14',model_path:'D:/models/wd14/model.onnx',tags_path:'D:/models/wd14/selected_tags.csv'};
const status = () => ({available:true,runtime_available:true,runtime_version:'1.23',providers:['cpu'],runtime_providers:['CPUExecutionProvider'],provider:'cpu',model_exists:true,tags_exists:true,model_path:model.model_path,tags_path:model.tags_path,input_size:448,models:[model],errors:[],notes:['The worker validates actual ONNX input shape.']});
const images = [{dataset_id:'d1',rel_path:'a.png'},{dataset_id:'d1',rel_path:'b.jpg'}];
const clients: QueryClient[] = [];
function Destination() { const location=useLocation(); return <output>{location.state?.backgroundLocation?.pathname}</output>; }
function mount({onSubmit=vi.fn().mockResolvedValue(undefined),busy=false,selected=images}={}) {
  const client=new QueryClient({defaultOptions:{queries:{retry:false,gcTime:0}}});clients.push(client);
  return render(<QueryClientProvider client={client}><MemoryRouter initialEntries={['/projects/p1/v/v2?step=data&data_step=captions']}><Routes><Route path="/projects/p1/v/v2" element={<TaggingPanel projectId="p1" versionId="v2" images={selected} busy={busy} onSubmit={onSubmit}/>}/><Route path="/settings/environment" element={<Destination/>}/></Routes></MemoryRouter></QueryClientProvider>);
}
beforeEach(async()=>{await i18n.changeLanguage('zh-CN');vi.spyOn(apiClient,'get').mockResolvedValue(status());});
afterEach(()=>{vi.restoreAllMocks();clients.splice(0).forEach(client=>client.clear());});
const start = () => screen.getByRole('button',{name:/为 .* 张图片生成标签/});

describe('local WD14 tagging setup',()=>{
  it('submits real local paths and raw thresholds once, then locks changes until submission completes',async()=>{
    let finish=()=>{};
    const onSubmit=vi.fn(()=>new Promise<void>(resolve=>{finish=resolve;}));
    mount({onSubmit});
    await waitFor(()=>expect(start()).toBeEnabled());
    expect(screen.getByRole('spinbutton',{name:'常规标签阈值'})).toHaveValue(35);
    expect(screen.getByRole('spinbutton',{name:'角色标签阈值'})).toHaveValue(85);
    fireEvent.change(screen.getByRole('spinbutton',{name:'常规标签阈值'}),{target:{value:'12.5'}});
    fireEvent.change(screen.getByRole('spinbutton',{name:'角色标签阈值'}),{target:{value:'100'}});
    fireEvent.change(screen.getByRole('combobox',{name:'写入方式'}),{target:{value:'overwrite'}});
    fireEvent.change(screen.getByRole('textbox',{name:'触发词（可选）'}),{target:{value:' sora_character '}});
    expect(screen.getByText(/将替换所选图片的标签；原标签会自动备份/)).toBeInTheDocument();
    fireEvent.click(start());
    expect(onSubmit).toHaveBeenCalledExactlyOnceWith({model_path:model.model_path,tags_path:model.tags_path,provider:'cpu',mode:'overwrite',trigger_word:'sora_character',general_threshold:0.125,character_threshold:1});
    expect(start()).toBeDisabled();
    expect(screen.getByRole('combobox',{name:'写入方式'})).toBeDisabled();
    fireEvent.submit(screen.getByRole('form'));
    expect(onSubmit).toHaveBeenCalledTimes(1);
    await act(async()=>finish());
    expect(start()).toBeEnabled();
  });

  it('invalidates a previous successful check after path edits and only allows the newly checked model',async()=>{
    const onSubmit=vi.fn().mockResolvedValue(undefined);
    mount({onSubmit});await waitFor(()=>expect(start()).toBeEnabled());
    fireEvent.change(screen.getByRole('combobox',{name:'本地打标模型'}),{target:{value:''}});
    fireEvent.change(screen.getByRole('textbox',{name:'ONNX 模型文件'}),{target:{value:'D:/custom'}});
    expect(start()).toBeDisabled();
    expect(screen.getByText('模型路径或设备已更改，请重新检查后开始。')).toBeInTheDocument();
    fireEvent.submit(screen.getByRole('form'));expect(onSubmit).not.toHaveBeenCalled();
    let reply:(value:any)=>void=()=>{};
    vi.mocked(apiClient.get).mockImplementation(()=>new Promise(resolve=>{reply=resolve;}));
    fireEvent.click(screen.getByRole('button',{name:'检查模型与依赖'}));
    expect(start()).toBeDisabled();
    await act(async()=>reply({...status(),model_path:'D:/custom/model.onnx'}));
    await waitFor(()=>expect(start()).toBeEnabled());
    fireEvent.click(start());
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({model_path:'D:/custom/model.onnx',tags_path:model.tags_path}));
    fireEvent.change(screen.getByRole('textbox',{name:'标签表 CSV'}),{target:{value:''}});
    expect(screen.getByRole('textbox',{name:'标签表 CSV'})).toHaveValue('');
    expect(start()).toBeDisabled();
  });

  it('explains missing dependencies and carries the current project as the settings drawer background',async()=>{
    vi.mocked(apiClient.get).mockResolvedValue({...status(),available:false,runtime_available:false,model_exists:false,tags_exists:false,models:[],model_path:null,tags_path:null,errors:['ONNX Runtime is not installed.','model.onnx is missing.']});
    mount();
    expect(await screen.findByRole('alert')).toHaveTextContent('ONNX Runtime is not installed.');
    expect(start()).toBeDisabled();
    expect(screen.getByRole('link',{name:'下载 WD14 模型与标签表'})).toHaveAttribute('href','/settings/environment?tab=models');
    fireEvent.click(screen.getByRole('link',{name:'安装 ONNX Runtime'}));
    expect(await screen.findByText('/projects/p1/v/v2')).toBeInTheDocument();
  });

  it('keeps archived or busy versions read-only even if a submit event is forced',async()=>{
    const onSubmit=vi.fn().mockResolvedValue(undefined);mount({onSubmit,busy:true});
    await screen.findByText(/本地文件与运行库就绪/);
    expect(start()).toBeDisabled();expect(screen.getByRole('combobox',{name:'本地打标模型'})).toBeDisabled();
    fireEvent.submit(screen.getByRole('form'));expect(onSubmit).not.toHaveBeenCalled();
  });

  it('reports status failures and supports retry, while an empty image selection cannot start',async()=>{
    vi.mocked(apiClient.get).mockRejectedValueOnce(new Error('Tagger status unavailable'));
    const onSubmit=vi.fn().mockResolvedValue(undefined);mount({onSubmit,selected:[]});
    expect(await screen.findByRole('alert')).toHaveTextContent('Tagger status unavailable');
    fireEvent.click(screen.getByRole('button',{name:'检查模型与依赖'}));
    await screen.findByText(/本地文件与运行库就绪/);
    expect(start()).toBeDisabled();fireEvent.submit(screen.getByRole('form'));expect(onSubmit).not.toHaveBeenCalled();
  });
});
