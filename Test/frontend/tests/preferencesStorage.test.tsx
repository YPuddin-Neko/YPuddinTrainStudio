import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Preferences from '../../../frontend/src/pages/Settings/Preferences';
import { apiClient } from '../../../frontend/src/api/client';
import i18n from '../../../frontend/src/i18n';

const settings = {paths:{data_root:'D:/studio_data',cache_dir:'D:/cache',models_dir:'D:/models',output_dir:'E:/outputs',output_mode:'project'},server:{ open_browser: true,host:'127.0.0.1',port:8765},ui:{language:'zh-CN',theme:'light'}};
beforeEach(async()=>{await i18n.changeLanguage('zh-CN');vi.spyOn(apiClient,'get').mockResolvedValue(structuredClone(settings));vi.spyOn(apiClient,'put').mockImplementation(async(_path,body)=>body as any);});
afterEach(()=>vi.restoreAllMocks());
function show(){render(<MemoryRouter initialEntries={['/settings/preferences?section=storage']}><Preferences/></MemoryRouter>);}
describe('project output settings',()=>{
  it('defaults to per-project output while retaining the custom path when switching modes',async()=>{
    show();
    const mode=await screen.findByRole('combobox',{name:'训练产物位置'});
    expect(mode).toHaveTextContent('项目版本目录（默认）');
    expect(screen.queryByRole('textbox',{name:i18n.t('settings.outputDir')})).not.toBeInTheDocument();
    fireEvent.click(mode);fireEvent.click(screen.getByRole('option',{name:'自定义输出根目录'}));
    expect(screen.getByRole('textbox',{name:i18n.t('settings.outputDir')})).toHaveValue('E:/outputs');
    fireEvent.change(screen.getByRole('textbox',{name:i18n.t('settings.outputDir')}),{target:{value:'F:/training'}});
    fireEvent.click(screen.getByTestId('settings-save-btn'));
    await waitFor(()=>expect(apiClient.put).toHaveBeenCalledWith('/settings',expect.objectContaining({paths:expect.objectContaining({output_mode:'custom',output_dir:'F:/training'})})));
    await waitFor(()=>expect(mode).toBeEnabled());
    fireEvent.click(mode);fireEvent.click(screen.getByRole('option',{name:'项目版本目录（默认）'}));
    fireEvent.click(screen.getByTestId('settings-save-btn'));
    await waitFor(()=>expect(apiClient.put).toHaveBeenLastCalledWith('/settings',expect.objectContaining({paths:expect.objectContaining({output_mode:'project',output_dir:'F:/training'})})));
  });
  it('locks selects and path controls during a delayed save, preserving the submitted path draft',async()=>{
    let finish=()=>{};
    const pending=new Promise<void>(resolve=>{finish=resolve;});
    vi.mocked(apiClient.put).mockImplementation(async(_path,body)=>{await pending;return body as any;});
    show();const mode=await screen.findByRole('combobox',{name:'训练产物位置'});
    fireEvent.click(mode);fireEvent.click(screen.getByRole('option',{name:'自定义输出根目录'}));
    const path=screen.getByRole('textbox',{name:i18n.t('settings.outputDir')});
    fireEvent.change(path,{target:{value:'F:/submitted'}});
    fireEvent.click(screen.getByTestId('settings-save-btn'));
    expect(mode).toBeDisabled();expect(path).toBeDisabled();
    screen.getAllByRole('button',{name:i18n.t('common.browse')}).forEach(button=>expect(button).toBeDisabled());
    expect(screen.getByRole('textbox',{name:i18n.t('settings.modelsDir')})).toBeDisabled();
    fireEvent.change(path,{target:{value:'G:/must_not_replace_pending_draft'}});
    act(()=>window.dispatchEvent(new CustomEvent('studio.settings.changed',{detail:{...settings,paths:{...settings.paths,output_dir:'E:/stale'},ui:{theme:'dark',language:'zh-CN'}}})));
    expect(path).toHaveValue('F:/submitted');
    expect(apiClient.put).toHaveBeenCalledOnce();
    await act(async()=>finish());
    await waitFor(()=>expect(mode).toBeEnabled());
    expect(path).toHaveValue('F:/submitted');
    fireEvent.change(path,{target:{value:'G:/next_draft'}});
    expect(path).toHaveValue('G:/next_draft');
  });
  it('unlocks after a failed save and retains the custom output mode and typed path for retry',async()=>{
    vi.mocked(apiClient.put).mockRejectedValueOnce(new Error('settings disk full'));
    show();const mode=await screen.findByRole('combobox',{name:'训练产物位置'});
    fireEvent.click(mode);fireEvent.click(screen.getByRole('option',{name:'自定义输出根目录'}));
    const path=screen.getByRole('textbox',{name:i18n.t('settings.outputDir')});
    fireEvent.change(path,{target:{value:'F:/retry_path'}});
    fireEvent.click(screen.getByTestId('settings-save-btn'));
    expect(await screen.findByRole('alert')).toHaveTextContent('settings disk full');
    expect(mode).toBeEnabled();expect(path).toBeEnabled();
    expect(mode).toHaveTextContent('自定义输出根目录');expect(path).toHaveValue('F:/retry_path');
    expect(screen.getByTestId('settings-save-btn')).not.toHaveTextContent(i18n.t('settings.saved'));
    fireEvent.click(screen.getByTestId('settings-save-btn'));
    await waitFor(()=>expect(apiClient.put).toHaveBeenCalledTimes(2));
    expect(apiClient.put).toHaveBeenLastCalledWith('/settings',expect.objectContaining({paths:expect.objectContaining({output_mode:'custom',output_dir:'F:/retry_path'})}));
  });
  it('saving a path edit preserves a theme changed through the persistent sidebar',async()=>{
    show();await screen.findByTestId('settings-page');
    fireEvent.change(screen.getByRole('textbox',{name:i18n.t('settings.modelsDir')}),{target:{value:'F:/weights'}});
    act(()=>window.dispatchEvent(new CustomEvent('studio.settings.changed',{detail:{...settings,ui:{theme:'dark',language:'zh-CN'}}})));
    fireEvent.click(screen.getByTestId('settings-save-btn'));
    await waitFor(()=>expect(apiClient.put).toHaveBeenCalledWith('/settings',expect.objectContaining({ui:{theme:'dark',language:'zh-CN'},paths:expect.objectContaining({models_dir:'F:/weights'})})));
  });
});

it('explains directory purposes and only shows restart guidance after a relevant edit', async () => {
  show();
  const root = await screen.findByRole('textbox', {name:i18n.t('settings.dataRoot')});
  expect(screen.getByText('保存项目、数据集、任务记录和服务设置。')).toBeVisible();
  expect(screen.getByText(/保存下载的模型及其组件/)).toBeVisible();
  expect(screen.queryByText(/重启服务.*生效/)).not.toBeInTheDocument();
  expect(screen.queryByText(/下次从启动脚本/)).not.toBeInTheDocument();
  fireEvent.change(root, {target:{value:'E:/new-studio'}});
  expect(screen.getByText('保存后需重启服务生效。')).toBeVisible();
  fireEvent.click(screen.getByTestId('settings-save-btn'));
  expect(await screen.findByText('已保存，重启服务后生效。')).toBeVisible();
  fireEvent.change(screen.getByRole('textbox', {name:i18n.t('settings.modelsDir')}), {target:{value:'F:/new-models'}});
  fireEvent.click(screen.getByTestId('settings-save-btn'));
  await waitFor(() => expect(apiClient.put).toHaveBeenCalledTimes(2));
  expect(screen.getByText('已保存，重启服务后生效。')).toBeVisible();
});

it('distinguishes an environment-directory launcher restart from a service restart', async () => {
  show();
  const env = await screen.findByRole('textbox', {name:'基础环境目录'});
  fireEvent.change(env, {target:{value:'E:/python-envs'}});
  expect(screen.getByText('保存后，下次从启动脚本启动时生效。')).toBeVisible();
  expect(screen.getByText('新目录需安装依赖，旧环境保留。')).toBeVisible();
  fireEvent.click(screen.getByTestId('settings-save-btn'));
  expect(await screen.findByText('已保存，下次从启动脚本启动时生效。')).toBeVisible();
});
