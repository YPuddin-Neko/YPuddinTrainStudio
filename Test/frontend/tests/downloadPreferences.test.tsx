import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import Preferences from '../../../frontend/src/pages/Settings/Preferences';
import { apiClient } from '../../../frontend/src/api/client';
import i18n from '../../../frontend/src/i18n';
const settings = {paths:{data_root:'D:/studio_data',cache_dir:'D:/cache',models_dir:'D:/models',output_dir:'E:/outputs',output_mode:'project'},server:{host:'127.0.0.1',port:8765},ui:{language:'zh-CN',theme:'light'}};
beforeEach(async()=>{await i18n.changeLanguage('zh-CN');vi.spyOn(apiClient,'get').mockResolvedValue(structuredClone(settings));vi.spyOn(apiClient,'put').mockImplementation(async(_path,body)=>body as any);});
afterEach(()=>vi.restoreAllMocks());
it('defaults to USTC and saves selected sources independently from model downloads',async()=>{
  render(<MemoryRouter initialEntries={['/settings/preferences?section=downloads']}><Preferences/></MemoryRouter>);
  const pypi = await screen.findByRole('combobox', { name: 'Python 依赖包' });
  expect(pypi).toHaveTextContent('中国科学技术大学（默认）');
  fireEvent.click(pypi);fireEvent.click(screen.getByRole('option',{name:'清华大学'}));
  fireEvent.click(screen.getByRole('combobox',{name:'PyTorch'}));fireEvent.click(screen.getByRole('option',{name:'上海交通大学'}));
  fireEvent.click(screen.getByRole('checkbox',{name:'自动尝试其他镜像和官方源'}));
  fireEvent.click(screen.getByTestId('settings-save-btn'));
  await waitFor(()=>expect(apiClient.put).toHaveBeenCalledWith('/settings',expect.objectContaining({downloads:{pypi:'tuna',pytorch:'sjtu',fallback:false}})));
  await waitFor(()=>expect(pypi).toBeEnabled());
  expect(pypi).toHaveTextContent('清华大学');
});
it('uses the Python package source on macOS without a redundant PyTorch source selector', async()=>{
  vi.mocked(apiClient.get).mockImplementation(async endpoint => endpoint === '/system/info' ? {platform: 'macOS-15.7.9-arm64'} as any : structuredClone(settings));
  render(<MemoryRouter initialEntries={['/settings/preferences?section=downloads']}><Preferences/></MemoryRouter>);
  await screen.findByRole('combobox', {name: 'Python 依赖包'});
  await waitFor(()=>expect(screen.queryByRole('combobox',{name:'PyTorch'})).not.toBeInTheDocument());
});
