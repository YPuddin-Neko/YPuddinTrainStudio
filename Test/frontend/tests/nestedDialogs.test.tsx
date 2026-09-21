import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import Dialog from '../../../frontend/src/components/Dialog';
import SettingsDrawer from '../../../frontend/src/components/SettingsDrawer';
import { PathInput } from '../../../frontend/src/components/PathBrowser';
import { apiClient } from '../../../frontend/src/api/client';
import '../../../frontend/src/i18n';

afterEach(() => vi.restoreAllMocks());

function NestedModelPicker({ onModelClose, onSettingsClose }: { onModelClose: () => void; onSettingsClose: () => void }) {
  const [settingsOpen, setSettingsOpen] = React.useState(true);
  const [modelOpen, setModelOpen] = React.useState(false);
  const [path, setPath] = React.useState('/qa/models');
  return settingsOpen ? <SettingsDrawer onClose={() => { onSettingsClose(); setSettingsOpen(false); }}>
    <button onClick={() => setModelOpen(true)}>添加本地模型</button>
    {modelOpen && <Dialog title="添加本地模型" onClose={() => { onModelClose(); setModelOpen(false); }}>
      <PathInput value={path} onChange={setPath} ariaLabel="文件路径"/>
      <button type="button">模型末尾按钮</button>
    </Dialog>}
  </SettingsDrawer> : <p>已回到工作区</p>;
}

async function openPicker() {
  vi.spyOn(apiClient, 'get').mockResolvedValue({ path: '/qa/models', parent: '/qa', entries: [] });
  const onModelClose = vi.fn(), onSettingsClose = vi.fn();
  render(<NestedModelPicker onModelClose={onModelClose} onSettingsClose={onSettingsClose}/>);
  const modelButton = screen.getByRole('button', { name: '添加本地模型' });
  modelButton.focus(); fireEvent.click(modelButton);
  const model = screen.getByRole('dialog', { name: '添加本地模型' });
  const browse = within(model).getByRole('button', { name: '浏览' });
  browse.focus(); fireEvent.click(browse);
  const picker = screen.getByRole('dialog', { name: '浏览训练机路径' });
  await within(picker).findByText('这个目录中没有文件。');
  return { modelButton, model, browse, picker, onModelClose, onSettingsClose };
}

describe('nested model path picker keyboard ownership', () => {
  it.each([true, false])('Escape closes exactly one layer and restores its trigger (reduced motion: %s)', async reducedMotion => {
    vi.spyOn(window, 'matchMedia').mockReturnValue({ matches: reducedMotion } as MediaQueryList);
    const { modelButton, model, browse, picker, onModelClose, onSettingsClose } = await openPicker();
    const address = within(picker).getByRole('textbox', { name: '训练机上的文件夹地址' });
    expect(address).toHaveFocus();

    fireEvent.keyDown(address, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('dialog', { name: '浏览训练机路径' })).not.toBeInTheDocument());
    expect(model).toBeInTheDocument();
    expect(model.parentElement).not.toHaveClass('is-closing');
    expect(screen.getByRole('dialog', { name: '系统设置' })).toBeInTheDocument();
    expect(browse).toHaveFocus();
    expect(onModelClose).not.toHaveBeenCalled();
    expect(onSettingsClose).not.toHaveBeenCalled();

    fireEvent.keyDown(browse, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('dialog', { name: '添加本地模型' })).not.toBeInTheDocument());
    expect(screen.getByRole('dialog', { name: '系统设置' })).toBeInTheDocument();
    expect(modelButton).toHaveFocus();
    expect(onModelClose).toHaveBeenCalledOnce();
    expect(onSettingsClose).not.toHaveBeenCalled();

    fireEvent.keyDown(modelButton, { key: 'Escape' });
    await screen.findByText('已回到工作区');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(onSettingsClose).toHaveBeenCalledOnce();
  });

  it('keeps Tab wrapping within the nested picker instead of reaching model controls', async () => {
    const { picker, onModelClose, onSettingsClose } = await openPicker();
    const first = within(picker).getByRole('button', { name: '关闭' });
    const last = within(picker).getByRole('button', { name: '选择当前目录' });
    first.focus(); fireEvent.keyDown(first, { key: 'Tab', shiftKey: true });
    expect(last).toHaveFocus();
    fireEvent.keyDown(last, { key: 'Tab' });
    expect(first).toHaveFocus();
    expect(onModelClose).not.toHaveBeenCalled();
    expect(onSettingsClose).not.toHaveBeenCalled();
  });
});
