import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { PathInput } from '../../../frontend/src/components/PathBrowser';
import { apiClient } from '../../../frontend/src/api/client';
import { ApiError, type FsListResponse } from '../../../frontend/src/api/types';
import '../../../frontend/src/i18n';

afterEach(() => vi.restoreAllMocks());
function Editor({ value = 'D:\\models\\current.safetensors', ariaLabel }: { value?: string; ariaLabel?: string }) {
  const [path, setPath] = React.useState(value);
  return <PathInput value={path} onChange={setPath} ariaLabel={ariaLabel} />;
}
function listing(path: string, names: Array<[string, boolean]> = [], parent: string | null = null): FsListResponse {
  return { path, parent, entries: names.map(([name, is_dir]) => ({ name, is_dir, size: 1024, mtime: 1 })) };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

describe('server path browser', () => {
  it('opens the parent of a Windows model file and navigates directories and files with semantic buttons', async () => {
    const get = vi.spyOn(apiClient, 'get').mockImplementation(async (_endpoint, options) => {
      const path = options?.params?.path;
      if (path === 'D:\\models') return listing('D:\\models', [['anima', true]], 'D:\\') as any;
      if (path === 'D:\\models\\anima') return listing('D:\\models\\anima', [['new.safetensors', false]], 'D:\\models') as any;
      throw new Error(`Unexpected directory: ${path}`);
    });
    render(<Editor ariaLabel="底模路径" />);
    const input = screen.getByRole('textbox', { name: '底模路径' });
    expect(input).toHaveClass('min-w-0');
    fireEvent.click(screen.getByRole('button', { name: '浏览' }));
    const dialog = screen.getByRole('dialog', { name: '浏览训练机路径' });
    await screen.findByRole('button', { name: /^anima/ });
    expect(get.mock.calls[0][1]?.params?.path).toBe('D:\\models');
    expect(within(dialog).getByRole('textbox', { name: '训练机上的文件夹地址' })).toHaveValue('D:\\models');
    fireEvent.click(screen.getByRole('button', { name: /^anima/ }));
    fireEvent.click(await screen.findByRole('button', { name: /new\.safetensors/ }));
    expect(input).toHaveValue('D:\\models\\anima\\new.safetensors');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('clears stale contents on a failed address, disables selection, then opens another Windows drive and closes with Escape', async () => {
    const get = vi.spyOn(apiClient, 'get').mockImplementation(async (_endpoint, options) => {
      const path = options?.params?.path;
      if (path === 'D:\\models') return listing('D:\\models', [['old.safetensors', false]]) as any;
      if (path === 'X:\\missing') throw new ApiError(404, { code: 'fs.not_found', message: 'Directory not found: X:\\missing' });
      if (path === 'E:\\') return listing('E:\\') as any;
      throw new Error(`Unexpected directory: ${path}`);
    });
    render(<Editor />);
    const browse = screen.getByRole('button', { name: '浏览' });
    browse.focus(); fireEvent.click(browse);
    await screen.findByRole('button', { name: /old\.safetensors/ });
    const address = screen.getByRole('textbox', { name: '训练机上的文件夹地址' });
    fireEvent.change(address, { target: { value: 'X:\\missing' } });
    expect(screen.getByRole('button', { name: '选择当前目录' })).toBeDisabled();
    expect(screen.queryByRole('button', { name: /old\.safetensors/ })).not.toBeInTheDocument();
    fireEvent.keyDown(address, { key: 'Enter' });
    expect(await screen.findByRole('alert')).toHaveTextContent('Directory not found: X:\\missing');
    fireEvent.click(screen.getByRole('button', { name: '选择当前目录' }));
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: '文件或目录路径' })).toHaveValue('D:\\models\\current.safetensors');
    fireEvent.change(address, { target: { value: 'E:' } });
    fireEvent.click(screen.getByRole('button', { name: '打开目录' }));
    await screen.findByText('这个目录中没有文件。');
    expect(get.mock.calls.at(-1)?.[1]?.params?.path).toBe('E:\\');
    expect(screen.getByRole('button', { name: '选择当前目录' })).toBeEnabled();
    fireEvent.keyDown(address, { key: 'Escape' });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(browse).toHaveFocus();
  });

  it('ignores slow responses after editing or navigating and never enables selection while loading', async () => {
    const old = deferred<FsListResponse>();
    const fresh = deferred<FsListResponse>();
    vi.spyOn(apiClient, 'get').mockImplementation((_endpoint, options) => {
      return (options?.params?.path === 'D:\\models' ? old.promise : fresh.promise) as any;
    });
    render(<Editor />);
    fireEvent.click(screen.getByRole('button', { name: '浏览' }));
    expect(screen.getByRole('status')).toHaveTextContent('正在读取目录');
    expect(screen.getByRole('button', { name: '选择当前目录' })).toBeDisabled();
    const address = screen.getByRole('textbox', { name: '训练机上的文件夹地址' });
    fireEvent.change(address, { target: { value: 'E:\\data' } });
    fireEvent.click(screen.getByRole('button', { name: '打开目录' }));
    await act(async () => { fresh.resolve(listing('E:\\data', [['fresh.png', false]])); await fresh.promise; });
    expect(screen.getByRole('button', { name: /fresh\.png/ })).toBeInTheDocument();
    await act(async () => { old.resolve(listing('D:\\models', [['obsolete.png', false]])); await old.promise; });
    expect(address).toHaveValue('E:\\data');
    expect(screen.queryByRole('button', { name: /obsolete\.png/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '选择当前目录' }));
    await waitFor(() => expect(screen.getByRole('textbox', { name: '文件或目录路径' })).toHaveValue('E:\\data'));
  });
});
