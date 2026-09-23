import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import WindowsAttentionWheelPicker, { type WindowsAttentionCatalog, type WindowsAttentionWheel } from '../../../frontend/src/components/WindowsAttentionWheelPicker';
import i18n from '../../../frontend/src/i18n';

const source = 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases';
const server = setupServer();
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

function wheel(release: string, version = '2.8.3', compatible = true): WindowsAttentionWheel {
  const filename = `flash_attn-${version}+cu130torch2.13-cp312-cp312-win_amd64.whl`;
  return {
    id: `${release}-${filename}`, release, package: 'flash-attn', version: `${version}+cu130torch2.13`, filename,
    url: `${source}/download/${release}/${encodeURIComponent(filename)}`, source_url: `${source}/tag/${release}`,
    provider: 'mjun0812-community-windows', size_bytes: 1024, sha256: 'a'.repeat(64), torch: '2.13', cuda: '13.0',
    python_tag: 'cp312', platform_tag: 'win_amd64', validation: 'kernel_probe_required', compatible,
    reason: compatible ? null : 'torch_version_mismatch',
  };
}

function catalog(wheels: WindowsAttentionWheel[], extra: Partial<WindowsAttentionCatalog> = {}): WindowsAttentionCatalog {
  return {
    source_url: source, release: 'v0.9.52', release_count: 2, provider: 'mjun0812-community-windows',
    origin: 'live', checked_at: 1, error: null, reason: null,
    runtime: { python: '3.12.10', torch: '2.13.0+cu130', cuda_runtime: '13.0', machine: 'AMD64' }, wheels, ...extra,
  };
}

function Picker() {
  const [selected, setSelected] = React.useState<WindowsAttentionWheel | null>(null);
  return <WindowsAttentionWheelPicker selected={selected} onSelect={setSelected} disabled={false}/>;
}

describe('Windows community FA2 release discovery', () => {
  it('offers compatible versions from multiple releases and clears selection when refreshing the catalog', async () => {
    const urls: URL[] = [];
    const newest = wheel('v0.9.52');
    const older = wheel('v0.9.40', '2.7.4');
    server.use(http.get('/api/environment/windows/wheels', ({ request }) => {
      const url = new URL(request.url); urls.push(url);
      return HttpResponse.json(catalog(url.searchParams.get('refresh') === 'true' ? [newest] : [newest, older, wheel('v0.9.6', '2.6.3', false)]));
    }));
    render(<Picker/>);
    const select = await screen.findByRole('combobox', { name: 'Windows FlashAttention 版本' });
    expect(screen.getByRole('link', { name: '维护者发布页' })).toHaveAttribute('href', source);
    expect(screen.queryByText(/发布批次 v0.9.6/)).not.toBeInTheDocument();
    fireEvent.click(select);
    expect(screen.getAllByRole('option')).toHaveLength(2);
    expect(screen.queryByRole('option', { name: '选择兼容版本' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('option', { name: /2.7.4.*v0.9.40/ }));
    expect(screen.getByText(older.filename)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '刷新版本' }));
    await waitFor(() => expect(urls).toHaveLength(2));
    expect(urls[1].searchParams.get('refresh')).toBe('true');
    const refreshed = await screen.findByRole('combobox', { name: 'Windows FlashAttention 版本' });
    expect(refreshed).toHaveTextContent('选择兼容版本');
    expect(screen.queryByText(older.filename)).not.toBeInTheDocument();
    fireEvent.click(refreshed);
    expect(screen.getAllByRole('option')).toHaveLength(1);
    expect(screen.getByRole('option', { name: /2.8.3\+cu130torch2.13.*v0.9.52/ })).toBeInTheDocument();
  });

  it('distinguishes an offline bundled list from a complete current release lookup', async () => {
    server.use(http.get('/api/environment/windows/wheels', () => HttpResponse.json(catalog([], {
      release: 'v0.9.6', release_count: 1, origin: 'bundled', error: '无法更新社区版本目录，请检查网络或全局代理。',
    }))));
    render(<Picker/>);
    expect(await screen.findByRole('alert')).toHaveTextContent('请检查网络或全局代理');
    expect(screen.getByText(/当前使用 v0.9.6 缓存列表/)).toBeInTheDocument();
    expect(screen.getByText(/已查询的版本列表中没有适合当前环境的安装包/)).toBeInTheDocument();
    expect(screen.queryByText(/已查询 1 个发布批次/)).not.toBeInTheDocument();
  });

  it('shows cached metadata and bounded or unsigned omissions without calling them incompatibility', async () => {
    server.use(http.get('/api/environment/windows/wheels', () => HttpResponse.json(catalog([wheel('v0.9.52')], {
      origin: 'cached', release_count: 50, limited: true, unverified_assets: 3,
    }))));
    render(<Picker/>);
    await screen.findByRole('combobox', { name: 'Windows FlashAttention 版本' });
    expect(screen.getByText(/使用上次获取的版本列表/)).toBeInTheDocument();
    expect(screen.queryByText(/已查询 50 个发布批次/)).not.toBeInTheDocument();
    expect(screen.getByRole('combobox', {name: 'Windows FlashAttention 版本'})).toBeEnabled();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});
