import { fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import Artifacts from '../../../frontend/src/pages/Artifacts/Artifacts';
import { apiClient } from '../../../frontend/src/api/client';
import i18n from '../../../frontend/src/i18n';

vi.mock('../../../frontend/src/events/useEventStream', () => ({ useEventStream: () => {} }));
const model = { id: 'model', name: 'character-final.model', project_id: 'project', version_id: 'version', job_id: 'job', kind: 'model', path: '/outputs/character-final.model', created_at: 1700000000, size: 4096, metadata: { components: ['backbone', 'text_encoder'] } };
const adapter = (kind: string) => ({ ...model, id: kind, name: `${kind}.safetensors`, path: `/outputs/${kind}.safetensors`, kind, algo: 'lora', rank: 16, alpha: 8, factor: -1, metadata: {} });
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
afterEach(() => vi.restoreAllMocks());
const show = () => render(<MemoryRouter><Artifacts embedded projectId="project" versionId="version"/></MemoryRouter>);

it('labels complete models and their components, downloads ZIP, and never offers adapter conversion', async () => {
  vi.spyOn(apiClient, 'get').mockResolvedValue([model] as any);
  show();
  const row = await screen.findByTestId('artifact-row-model');
  expect(row).toHaveTextContent('全量模型');
  expect(row).toHaveTextContent('UNet / DiT + 文本编码器');
  expect(row).not.toHaveTextContent('Rank');
  expect(screen.getByRole('columnheader', { name: '模型组件' })).toBeInTheDocument();
  expect(within(row).queryByRole('combobox', { name: /转换格式/ })).not.toBeInTheDocument();
  expect(within(row).getByRole('link', { name: '下载: character-final.model' })).toHaveAttribute('download', 'character-final.model.zip');
  expect(within(row).getByRole('link', { name: '下载: character-final.model' })).toHaveAttribute('href', expect.stringContaining('/api/artifacts/model/download'));
});

it('keeps raw and converted adapter outputs convertible but excludes model and unknown kinds', async () => {
  const rows = [model, ...['weights', 'comfyui', 'kohya', 'unrecognized'].map(adapter)];
  vi.spyOn(apiClient, 'get').mockResolvedValue(rows as any);
  const convert = vi.spyOn(apiClient, 'post').mockResolvedValue(adapter('kohya') as any);
  show();
  await screen.findByTestId('artifact-row-model');
  for (const kind of ['weights', 'comfyui', 'kohya']) expect(within(screen.getByTestId(`artifact-row-${kind}`)).getByRole('combobox', { name: /转换格式/ })).toBeInTheDocument();
  expect(within(screen.getByTestId('artifact-row-unrecognized')).queryByRole('combobox')).not.toBeInTheDocument();
  fireEvent.click(within(screen.getByTestId('artifact-row-comfyui')).getByRole('combobox', { name: /转换格式/ }));
  expect(screen.getByRole('option', { name: 'comfyui' })).toHaveAttribute('aria-disabled', 'true');
  fireEvent.click(screen.getByRole('option', { name: 'kohya' }));
  expect(convert).toHaveBeenCalledWith('/artifacts/comfyui/convert', { format: 'kohya' }, { silent: true });
});
