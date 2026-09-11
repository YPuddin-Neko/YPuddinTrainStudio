import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import Projects from '../src/pages/Projects/Projects';
import { apiClient } from '../src/api/client';
import type { Project } from '../src/api/types';
import i18n from '../src/i18n';

let projects: Project[];
function Location() { const location = useLocation(); return <output data-testid="project-location">{location.pathname}{location.search}</output>; }
function Destination() { const navigate = useNavigate(); return <div>Opened project<button onClick={() => navigate(-1)}>Browser back</button></div>; }
function show(url = '/projects') {
  return render(<MemoryRouter initialEntries={[url]}><Routes><Route path="/projects" element={<Projects/>}/><Route path="/projects/:id" element={<Destination/>}/></Routes><Location/></MemoryRouter>);
}
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  projects = Array.from({ length: 73 }, (_, index) => ({ id: `p${index}`, name: `样本 ${String(index).padStart(3, '0')}`, note: `notes-${index}`, archived: index % 2 === 1, created_at: 1, updated_at: 1, dataset_ids: [], version_count: 1, layout_version: 2, stats: { jobs: 0, artifacts: 0 } }) as Project);
  vi.spyOn(apiClient, 'get').mockImplementation(async () => projects as any);
  vi.spyOn(apiClient, 'post').mockImplementation(async (_url, body) => body as any);
  vi.spyOn(apiClient, 'patch').mockImplementation(async (url, body) => { const row = projects.find(project => url === `/projects/${project.id}`)!; Object.assign(row, body); return row as any; });
  vi.spyOn(apiClient, 'delete').mockImplementation(async url => { projects = projects.filter(project => url !== `/projects/${project.id}`); return undefined as any; });
});
afterEach(() => vi.restoreAllMocks());

it('paginates 73 projects and resets the page when searching or changing the archive filter', async () => {
  show(); await screen.findByTestId('project-card-p0');
  expect(screen.getAllByTestId(/^project-card-/)).toHaveLength(24);
  fireEvent.click(screen.getByRole('button', { name: '下一页' }));
  expect(screen.getByTestId('project-card-p24')).toBeInTheDocument();
  expect(screen.queryByTestId('project-card-p0')).not.toBeInTheDocument();
  expect(screen.getByTestId('project-location')).toHaveTextContent('page=2');
  fireEvent.change(screen.getByRole('textbox', { name: '搜索项目' }), { target: { value: '样本 06' } });
  expect(screen.getAllByTestId(/^project-card-/)).toHaveLength(10);
  expect(screen.queryByRole('navigation', { name: '项目分页' })).not.toBeInTheDocument();
  expect(screen.getByTestId('project-location')).not.toHaveTextContent('page=');
  fireEvent.change(screen.getByRole('textbox', { name: '搜索项目' }), { target: { value: '' } });
  fireEvent.click(screen.getByRole('button', { name: '下一页' }));
  fireEvent.click(screen.getByTestId('show-archived-toggle'));
  expect(screen.getByLabelText('当前页')).toHaveTextContent('1 / 2');
  expect(screen.getByTestId('project-card-p0')).toBeInTheDocument();
  expect(screen.queryByTestId('project-card-p1')).not.toBeInTheDocument();
  expect(screen.getByRole('navigation', { name: '项目分页' })).toHaveAttribute('title', '共 37 个项目 · 每页 24 个');
  expect(document.querySelector('select')).toBeNull();
});

it('restores the list search and page after opening a project and using browser back', async () => {
  show('/projects?q=%E6%A0%B7%E6%9C%AC&page=2');
  const card = await screen.findByTestId('project-card-p24');
  fireEvent.click(within(card).getByRole('link', { name: '打开项目：样本 024' }));
  expect(screen.getByTestId('project-location')).toHaveTextContent('/projects/p24');
  fireEvent.click(screen.getByRole('button', { name: 'Browser back' }));
  await screen.findByTestId('project-card-p24');
  expect(screen.getByRole('textbox', { name: '搜索项目' })).toHaveValue('样本');
  expect(screen.getByLabelText('当前页')).toHaveTextContent('2 / 4');
  expect(screen.getAllByTestId(/^project-card-/)).toHaveLength(24);
});

it('clamps an empty last page after deleting its final record', async () => {
  vi.spyOn(window, 'confirm').mockReturnValue(true);
  show('/projects?page=4');
  const card = await screen.findByTestId('project-card-p72');
  fireEvent.click(within(card).getByTitle('删除'));
  await screen.findByTestId('project-card-p48');
  expect(screen.getByLabelText('当前页')).toHaveTextContent('3 / 3');
  await waitFor(() => expect(screen.getByTestId('project-location')).toHaveTextContent('page=3'));
  expect(screen.queryByTestId('projects-empty')).not.toBeInTheDocument();
});

it('clears a failed list load after retry and retains a failed rename inside its editable dialog', async () => {
  vi.mocked(apiClient.get).mockRejectedValueOnce(new Error('List temporarily unavailable'));
  show();
  expect(await screen.findByRole('alert')).toHaveTextContent('List temporarily unavailable');
  fireEvent.click(screen.getByRole('button', { name: '重试' }));
  const card = await screen.findByTestId('project-card-p0');
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  const opener = within(card).getByTitle('重命名'); opener.focus(); fireEvent.click(opener);
  const dialog = screen.getByRole('dialog', { name: '重命名项目' });
  fireEvent.change(within(dialog).getByRole('textbox', { name: '项目名称' }), { target: { value: '保留的新名称' } });
  let reject: (error: Error) => void = () => {};
  vi.mocked(apiClient.patch).mockImplementationOnce(() => new Promise((_resolve, fail) => { reject = fail; }));
  fireEvent.click(within(dialog).getByRole('button', { name: '保存' }));
  expect(within(dialog).getByRole('button', { name: '关闭' })).toBeDisabled();
  expect(within(dialog).getByRole('textbox', { name: '项目名称' })).toBeDisabled();
  fireEvent.keyDown(dialog, { key: 'Escape' });
  expect(dialog).toBeInTheDocument();
  await act(async () => reject(new Error('Rename failed; retry')));
  expect(within(dialog).getByRole('alert')).toHaveTextContent('Rename failed; retry');
  expect(within(dialog).getByRole('textbox', { name: '项目名称' })).toHaveValue('保留的新名称');
  expect(within(dialog).getByRole('textbox', { name: '项目名称' })).toBeEnabled();
  fireEvent.click(within(dialog).getByRole('button', { name: '保存' }));
  await screen.findByRole('link', { name: '打开项目：保留的新名称' });
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(screen.queryByRole('alert')).not.toBeInTheDocument();
});

it('traps focus in project dialogs and restores the opener after Escape', async () => {
  show(); await screen.findByTestId('project-card-p0');
  const opener = screen.getByRole('button', { name: '新建项目' }); opener.focus(); fireEvent.click(opener);
  const dialog = screen.getByRole('dialog', { name: '新建项目' });
  fireEvent.change(screen.getByTestId('project-name-input'), { target: { value: '角色新项目' } });
  fireEvent.change(screen.getByTestId('project-id-input'), { target: { value: 'Character_01' } });
  const submit = within(dialog).getByRole('button', { name: '创建' }); submit.focus(); fireEvent.keyDown(submit, { key: 'Tab' });
  expect(within(dialog).getByRole('button', { name: '关闭' })).toHaveFocus();
  fireEvent.keyDown(document.activeElement!, { key: 'Escape' });
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(opener).toHaveFocus();
});

it('prevents dismissing a pending create and opens the created project once the server responds', async () => {
  show(); await screen.findByTestId('project-card-p0');
  fireEvent.click(screen.getByRole('button', { name: '新建项目' }));
  const dialog = screen.getByRole('dialog', { name: '新建项目' });
  fireEvent.change(screen.getByTestId('project-name-input'), { target: { value: '角色新项目' } });
  fireEvent.change(screen.getByTestId('project-id-input'), { target: { value: 'Character_01' } });
  let resolve: (value: any) => void = () => {};
  vi.mocked(apiClient.post).mockImplementationOnce(() => new Promise(done => { resolve = done; }));
  fireEvent.click(within(dialog).getByRole('button', { name: '创建' }));
  expect(within(dialog).getByRole('button', { name: '关闭' })).toBeDisabled();
  expect(within(dialog).getByRole('button', { name: '取消' })).toBeDisabled();
  fireEvent.keyDown(dialog, { key: 'Escape' });
  fireEvent.mouseDown(dialog.parentElement!);
  expect(dialog).toBeInTheDocument();
  await act(async () => resolve({ id: 'Character_01', name: '角色新项目', active_version_id: 'v_first' }));
  await waitFor(() => expect(screen.getByTestId('project-location')).toHaveTextContent('/projects/Character_01'));
  expect(apiClient.post).toHaveBeenCalledTimes(1);
  expect(apiClient.post).toHaveBeenCalledWith('/projects', { id: 'Character_01', name: '角色新项目', note: '' }, { silent: true });
});

it('shows a compact single-project row with one entry link, inline metadata and no empty-note or single-page filler', async () => {
  projects = [{ ...projects[0], note: '', version_count: 3, dataset_ids: ['d1', 'd2'], stats: { jobs: 7, artifacts: 4 } }];
  show();
  const row = await screen.findByRole('listitem');
  expect(within(row).getAllByRole('link')).toHaveLength(1);
  expect(within(row).getByRole('link', { name: '打开项目：样本 000' })).toHaveAttribute('href', '/projects/p0');
  expect(within(row).getByLabelText('版本: 3')).toBeInTheDocument();
  expect(within(row).getByLabelText(`${i18n.t('projects.datasets')}: 2`)).toBeInTheDocument();
  expect(within(row).getByLabelText(`${i18n.t('projects.jobs')}: 7`)).toBeInTheDocument();
  expect(within(row).getByLabelText(`${i18n.t('projects.artifacts')}: 4`)).toBeInTheDocument();
  expect(screen.queryByText(i18n.t('projects.noNote'))).not.toBeInTheDocument();
  expect(screen.queryByText(i18n.t('projects.createdAt'))).not.toBeInTheDocument();
  expect(screen.queryByRole('navigation', { name: '项目分页' })).not.toBeInTheDocument();
  fireEvent.click(within(row).getByRole('button', { name: `${i18n.t('projects.archive')}: 样本 000` }));
  const restored = await screen.findByRole('button', { name: `${i18n.t('projects.unarchive')}: 样本 000` });
  expect(apiClient.patch).toHaveBeenCalledWith('/projects/p0', { archived: true });
  expect(screen.getByTestId('project-location')).toHaveTextContent('/projects');
  fireEvent.click(restored);
  await screen.findByRole('button', { name: `${i18n.t('projects.archive')}: 样本 000` });
  expect(apiClient.patch).toHaveBeenLastCalledWith('/projects/p0', { archived: false });
});
