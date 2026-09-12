import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Projects from '../src/pages/Projects/Projects';
import { apiClient } from '../src/api/client';
import type { Project } from '../src/api/types';
import i18n from '../src/i18n';

let projects: Project[];
function Location() { const location = useLocation(); return <output data-testid="project-location">{location.pathname}{location.search}</output>; }
function Destination() { const navigate = useNavigate(); return <div>Opened project<button onClick={() => navigate(-1)}>Browser back</button></div>; }
function show(url = '/projects') {
  return render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><MemoryRouter initialEntries={[url]}><Routes><Route path="/projects" element={<Projects/>}/><Route path="/projects/:id" element={<Destination/>}/></Routes><Location/></MemoryRouter></QueryClientProvider>);
}
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, writable: true, value: vi.fn(() => 'blob:project-cover') });
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, writable: true, value: vi.fn() });
  projects = Array.from({ length: 73 }, (_, index) => ({ id: `p${index}`, name: `样本 ${String(index).padStart(3, '0')}`, note: `notes-${index}`, archived: index % 2 === 1, created_at: 1, updated_at: 1, dataset_ids: [], version_count: 1, layout_version: 2, stats: { jobs: 0, artifacts: 0 } }) as Project);
  vi.spyOn(apiClient, 'get').mockImplementation(async (endpoint, options) => (endpoint === '/families' ? [{name:'anima',label:'Anima'},{name:'krea2',label:'Krea 2'}] : projects.filter(project => options?.params?.include_archived === true || !project.archived)) as any);
  vi.spyOn(apiClient, 'post').mockImplementation(async (_url, body) => body as any);
  vi.spyOn(apiClient, 'patch').mockImplementation(async (url, body) => { const row = projects.find(project => url === `/projects/${project.id}`)!; Object.assign(row, body); return row as any; });
  vi.spyOn(apiClient, 'delete').mockImplementation(async url => { projects = projects.filter(project => url !== `/projects/${project.id}`); return undefined as any; });
});
afterEach(() => vi.restoreAllMocks());
function editCard(id = 'p0') { fireEvent.click(within(screen.getByTestId(`project-card-${id}`)).getByRole('button', { name: /更多操作/ })); fireEvent.click(screen.getByRole('menuitem', { name: '编辑项目' })); return screen.getByRole('dialog', { name: '编辑项目' }); }
function select(label: string, option: string | RegExp) { fireEvent.click(screen.getByRole('combobox', { name: label })); fireEvent.click(screen.getByRole('option', { name: option })); }
function coverFile() { return new File(['image bytes'], 'chosen-cover.png', { type: 'image/png' }); }

it('paginates 73 projects and resets the page when searching or changing the archive filter', async () => {
  show(); await screen.findByTestId('project-card-p0');
  expect(apiClient.get).toHaveBeenCalledWith('/projects', { params: { include_archived: true }, silent: true });
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
  fireEvent.click(within(card).getByRole('button', { name: /更多操作/ }));
  fireEvent.click(screen.getByRole('menuitem', { name: '删除项目' }));
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
  const opener = within(card).getByRole('button', { name: /更多操作/ }); opener.focus(); fireEvent.click(opener);
  fireEvent.click(screen.getByRole('menuitem', { name: '编辑项目' }));
  const dialog = screen.getByRole('dialog', { name: '编辑项目' });
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
  await waitFor(() => expect(within(dialog).getByRole('button', { name: '创建' })).toBeEnabled());
  fireEvent.click(within(dialog).getByRole('button', { name: '创建' }));
  expect(within(dialog).getByRole('button', { name: '关闭' })).toBeDisabled();
  expect(within(dialog).getByRole('button', { name: '取消' })).toBeDisabled();
  fireEvent.keyDown(dialog, { key: 'Escape' });
  fireEvent.mouseDown(dialog.parentElement!);
  expect(dialog).toBeInTheDocument();
  await act(async () => resolve({ id: 'Character_01', name: '角色新项目', active_version_id: 'v_first' }));
  await waitFor(() => expect(screen.getByTestId('project-location')).toHaveTextContent('/projects/Character_01'));
  expect(apiClient.post).toHaveBeenCalledTimes(1);
  expect(apiClient.post).toHaveBeenCalledWith('/projects', { id: 'Character_01', name: '角色新项目', note: '', category: null, family: 'anima' }, { silent: true });
});

it('shows a compact cover card with one entry link, inline metadata and no empty-note or single-page filler', async () => {
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
  fireEvent.click(within(row).getByRole('button', { name: /更多操作/ }));
  fireEvent.click(screen.getByRole('menuitem', { name: '归档项目' }));
  await waitFor(() => expect(projects[0].archived).toBe(true));
  const archived = await screen.findByTestId('project-card-p0');
  fireEvent.click(within(archived).getByRole('button', { name: /更多操作/ }));
  const restored = screen.getByRole('menuitem', { name: '恢复项目' });
  expect(apiClient.patch).toHaveBeenCalledWith('/projects/p0', { archived: true });
  expect(screen.getByTestId('project-location')).toHaveTextContent('/projects');
  fireEvent.click(restored);
  await waitFor(() => expect(projects[0].archived).toBe(false));
  await screen.findByTestId('project-card-p0');
  expect(apiClient.patch).toHaveBeenLastCalledWith('/projects/p0', { archived: false });
});

it('combines a full-library category with search and archive filters, resetting only pagination', async () => {
  projects = projects.map((project, index) => ({ ...project, category: index < 40 ? '人物 LoRA' : index < 65 ? '画风 LoRA' : null }));
  show('/projects?page=3'); await screen.findByTestId('project-card-p48');
  select('按分类筛选', '人物 LoRA · 40');
  expect(screen.getByLabelText('当前页')).toHaveTextContent('1 / 2');
  expect(screen.getByTestId('project-card-p0')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: '下一页' }));
  expect(screen.getByTestId('project-card-p24')).toBeInTheDocument();
  fireEvent.change(screen.getByRole('textbox', { name: '搜索项目' }), { target: { value: '样本 03' } });
  expect(screen.getAllByRole('listitem')).toHaveLength(10);
  expect(screen.getByTestId('project-location')).toHaveTextContent('category=');
  expect(screen.getByTestId('project-location')).not.toHaveTextContent('page=');
  fireEvent.click(screen.getByTestId('show-archived-toggle'));
  expect(screen.getAllByRole('listitem')).toHaveLength(5);
  fireEvent.change(screen.getByRole('textbox', { name: '搜索项目' }), { target: { value: '' } });
  select('按分类筛选', '未分类 · 8');
  expect(screen.getAllByRole('listitem')).toHaveLength(4);
  expect(screen.getByTestId('project-location')).toHaveTextContent('uncategorized=true');
  expect(screen.getByTestId('project-location')).toHaveTextContent('archived=0');
});

it('keeps cover uploads local until Save and discards cancelled name, category and cover edits', async () => {
  projects = [{ ...projects[0], category: '人物 LoRA', cover_url: '/api/projects/p0/cover?v=1' } as Project];
  show(); await screen.findByTestId('project-card-p0');
  let dialog = editCard();
  fireEvent.change(within(dialog).getByRole('textbox', { name: '项目名称' }), { target: { value: '未保存名称' } });
  select('项目分类', '自定义分类…');
  fireEvent.change(screen.getByRole('textbox', { name: '自定义分类名称' }), { target: { value: '产品实验' } });
  fireEvent.change(within(dialog).getByLabelText('上传项目封面'), { target: { files: [coverFile()] } });
  expect(within(dialog).getByRole('img')).toHaveAttribute('src', 'blob:project-cover');
  fireEvent.click(within(dialog).getByRole('button', { name: '取消' }));
  expect(apiClient.post).not.toHaveBeenCalled(); expect(apiClient.patch).not.toHaveBeenCalled(); expect(apiClient.delete).not.toHaveBeenCalled();
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:project-cover');
  expect(screen.getByRole('img')).toHaveAttribute('src', 'http://localhost:3000/api/projects/p0/cover?v=1');
  dialog = editCard();
  expect(within(dialog).getByRole('textbox', { name: '项目名称' })).toHaveValue('样本 000');
  expect(within(dialog).getByRole('combobox', { name: '项目分类' })).toHaveTextContent('人物 LoRA');
  expect(screen.queryByRole('textbox', { name: '自定义分类名称' })).not.toBeInTheDocument();
});

it('retains a created project and selected file after upload failure, and retries without duplicating creation', async () => {
  show(); await screen.findByTestId('project-card-p0');
  fireEvent.click(screen.getByRole('button', { name: '新建项目' }));
  const dialog = screen.getByRole('dialog', { name: '新建项目' });
  fireEvent.change(screen.getByTestId('project-name-input'), { target: { value: '封面与模型实验' } });
  fireEvent.change(screen.getByTestId('project-id-input'), { target: { value: 'Cover_01' } });
  await waitFor(() => expect(within(dialog).getByRole('button', { name: '创建' })).toBeEnabled());
  select('初始模型类型', 'Krea 2');
  select('项目分类', '画风 LoRA');
  const file = coverFile();
  fireEvent.change(within(dialog).getByLabelText('上传项目封面'), { target: { files: [file] } });
  const created = { ...projects[0], id: 'Cover_01', name: '封面与模型实验', category: '画风 LoRA', cover_url: null };
  let uploads = 0;
  vi.mocked(apiClient.post).mockImplementation(async (endpoint, body) => {
    if (endpoint === '/projects') return created as any;
    expect(endpoint).toBe('/projects/Cover_01/cover'); expect((body as FormData).get('file')).toBe(file);
    uploads += 1; if (uploads === 1) throw new Error('Upload connection lost');
    return { ...created, cover_url: '/api/projects/Cover_01/cover?v=2' } as any;
  });
  fireEvent.click(within(dialog).getByRole('button', { name: '创建' }));
  expect(await within(dialog).findByRole('alert')).toHaveTextContent('项目信息已保存，封面未保存');
  expect(within(dialog).getByRole('alert')).toHaveTextContent('Upload connection lost');
  expect(screen.getByTestId('project-id-input')).toBeDisabled();
  expect(screen.getByTestId('project-name-input')).toHaveValue('封面与模型实验');
  expect(within(dialog).getByRole('combobox', { name: '项目分类' })).toHaveTextContent('画风 LoRA');
  expect(within(dialog).getByText('chosen-cover.png')).toBeInTheDocument();
  fireEvent.click(within(dialog).getByRole('button', { name: '保存' }));
  await waitFor(() => expect(screen.getByTestId('project-location')).toHaveTextContent('/projects/Cover_01'));
  expect(vi.mocked(apiClient.post).mock.calls.filter(([url]) => url === '/projects')).toHaveLength(1);
  expect(apiClient.post).toHaveBeenCalledWith('/projects', { id: 'Cover_01', name: '封面与模型实验', note: '', category: '画风 LoRA', family: 'krea2' }, { silent: true });
  expect(uploads).toBe(2); expect(apiClient.patch).not.toHaveBeenCalled();
});

it('removes a cover only after saving and stores custom categories independently from the model family', async () => {
  projects = [{ ...projects[0], category: null, cover_url: '/api/projects/p0/cover?v=1' } as Project];
  show(); await screen.findByTestId('project-card-p0');
  let dialog = editCard();
  fireEvent.click(within(dialog).getByRole('button', { name: '移除封面' }));
  expect(within(dialog).queryByRole('img')).not.toBeInTheDocument();
  fireEvent.click(within(dialog).getByRole('button', { name: '取消' }));
  expect(apiClient.delete).not.toHaveBeenCalled(); expect(screen.getByRole('img')).toBeInTheDocument();
  dialog = editCard(); select('项目分类', '自定义分类…');
  fireEvent.change(screen.getByRole('textbox', { name: '自定义分类名称' }), { target: { value: ' 产品 LoRA ' } });
  fireEvent.click(within(dialog).getByRole('button', { name: '移除封面' }));
  vi.mocked(apiClient.delete).mockImplementationOnce(async () => ({ ...projects[0], cover_url: null }) as any);
  fireEvent.click(within(dialog).getByRole('button', { name: '保存' }));
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  expect(apiClient.patch).toHaveBeenCalledWith('/projects/p0', { name: '样本 000', note: 'notes-0', category: '产品 LoRA' }, { silent: true });
  expect(apiClient.delete).toHaveBeenCalledWith('/projects/p0/cover', { silent: true });
  expect(screen.queryByRole('img')).not.toBeInTheDocument();
  expect(within(screen.getByTestId('project-card-p0')).getByText('产品 LoRA')).toBeInTheDocument();
});

it('rejects unsupported or oversized cover files without uploading and preserves the editable form', async () => {
  show(); await screen.findByTestId('project-card-p0'); const dialog = editCard();
  const input = within(dialog).getByLabelText('上传项目封面');
  fireEvent.change(input, { target: { files: [new File(['svg'], 'unsafe.svg', { type: 'image/svg+xml' })] } });
  expect(within(dialog).getByRole('alert')).toHaveTextContent('请选择 JPEG、PNG 或 WebP');
  const large = new File(['large'], 'large.png', { type: 'image/png' }); Object.defineProperty(large, 'size', { value: 8 * 1024 * 1024 + 1 });
  fireEvent.change(input, { target: { files: [large] } });
  expect(within(dialog).getByRole('alert')).toHaveTextContent('不能超过 8 MiB');
  expect(within(dialog).getByRole('textbox', { name: '项目名称' })).toBeEnabled();
  expect(apiClient.post).not.toHaveBeenCalled(); expect(apiClient.patch).not.toHaveBeenCalled();
});

it('supports the More menu with keyboard focus and exposes unsupported families as disabled', async () => {
  show(); await screen.findByTestId('project-card-p0');
  const more = within(screen.getByTestId('project-card-p0')).getByRole('button', { name: /更多操作/ }); more.focus();
  fireEvent.keyDown(more, { key: 'ArrowDown' });
  expect(screen.getByRole('menuitem', { name: '编辑项目' })).toHaveFocus();
  fireEvent.keyDown(document.activeElement!, { key: 'ArrowDown' });
  expect(screen.getByRole('menuitem', { name: '归档项目' })).toHaveFocus();
  fireEvent.keyDown(document.activeElement!, { key: 'Escape' });
  expect(screen.queryByRole('menu')).not.toBeInTheDocument(); expect(more).toHaveFocus();
  fireEvent.click(screen.getByRole('button', { name: '新建项目' }));
  await waitFor(() => expect(screen.getByRole('combobox', { name: '初始模型类型' })).toHaveTextContent('Anima'));
  fireEvent.click(screen.getByRole('combobox', { name: '初始模型类型' }));
  expect(screen.getByRole('option', { name: /Flux/ })).toHaveAttribute('aria-disabled', 'true');
  fireEvent.click(screen.getByRole('option', { name: /SDXL/ }));
  expect(screen.getByRole('combobox', { name: '初始模型类型' })).toHaveTextContent('Anima');
  expect(screen.queryByRole('option', { name: /toy/i })).not.toBeInTheDocument();
});

it('preserves the previous cover and editable values when replacing a cover fails, then retries just the upload', async () => {
  projects = [{ ...projects[0], category: '人物 LoRA', cover_url: '/api/projects/p0/cover?v=1' } as Project];
  show(); await screen.findByTestId('project-card-p0'); const dialog = editCard();
  fireEvent.change(within(dialog).getByRole('textbox', { name: '项目名称' }), { target: { value: '已改名称' } });
  fireEvent.change(within(dialog).getByLabelText('上传项目封面'), { target: { files: [coverFile()] } });
  vi.mocked(apiClient.post).mockRejectedValueOnce(new Error('Cover storage is read only'));
  fireEvent.click(within(dialog).getByRole('button', { name: '保存' }));
  expect(await within(dialog).findByRole('alert')).toHaveTextContent('项目信息已保存，封面未保存');
  expect(within(dialog).getByRole('textbox', { name: '项目名称' })).toHaveValue('已改名称');
  expect(within(dialog).getByRole('textbox', { name: '项目名称' })).toBeEnabled();
  expect(within(screen.getByTestId('project-card-p0')).getByRole('img')).toHaveAttribute('src', 'http://localhost:3000/api/projects/p0/cover?v=1');
  vi.mocked(apiClient.post).mockResolvedValueOnce({ ...projects[0], cover_url: '/api/projects/p0/cover?v=2' });
  fireEvent.click(within(dialog).getByRole('button', { name: '保存' }));
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  expect(apiClient.patch).toHaveBeenCalledTimes(1);
  expect(apiClient.post).toHaveBeenCalledTimes(2);
  expect(screen.getByRole('img')).toHaveAttribute('src', 'http://localhost:3000/api/projects/p0/cover?v=2');
});
