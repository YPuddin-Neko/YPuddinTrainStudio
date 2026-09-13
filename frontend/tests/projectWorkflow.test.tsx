import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeAll, afterEach, afterAll, describe, expect, it, vi } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { handlers } from '../src/mocks/handlers';
import Projects from '../src/pages/Projects/Projects';
import ProjectDetail from '../src/pages/ProjectDetail/ProjectDetail';
import ProjectDataImport from '../src/pages/ProjectDetail/ProjectDataImport';
import TrainConfig from '../src/pages/TrainConfig/TrainConfig';
import { schemaDefaults } from '../src/utils/config';
import { fillDefaultModels } from '../src/utils/workspaceConfig';
import trainSchema from '../src/schema/train-schema.json';
import type { ModelAsset } from '../src/api/types';
import { File as NodeFile } from 'node:buffer';
import '../src/i18n';

vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {} }));
const server = setupServer(...handlers);
beforeAll(async () => {
  // Fetch uses Undici; use its native multipart classes rather than jsdom Blobs.
  const nativeForm = await new Request('http://localhost', { method: 'POST', headers: { 'content-type': 'application/x-www-form-urlencoded' }, body: '' }).formData();
  vi.stubGlobal('FormData', nativeForm.constructor);
  vi.stubGlobal('File', NodeFile);
  server.listen({ onUnhandledRequest: 'error' });
});
afterEach(() => { sessionStorage.clear(); server.resetHandlers(); vi.restoreAllMocks(); });
afterAll(() => { server.close(); vi.unstubAllGlobals(); });
const project = { id: 'p_work', name: 'Character workspace', dataset_ids: [], note: '', archived: false, created_at: 1, updated_at: 1, stats: { jobs: 0, artifacts: 0 } };
const source = { id: 'd_uploaded', project_id: 'p_work', path: 'D:/trainer/data/uploaded', repeats: 3, caption_ext: '.txt', is_reg: false, prior_weight: 1, class_prompt: null, created_at: 1 };
const indexed = { source, stats: { images: 1, captioned: 1, masks: 0 }, index_status: 'ready', cache: {} };
const assets = [
  { id: 'anima-dit', family: 'anima', kind: 'dit', path: 'D:/models/anima.safetensors', exists: true, is_default: true },
  { id: 'anima-te', family: 'anima', kind: 'text_encoder', path: 'D:/models/qwen3.safetensors', exists: true, is_default: true },
  { id: 'anima-vae', family: 'anima', kind: 'vae', path: 'D:/models/vae.safetensors', exists: true, is_default: true },
  { id: 'krea-dit', family: 'krea2', kind: 'dit', path: 'D:/models/krea.safetensors', exists: true, is_default: true },
] as ModelAsset[];
function show(path = '/projects/p_work') {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <MemoryRouter initialEntries={[path]}><Routes>
      <Route path="/projects" element={<Projects />} />
      <Route path="/projects/:id" element={<ProjectDetail />} />
      <Route path="/projects/:id/train" element={<TrainConfig />} />
      <Route path="/jobs/:id" element={<div>Training job created</div>} />
      <Route path="/datasets/:id" element={<div>Caption editor opened</div>} />
    </Routes></MemoryRouter>
  </QueryClientProvider>);
}
function workspaceHandlers() {
  let config = schemaDefaults(trainSchema);
  config.dataset.sources = [];
  let datasets: any[] = [];
  server.use(
    http.get('/api/projects', () => HttpResponse.json([])),
    http.post('/api/projects', async ({ request }) => HttpResponse.json({ ...project, ...await request.json() as object })),
    http.get('/api/projects/p_work', () => HttpResponse.json(project)),
    http.get('/api/projects/p_work/versions', () => HttpResponse.json([])),
    http.get('/api/projects/p_work/config', () => HttpResponse.json(config)),
    http.put('/api/projects/p_work/config', async ({ request }) => { config = await request.json() as any; return HttpResponse.json(config); }),
    http.get('/api/projects/p_work/datasets', () => HttpResponse.json(datasets)),
    http.get('/api/models', () => HttpResponse.json(assets)),
    http.get('/api/jobs', () => HttpResponse.json({ items: [], total: 0, page: 1, page_size: 50 })),
    http.get('/api/artifacts', () => HttpResponse.json([])),
  );
  return { config: () => config, addSource: () => { datasets = [indexed]; config.dataset.sources = [{ path: source.path, repeats: source.repeats, caption_ext: '.txt', is_reg: false, prior_weight: 1, class_prompt: null }]; } };
}

describe('project training workspace', () => {
  it('creates a project, uploads files, saves model choices and starts a job with that data and edited parameters', async () => {
    const backend = workspaceHandlers();
    let uploadText = '';
    let submitted: any;
    let planned: any;
    server.use(
      http.post('/api/projects/p_work/datasets/upload', async ({ request }) => {
        expect(request.headers.get('content-type')).toContain('multipart/form-data; boundary=');
        // Read multipart bytes directly: jsdom File differs from Undici's File class.
        uploadText = await request.text(); backend.addSource(); return HttpResponse.json(indexed);
      }),
      http.post('/api/plan', async ({ request }) => { planned = await request.json(); return HttpResponse.json({ ok: true, errors: [], warnings: [], total_steps: 8, steps_per_epoch: 2 }); }),
      http.post('/api/jobs', async ({ request }) => { submitted = await request.json(); return HttpResponse.json({ id: 'j_created' }); }),
    );
    show('/projects');
    fireEvent.click(screen.getByRole('button', { name: '新建项目' }));
    fireEvent.change(screen.getByTestId('project-name-input'), { target: { value: 'Character workspace' } });
    fireEvent.change(screen.getByTestId('project-id-input'), { target: { value: 'p_work' } });
    await waitFor(() => expect(within(screen.getByTestId('create-project-modal')).getByRole('button', { name: '创建' })).toBeEnabled());
    fireEvent.click(within(screen.getByTestId('create-project-modal')).getByRole('button', { name: '创建' }));
    await screen.findByTestId('project-overview');
    fireEvent.click(screen.getByRole('link', { name: /^1\s*训练数据$/ }));
    await screen.findByTestId('project-data-import');
    expect(within(screen.getByRole('navigation', { name: '项目训练步骤' })).getAllByRole('link')).toHaveLength(4);
    expect(screen.queryByRole('link', { name: /模型准备/ })).not.toBeInTheDocument();
    const image = new File(['image bytes'], 'portrait.png', { type: 'image/png' });
    const caption = new File(['a character'], 'portrait.txt', { type: 'text/plain' });
    const discarded = new File(['unused'], 'discard.png', { type: 'image/png' });
    fireEvent.click(screen.getByText('导入选项'));
    expect(screen.getByRole('combobox', { name: '标签格式' })).toHaveTextContent('自动');
    expect(screen.queryByRole('spinbutton', { name: '正则损失权重' })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('选择训练文件'), { target: { files: [image, caption, discarded] } });
    fireEvent.click(screen.getByRole('button', { name: '移除 discard.png' }));
    fireEvent.change(screen.getByRole('spinbutton', { name: '每张图片重复次数' }), { target: { value: '3' } });
    fireEvent.click(screen.getByRole('button', { name: '导入当前版本' }));
    await screen.findByRole('link', { name: '查看图片与标签' });
    expect(uploadText).toContain('filename="portrait.png"');
    expect(uploadText).toContain('filename="portrait.txt"');
    expect(uploadText).not.toContain('discard.png');
    expect(uploadText).toContain('name="repeats"\r\n\r\n3');
    expect(uploadText).toContain('name="caption_ext"\r\n\r\nauto');
    expect(await screen.findByTestId('dataset-card-d_uploaded')).toHaveTextContent('1 张图片');
    fireEvent.click(screen.getByRole('link', { name: /^2\s*训练参数$/ }));
    fireEvent.click(await screen.findByRole('tab', { name: '底模与输出' }));
    expect(await screen.findByRole('combobox', { name: 'model.family' })).toHaveTextContent(/anima/i);
    await waitFor(() => expect(screen.getByDisplayValue('D:/models/anima.safetensors')).toBeInTheDocument());
    fireEvent.click(screen.getByRole('tab', { name: '训练参数' }));
    await screen.findByTestId('field-loop.epochs');
    expect(screen.queryByRole('region', {name:'常用训练参数'})).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole('spinbutton', { name: 'loop.epochs' }), { target: { value: '4' } });
    fireEvent.change(screen.getByRole('spinbutton', { name: '学习率' }), { target: { value: '0.0002' } });
    fireEvent.click(screen.getByRole('tab', {name:'数据与分桶'}));
    fireEvent.change(within(screen.getByTestId('field-dataset.resolutions')).getByRole('textbox'), { target: { value: '768' } });
    const start = screen.getByRole('button', { name: '开始训练' });
    await waitFor(() => expect(start).toBeEnabled());
    expect(start).toHaveTextContent('开始训练');
    expect(planned.config.dataset.sources).toEqual(backend.config().dataset.sources);
    fireEvent.click(start);
    await screen.findByText('Training job created');
    expect(submitted).toMatchObject({ project_id: 'p_work', type: 'train', config: { model: { family: 'anima', dit_path: 'D:/models/anima.safetensors' }, loop: { epochs: 4 }, optimizer: { lr: 0.0002 }, dataset: { resolutions: [768], sources: [{ path: source.path, repeats: 3 }] } } });
    expect(backend.config().optimizer.lr).toBe(0.0002);
  });

  it('keeps failed uploads selected and shows the backend reason', async () => {
    workspaceHandlers();
    server.use(http.post('/api/projects/p_work/datasets/upload', () => HttpResponse.json({ error: { code: 'upload.invalid', message: 'ZIP contains no supported images' } }, { status: 400 })));
    show();
    await screen.findByTestId('project-overview');
    fireEvent.click(screen.getByRole('link', { name: /^1\s*训练数据$/ }));
    await screen.findByTestId('project-data-import');
    fireEvent.change(screen.getByLabelText('选择训练文件'), { target: { files: [new File(['zip'], 'empty.zip', { type: 'application/zip' })] } });
    fireEvent.click(screen.getByRole('button', { name: '导入当前版本' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('ZIP contains no supported images');
    expect(screen.getByText('empty.zip')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '导入当前版本' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: '清空选择' }));
    expect(screen.getByRole('button', { name: '导入当前版本' })).toBeDisabled();
  });

  it('uploads regularization images into the explicit version with source metadata', async () => {
    let fields:Record<string,string>={};let scope='';const imported=vi.fn();
    server.use(http.post('/api/projects/p_work/datasets/upload',async({request})=>{
      scope=new URL(request.url).searchParams.get('version_id')||'';
      const form=await request.formData();fields=Object.fromEntries(['is_reg','prior_weight','class_prompt','repeats','caption_ext'].map(key=>[key,String(form.get(key))]));
      expect(form.getAll('files')).toHaveLength(2);
      return HttpResponse.json({...indexed,source:{...source,is_reg:true,version_id:'v2'}});
    }));
    render(<MemoryRouter><ProjectDataImport projectId="p_work" versionId="v2" defaultIsReg onImported={imported}/></MemoryRouter>);
    expect(screen.getByRole('heading', { name: '添加已有正则图' })).toBeInTheDocument();
    fireEvent.click(screen.getByText('导入选项'));
    expect(screen.queryByRole('checkbox', { name: /正则/ })).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('选择训练文件'),{target:{files:[new File(['image'],'class.png'),new File(['a person'],'class.txt')]}});
    fireEvent.change(screen.getByRole('textbox',{name:'类别提示词'}),{target:{value:'a person'}});
    fireEvent.change(screen.getByRole('spinbutton',{name:'正则损失权重'}),{target:{value:'0.5'}});
    fireEvent.click(screen.getByRole('button',{name:'导入当前版本'}));
    await waitFor(()=>expect(imported).toHaveBeenCalledOnce());
    expect(scope).toBe('v2');expect(fields).toEqual({is_reg:'true',prior_weight:'0.5',class_prompt:'a person',repeats:'1',caption_ext:'auto'});
  });

  it('imports a training-machine path and includes regularization fields', async () => {
    const backend = workspaceHandlers(); let body: any;
    server.use(http.post('/api/projects/p_work/datasets', async ({ request }) => { body = await request.json(); backend.addSource(); return HttpResponse.json(indexed); }));
    render(<MemoryRouter><ProjectDataImport projectId="p_work" defaultIsReg onImported={backend.addSource}/></MemoryRouter>);
    expect(screen.getByRole('heading', { name: '添加已有正则图' })).toBeInTheDocument();
    fireEvent.click(screen.getByText('导入选项'));
    expect(screen.queryByRole('checkbox', { name: /正则/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '从训练电脑导入' }));
    fireEvent.change(within(screen.getByRole('group', { name: '训练图片文件夹路径' })).getByRole('textbox'), { target: { value: 'D:\\photos\\regularization' } });
    fireEvent.change(screen.getByRole('textbox', { name: '类别提示词' }), { target: { value: 'a person' } });
    fireEvent.change(screen.getByRole('spinbutton', { name: '正则损失权重' }), { target: { value: '0.5' } });
    fireEvent.click(screen.getByRole('button', { name: '导入当前版本' }));
    await screen.findByRole('link', { name: '查看图片与标签' });
    expect(body).toMatchObject({ path: 'D:\\photos\\regularization', is_reg: true, prior_weight: 0.5, class_prompt: 'a person', caption_ext: 'auto' });
  });

  it('changes model families without carrying old component paths into the new family', async () => {
    const backend = workspaceHandlers();
    show('/projects/p_work?step=models');
    await waitFor(() => expect(screen.getByDisplayValue('D:/models/anima.safetensors')).toBeInTheDocument());
    fireEvent.click(screen.getByRole('combobox', { name: 'model.family' }));
    fireEvent.click(await screen.findByRole('option', { name: /krea/i }));
    expect(await screen.findByDisplayValue('D:/models/krea.safetensors')).toBeInTheDocument();
    expect(screen.queryByDisplayValue('D:/models/qwen3.safetensors')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }));
    await waitFor(() => expect(backend.config().model).toMatchObject({ family: 'krea2', dit_path: 'D:/models/krea.safetensors', text_encoder_path: null, vae_path: null }));
  });

  it('fills only existing defaults of the right family and preserves explicit model paths', () => {
    const config = { model: { family: 'anima', dit_path: 'D:/custom/my-anima.safetensors' } };
    const result = fillDefaultModels(config, [...assets, { ...assets[0], kind: 'tokenizer', path: 'D:/missing', exists: false }]);
    expect(result.model).toMatchObject({ dit_path: 'D:/custom/my-anima.safetensors', text_encoder_path: 'D:/models/qwen3.safetensors', tokenizer_path: null });
  });

  it('saves parameter changes when immediately navigating back to another project step', async () => {
    const backend = workspaceHandlers();
    show('/projects/p_work/train?tab=train');
    await screen.findByRole('spinbutton', { name: '学习率' });
    fireEvent.change(screen.getByRole('spinbutton', { name: '学习率' }), { target: { value: '0.0007' } });
    fireEvent.click(screen.getByRole('link', { name: /^1\s*训练数据$/ }));
    await screen.findByTestId('project-data-import');
    await waitFor(() => expect(backend.config().optimizer.lr).toBe(0.0007));
  });
});
