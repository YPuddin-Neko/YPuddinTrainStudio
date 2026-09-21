import trainSchema from '../../../frontend/src/schema/train-schema.json';
import { schemaDefaults } from '../../../frontend/src/utils/config';
import { http, HttpResponse } from 'msw';
import { mockJobs, updateMockJob, removeMockJob } from './mockStore';
import {
  SystemStats,
  Project,
  Job,
  JobListResponse,
  Artifact,
  Plan,
  Settings,
  Preset,
  JobMetrics,
  JobSample,
  JobCheckpoint,
  JobLogResponse,
  QueueSettings,
  FsListResponse,
  DatasetInfo,
  DatasetImage,
  DatasetImagesPage,
  ModelAsset,
} from '../../../frontend/src/api/types';

const customPresets: Preset[] = [];

let mockQueueSettings: QueueSettings = {
  held: false,
  max_concurrent: 1,
  memory_admission: true,
};

let mockSettings: Settings = {
  paths: { bootstrap_env_dir: '',
    data_root: '/Volumes/Service/Dev/data',
    cache_dir: '/Volumes/Service/Dev/cache',
    models_dir: '/Volumes/Service/Dev/models',
    output_dir: '/Volumes/Service/Dev/output',
    output_mode: 'project',
  },
  server: { host: '127.0.0.1', port: 8765 },
  ui: { language: 'zh-CN', theme: 'system' },
};

const mockProjects: Project[] = [
  {
    id: 'proj_01',
    layout_version: 1,
    name: 'Anima Anime Style',
    note: 'Fine-tuning with anime illustration dataset',
    created_at: 1789000000,
    updated_at: 1789000000,
    archived: false,
    version_count: 1,
    dataset_ids: ['ds_01'],
    stats: { jobs: 2, artifacts: 1 },
  },
];



const mockArtifacts: Artifact[] = [
  {
    id: 'art_01',
    project_id: 'proj_01',
    job_id: 'job_01',
    name: 'chara-v1-epoch1.safetensors',
    path: '/models/chara-v1-epoch1.safetensors',
    size: 85000000,
    algo: 'lokr',
    rank: 16,
    alpha: 16,
    factor: 8,
    family: 'anima',
    kind: 'weights',
    step: 200,
    created_at: 1789000000,
    metadata: {},
  },
];

const mockModels: ModelAsset[] = [
  {
    id: 'm_01',
    family: 'anima',
    kind: 'dit',
    path: '/models/anima-dit.safetensors',
    size: 4200000000,
    dtype: 'bf16',
    exists: true,
    is_default: true,
    purpose: 'training',
    created_at: 1789000000,
  },
  {
    id: 'm_02',
    family: 'anima',
    kind: 'text_encoder',
    path: '/models/qwen3-0.6b/',
    size: 1200000000,
    dtype: 'bf16',
    exists: true,
    is_default: true,
    purpose: 'training',
    created_at: 1789000000,
  },
  {
    id: 'm_03',
    family: 'krea2',
    kind: 'dit',
    path: '/models/krea2_fp8_scaled.safetensors',
    variant: 'raw',
    size: 13000000000,
    dtype: 'fp8',
    exists: true,
    is_default: true,
    purpose: 'training',
    created_at: 1789000000,
  },
];

const mockFamilies = [
  {
    name: 'anima',
    label: 'Anima 2B',
    architecture: 'anima',
    adapter_prefix: 'lora_unet',
    capabilities: ['activation_checkpointing', 'block_swap', 'compile', 'fp8_base', 'llm_adapter', 'masked_loss', 'online_text'],
    text_modes: ['auto', 'cached', 'online'],
    presets: [
      { name: 'attn-mlp', description: 'DiT 注意力 + MLP（默认）', include: ['blocks.*.self_attn.{q_proj,k_proj,v_proj,output_proj}', 'blocks.*.mlp.layer1', 'blocks.*.mlp.layer2'], exclude: [], layers: 280 },
      { name: 'attn-only', description: '仅 DiT 注意力投影', include: ['blocks.*.self_attn.{q_proj,k_proj,v_proj,output_proj}'], exclude: [], layers: 224 },
      { name: 'full-linear', description: 'DiT 内全部 Linear（含 AdaLN 调制）', include: ['blocks.*.self_attn.{q_proj,k_proj,v_proj,output_proj}', 'blocks.*.mlp.layer1', 'blocks.*.mlp.layer2', 'blocks.*.adaln_modulation_*.*'], exclude: [], layers: 448 },
    ],
    default_preset: 'attn-mlp',
    sampling: { steps: 25, cfg: 4.0, shift: 3.0, sampler: 'euler' },
    latent: { channels: 16, stride: 8, patch: 2, align: 16 },
    text_max_len: 512,
    weights: [
      { field: 'dit_path', label: 'DiT', hint: 'anima-dit.safetensors（bf16 / fp8_scaled）' },
      { field: 'text_encoder_path', label: 'Qwen3-0.6B', hint: 'HF 目录或单文件 safetensors' },
      { field: 'vae_path', label: 'Qwen-Image VAE', hint: 'qwen_image_vae.safetensors' },
    ],
    linear_modules: 448,
  },
  {
    name: 'krea2',
    label: 'Krea 2 Raw 12.9B',
    architecture: 'krea2',
    adapter_prefix: 'lora_unet',
    capabilities: ['activation_checkpointing', 'block_swap', 'compile', 'fp8_base', 'masked_loss'],
    text_modes: ['auto', 'cached'],
    presets: [
      { name: 'all-linear', description: '全部 264 个 Linear（Krea 官方默认：rank 32 / alpha 32）', include: ['*'], exclude: [], layers: 264 },
      { name: 'attn-mlp', description: '28 个主 block 的注意力 + SwiGLU', include: ['blocks.*.attn.{wq,wk,wv,gate,wo}', 'blocks.*.mlp.{gate,up,down}'], exclude: [], layers: 224 },
      { name: 'attn-only', description: '仅主 block 注意力投影', include: ['blocks.*.attn.{wq,wk,wv,gate,wo}'], exclude: [], layers: 140 },
      { name: 'attn-mlp-text', description: '主 block + 文本融合 transformer + 文本 MLP', include: ['blocks.*.attn.{wq,wk,wv,gate,wo}', 'txtfusion.*_blocks.*.attn.{wq,wk,wv,gate,wo}'], exclude: [], layers: 259 },
    ],
    default_preset: 'attn-mlp',
    sampling: { steps: 28, cfg: 5.5, shift: null, sampler: 'euler' },
    latent: { channels: 16, stride: 8, patch: 2, align: 16 },
    text_max_len: 512,
    weights: [
      { field: 'dit_path', label: 'DiT', hint: 'krea2_raw_bf16.safetensors（约 26 GB）或 Comfy-Org krea2_fp8_scaled.safetensors（约 13 GB，按 fp8 加载）' },
      { field: 'text_encoder_path', label: 'Qwen3-VL-4B-Instruct', hint: 'HF 目录（推荐）或 ComfyUI 单文件 qwen_3vl_4b*.safetensors（bf16 / fp8_scaled）' },
      { field: 'vae_path', label: 'Qwen-Image VAE', hint: 'qwen_image_vae.safetensors（与 Anima 共用）' },
    ],
    linear_modules: 264,
  },
  {
    name: 'toy',
    label: 'Toy DiT',
    architecture: 'toy',
    adapter_prefix: 'lora_unet',
    capabilities: ['online_text'],
    text_modes: ['auto', 'cached', 'online'],
    presets: [
      { name: 'attn-mlp', description: '注意力 + MLP', include: ['blocks.*.attn.*', 'blocks.*.mlp.*'], exclude: [], layers: 20 },
    ],
    default_preset: 'attn-mlp',
    sampling: { steps: 8, cfg: 1.0, shift: 1.0, sampler: 'euler' },
    latent: { channels: 4, stride: 4, patch: 1, align: 8 },
    text_max_len: 128,
    weights: [],
    linear_modules: 20,
  },
];

const mockDatasetInfo: DatasetInfo = {
  source: {
    id: 'ds_01',
    project_id: 'proj_01',
    path: '/data/anime',
    repeats: 2,
    caption_ext: '.txt',
    is_reg: false,
    prior_weight: 1.0,
    class_prompt: null,
    created_at: 1789000000,
  },
  stats: {
    images: 240,
    captioned: 228,
    resolutions: [
      { w: 1024, h: 1024, count: 80 },
      { w: 1024, h: 768, count: 100 },
      { w: 768, h: 1024, count: 60 },
    ],
    ar_hist: [
      { ar: '1.0', count: 80 },
      { ar: '1.3', count: 100 },
      { ar: '0.75', count: 60 },
    ],
    masks: 12,
  },
  index_status: 'ready',
  cache: {
    latents: { cached: 200, total: 240 },
    text: { cached: 240, total: 240 },
  },
};

// 生成 240 张 mock 图片
const TAG_POOL = [
  '1girl', 'solo', 'long_hair', 'blue_eyes', 'smile', 'cat_ears',
  'school_uniform', 'outdoors', 'night', 'sword', 'white_dress', 'looking_at_viewer',
];
const mockImages = Array.from({ length: 240 }, (_, i) => {
  const tags = [TAG_POOL[i % 12], TAG_POOL[(i * 5) % 12], TAG_POOL[(i * 7) % 12]];
  return {
    hash: `hash_${String(i).padStart(5, '0')}`,
    rel_path: `img_${String(i).padStart(4, '0')}.jpg`,
    width: [1024, 768, 832][i % 3],
    height: [1024, 1024, 1216][i % 3],
    caption: tags.join(', '),
    has_mask: i % 20 === 0,
  };
});

const mockCaptions: Record<string, string> = Object.fromEntries(
  mockImages.map((img) => [img.hash, img.caption])
);

function mockCaptionImages(): DatasetImage[] {
  return mockImages.map((image) => {
    const caption = (mockCaptions[image.hash] ?? image.caption).trim();
    return {
      ...image,
      caption,
      caption_tags: caption,
      caption_description: '',
      caption_format: 'txt',
      caption_error: null,
      caption_status: caption ? 'captioned' : 'missing',
    };
  });
}

function mockCaptionTags(caption: string): Map<string, string> {
  const tags = new Map<string, string>();
  for (const value of caption.split(',')) {
    const tag = value.trim();
    if (tag && !tags.has(tag.toLowerCase())) tags.set(tag.toLowerCase(), tag);
  }
  return tags;
}

export const handlers = [
  http.get('/api/system/stats', () => {
    const stats: SystemStats = {
      cpu_pct: 12.5,
      ram: { used_mb: 16384, total_mb: 65536 },
      disks: [{ path: '/data', used_gb: 250, total_gb: 1000 }],
      gpus: [
        { index: 0, kind: 'cuda', name: 'NVIDIA RTX 4090', util_pct: 78, mem_used_mb: 18432, mem_total_mb: 24576, temp_c: 65 },
      ],
    };
    return HttpResponse.json(stats);
  }),

  http.get('/api/system/info', () => {
    return HttpResponse.json({
      python: '3.12.12',
      platform: 'macOS-15.7.9-arm64-arm-64bit',
      packages: { torch: '2.14.0', fastapi: '0.141.1' },
      ypuddin: '0.1.0',
    });
  }),

  http.get('/api/fs/list', ({ request }) => {
    const url = new URL(request.url);
    const path = url.searchParams.get('path') || '/';
    const resp: FsListResponse = {
      path,
      parent: path === '/' ? null : path.substring(0, path.lastIndexOf('/')) || '/',
      entries: [
        { name: 'datasets', is_dir: true, size: 4096, mtime: 1788996000 },
        { name: 'models', is_dir: true, size: 4096, mtime: 1788996000 },
        { name: 'config.toml', is_dir: false, size: 1024, mtime: 1789118400 },
      ],
    };
    return HttpResponse.json(resp);
  }),

  http.get('/api/projects', () => {
    return HttpResponse.json({ items: mockProjects, total: mockProjects.length, page: 1, page_size: 50 });
  }),

  http.post('/api/projects', async ({ request }) => {
    const body = (await request.json()) as any;
    const proj: Project = {
      id: body.id || `p_${Date.now().toString(36)}`,
      layout_version: 2,
      name: body.name,
      note: body.note || '',
      created_at: Date.now() / 1000,
      updated_at: Date.now() / 1000,
      archived: false,
      version_count: 1,
      dataset_ids: [],
      stats: { jobs: 0, artifacts: 0 },
    };
    mockProjects.push(proj);
    return HttpResponse.json(proj);
  }),

  http.get('/api/projects/:id', ({ params }) => {
    const project = mockProjects.find((p) => p.id === params.id) || mockProjects[0];
    return HttpResponse.json(project);
  }),

  http.get('/api/projects/:id/datasets', () => {
    return HttpResponse.json([mockDatasetInfo.source]);
  }),

  http.get('/api/projects/:id/datasets/import-progress/:progressId', () => HttpResponse.json({}, { status: 404 })),

  http.post('/api/projects/:id/datasets', async ({ request }) => {
    const body = (await request.json()) as any;
    return HttpResponse.json({ ...mockDatasetInfo, source: { ...mockDatasetInfo.source, ...body, id: 'ds_new', project_id: 'proj_01' } });
  }),

  http.get('/api/schema/train', () => HttpResponse.json(trainSchema)),
  http.get('/api/config/defaults', () => HttpResponse.json(schemaDefaults(trainSchema))),
  http.put('/api/projects/:id/config', async ({ request }) => HttpResponse.json(await request.json())),
  http.post('/api/config/import', () => HttpResponse.json({ error: { code: 'mock.real_backend_required', message: 'TOML import requires the real backend (VITE_USE_MOCK=false).' } }, { status: 501 })),
  http.post('/api/config/export', () => HttpResponse.json({ error: { code: 'mock.real_backend_required', message: 'TOML export requires the real backend (VITE_USE_MOCK=false).' } }, { status: 501 })),
  http.post('/api/presets', async ({ request }) => {
    const body = await request.json() as any;
    if (customPresets.some(preset => preset.name.toLocaleLowerCase() === String(body.name).toLocaleLowerCase())) {
      return HttpResponse.json({ error: { code: 'preset.duplicate', message: 'Preset already exists.' } }, { status: 409 });
    }
    const preset = { ...body, builtin: false, updated_at: Date.now() / 1000 } as Preset;
    customPresets.push(preset);
    return HttpResponse.json(preset);
  }),
  http.put('/api/presets/:name', async ({ request, params }) => {
    const index = customPresets.findIndex(preset => preset.name === params.name);
    if (index < 0) return HttpResponse.json({ error: { code: 'preset.not_found', message: 'Preset not found.' } }, { status: 404 });
    const body = await request.json() as any;
    const preset = { ...body, name: String(params.name), builtin: false, updated_at: Date.now() / 1000 } as Preset;
    customPresets[index] = preset;
    return HttpResponse.json(preset);
  }),
  http.delete('/api/presets/:name', ({ params }) => {
    const index = customPresets.findIndex(preset => preset.name === params.name);
    if (index < 0) return HttpResponse.json({ error: { code: 'preset.not_found', message: 'Preset not found.' } }, { status: 404 });
    customPresets.splice(index, 1);
    return HttpResponse.json({ ok: true });
  }),

  http.get('/api/projects/:id/config', () => {
    return HttpResponse.json({
      model: { family: 'toy', dtype: 'fp32' },
      dataset: { resolutions: [64], batch_size: 2 },
    });
  }),

  // Explicit mock mode cannot infer filesystem ancestry. Keep fixture metadata.
  http.post('/api/projects/:id/source-roles', async ({ request }) => {
    const { config } = await request.json() as {config:Record<string, any>};
    return HttpResponse.json(['dataset','validation'].flatMap(section => (config[section]?.sources || []).filter((source:any) => source.path).map((source:any) => ({path:source.path,section,is_reg:!!source.is_reg,managed:false,root:null,origin:'external',images:null}))));
  }),
  http.post('/api/projects/:id/output-binding', async ({ request, params }) => {
    const { config } = await request.json() as {config:Record<string, any>};
    const checkpoint = config.checkpoint || {};
    const automatic = !checkpoint.name || checkpoint.name === 'lora';
    return HttpResponse.json({directory_template:`/mock/projects/${params.id}/v1/output/{job_id}`,name:automatic ? `${params.id}_v1` : checkpoint.name,automatic_name:automatic,inherits_output_dir:!checkpoint.output_dir || checkpoint.output_dir === 'outputs/run'});
  }),

  // ---- Datasets ----
  http.get('/api/datasets/:id', () => {
    return HttpResponse.json(mockDatasetInfo);
  }),

  http.post('/api/datasets/:id/rescan', () => {
    return HttpResponse.json(mockDatasetInfo);
  }),

  http.delete('/api/datasets/:id', () => {
    return HttpResponse.json({ ok: true });
  }),

  http.get('/api/datasets/:id/caption-stats', () => {
    // These fixtures contain TXT sidecars only; empty saved files are still TXT.
    const images = mockCaptionImages();
    const tags = new Map<string, { tag: string; count: number }>();
    for (const image of images) {
      for (const [key, tag] of mockCaptionTags(image.caption)) {
        const current = tags.get(key);
        if (current) current.count += 1;
        else tags.set(key, { tag, count: 1 });
      }
    }
    const captioned = images.filter((image) => image.caption_status === 'captioned').length;
    return HttpResponse.json({
      images: images.length,
      captioned,
      missing: images.length - captioned,
      invalid: 0,
      formats: { txt: images.length },
      unique_tags: tags.size,
      tags: [...tags.values()].sort((a, b) => b.count - a.count || a.tag.toLowerCase().localeCompare(b.tag.toLowerCase())),
    });
  }),

  http.get('/api/datasets/:id/images', ({ request }) => {
    const url = new URL(request.url);
    const page = Number(url.searchParams.get('page') || '1');
    const pageSize = Number(url.searchParams.get('page_size') || '60');
    const q = (url.searchParams.get('q') || '').toLowerCase();
    const tag = (url.searchParams.get('tag') || '').trim().toLowerCase();
    const status = url.searchParams.get('caption_status');
    if (status && !['captioned', 'missing', 'invalid'].includes(status)) {
      return HttpResponse.json({ error: { code: 'validation', message: 'Invalid caption status' } }, { status: 422 });
    }
    const filtered = mockCaptionImages().filter((image) =>
      (!q || image.caption.toLowerCase().includes(q) || image.rel_path.toLowerCase().includes(q)) &&
      (!tag || mockCaptionTags(image.caption).has(tag)) &&
      (!status || image.caption_status === status)
    );
    const start = (page - 1) * pageSize;
    const items = filtered.slice(start, start + pageSize);
    const resp: DatasetImagesPage = { items, total: filtered.length, page, page_size: pageSize };
    return HttpResponse.json(resp);
  }),

  http.get('/api/datasets/:id/images/:hash/caption', ({ params, request }) => {
    const path = new URL(request.url).searchParams.get('rel_path');
    const image = mockImages.find((item) => item.hash === params.hash && (!path || item.rel_path === path));
    if (!image) return HttpResponse.json({ error: { code: 'image.not_found', message: 'Image not found' } }, { status: 404 });
    return HttpResponse.json({ caption: (mockCaptions[image.hash] || '').trim() });
  }),

  http.put('/api/datasets/:id/images/:hash/caption', async ({ params, request }) => {
    const path = new URL(request.url).searchParams.get('rel_path');
    const image = mockImages.find((item) => item.hash === params.hash && (!path || item.rel_path === path));
    if (!image) return HttpResponse.json({ error: { code: 'image.not_found', message: 'Image not found' } }, { status: 404 });
    const body = (await request.json()) as { caption: string; description?: string | null };
    if (typeof body.caption !== 'string' || body.description != null) {
      return HttpResponse.json({ error: { code: 'dataset.caption_format', message: 'TXT captions use a single caption field' } }, { status: 422 });
    }
    mockCaptions[image.hash] = body.caption.trim();
    return HttpResponse.json({ caption: body.caption.trim() });
  }),

  http.post('/api/datasets/:id/tags/batch', async ({ request }) => {
    const body = (await request.json()) as { hashes: string[]; add: string[]; remove: string[] };
    for (const h of body.hashes || []) {
      const current = (mockCaptions[h] || '').split(',').map((t) => t.trim()).filter(Boolean);
      const next = current.filter((t) => !(body.remove || []).includes(t));
      for (const a of body.add || []) {
        if (!next.includes(a)) next.push(a);
      }
      mockCaptions[h] = next.join(', ');
    }
    return HttpResponse.json({ updated: (body.hashes || []).length });
  }),

  http.get('/api/datasets/:id/images/:hash/thumb', () => {
    // 返回 1x1 png 占位即可（mock）
    const png = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==',
      'base64'
    );
    return new HttpResponse(png, { headers: { 'Content-Type': 'image/png' } });
  }),

  http.get('/api/queue/devices', () => HttpResponse.json({ devices: [], max_concurrent: mockQueueSettings.max_concurrent })),

  http.get('/api/queue/settings', () => HttpResponse.json(mockQueueSettings)),

  http.put('/api/queue/settings', async ({ request }) => {
    const body = (await request.json()) as Partial<QueueSettings>;
    mockQueueSettings = { ...mockQueueSettings, ...body };
    return HttpResponse.json(mockQueueSettings);
  }),

  http.get('/api/jobs', ({ request }) => {
    const url = new URL(request.url);
    const projectId = url.searchParams.get('project_id');
    const statuses = url.searchParams.get('status')?.split(',');
    const page = Number(url.searchParams.get('page') || 1);
    const pageSize = Number(url.searchParams.get('page_size') || 50);
    const filtered = mockJobs.filter((j) => (!projectId || j.project_id === projectId) && (!statuses || statuses.includes(j.status)));
    const resp: JobListResponse = { items: filtered.slice((page - 1) * pageSize, page * pageSize), total: filtered.length, page, page_size: pageSize };
    return HttpResponse.json(resp);
  }),

  http.post('/api/jobs', async ({ request }) => {
    const body = (await request.json()) as any;
    const job: Job = {
      id: `j_${Date.now().toString(36)}`,
      type: body.type || 'train',
      name: body.name,
      project_id: body.project_id ?? null,
      status: 'queued',
      priority: body.priority ?? 0,
      scheduled_at: null,
      created_at: Date.now() / 1000,
      started_at: null,
      finished_at: null,
      progress: {},
      latest: {},
      error: null,
      resume_from: null,
      artifact_ids: [],
      run_dir: null,
      pid: null,
      exit_code: null,
    };
    mockJobs.unshift(job);
    return HttpResponse.json(job);
  }),

  http.get('/api/jobs/:id', ({ params }) => {
    const job = mockJobs.find((j) => j.id === params.id) || mockJobs[0];
    return HttpResponse.json(job);
  }),

  http.patch('/api/jobs/:id', async ({ params, request }) => {
    const body = (await request.json()) as any;
    return HttpResponse.json(updateMockJob(String(params.id), body));
  }),

  http.delete('/api/jobs/:id', ({ params }) => {
    removeMockJob(String(params.id));
    return HttpResponse.json({ ok: true });
  }),

  http.post('/api/jobs/:id/pause', ({ params }) => {
    const job = mockJobs.find((j) => j.id === params.id);
    if (job) job.status = 'paused';
    return HttpResponse.json(job);
  }),

  http.post('/api/jobs/:id/resume', ({ params }) => {
    const job = mockJobs.find((j) => j.id === params.id);
    if (job) job.status = 'running';
    return HttpResponse.json(job);
  }),

  http.post('/api/jobs/:id/cancel', ({ params }) => {
    const job = mockJobs.find((j) => j.id === params.id);
    if (job) job.status = 'cancelled';
    return HttpResponse.json(job);
  }),

  http.post('/api/jobs/:id/save', ({ params }) => {
    return HttpResponse.json(mockJobs.find((j) => j.id === params.id));
  }),

  http.post('/api/jobs/:id/retry', ({ params }) => {
    return HttpResponse.json(mockJobs.find((j) => j.id === params.id));
  }),

  http.get('/api/jobs/:id/metrics', () => {
    const steps: number[] = [];
    const loss: number[] = [];
    const loss_ema: number[] = [];
    const grad_norm: number[] = [];
    const vram_mb: number[] = [];
    const it_s: number[] = [];
    const default_lr: number[] = [];
    for (let i = 1; i <= 500; i++) {
      steps.push(i);
      const l = Math.exp(-i / 200) + Math.random() * 0.02 + 0.05;
      loss.push(l);
      loss_ema.push(l * 0.98 + 0.001);
      grad_norm.push(0.5 + Math.random() * 0.2);
      vram_mb.push(16000 + Math.random() * 2000);
      it_s.push(2.1 + (Math.random() - 0.5) * 0.1);
      default_lr.push(0.0001);
    }
    const metrics: JobMetrics = {
      steps, loss, loss_ema, lr: { default: default_lr }, grad_norm, vram_mb, it_s,
      validation: [
        { step: 100, per_t: { '0.1': 0.25, '0.5': 0.18, '0.9': 0.12 }, mean: 0.183 },
        { step: 200, per_t: { '0.1': 0.21, '0.5': 0.15, '0.9': 0.09 }, mean: 0.150 },
      ],
    };
    return HttpResponse.json(metrics);
  }),

  http.get('/api/jobs/:id/samples', () => {
    const samples: JobSample[] = [
      {
        step: 100, prompt_index: 0, prompt: '1girl, anime', seed: 42, loss: null,
        url: 'https://images.unsplash.com/photo-1578632767115-351597cf2477?w=512&auto=format&fit=crop',
        width: 512, height: 512, created_at: 1789000000,
      },
    ];
    return HttpResponse.json(samples);
  }),

  http.get('/api/jobs/:id/checkpoints', () => {
    const checkpoints: JobCheckpoint[] = [
      {
        step: 200, kind: 'weights', path: '/models/checkpoints/chara-v1-step200.safetensors',
        size: 85000000, created_at: 1789000000, artifact_id: 'art_01', ema: false,
      },
    ];
    return HttpResponse.json(checkpoints);
  }),

  http.get('/api/jobs/:id/log', ({ request }) => {
    const url = new URL(request.url);
    const offset = Number(url.searchParams.get('offset') || '0');
    const lines = [
      { ts: 1789000001, level: 'info' as const, msg: 'Initializing trainer...' },
      { ts: 1789000005, level: 'info' as const, msg: 'Starting training loop at epoch 1...' },
      { ts: 1789000010, level: 'warn' as const, msg: 'VRAM usage close to peak threshold.' },
    ];
    const resp: JobLogResponse = { lines: lines.slice(offset), next_offset: lines.length, has_more: false };
    return HttpResponse.json(resp);
  }),

  http.get('/api/jobs/:id/config', () => {
    return HttpResponse.json({
      model: { family: 'toy', dtype: 'fp32' },
      adapter: { algo: 'lokr', rank: 16, alpha: 16 },
      optimizer: { type: 'adamw', lr: 0.0001 },
    });
  }),

  http.post('/api/config/validate', async ({ request }) => {
    const body = (await request.json()) as any;
    if (!body?.config || typeof body.config !== 'object') return HttpResponse.json({ error: { code: 'config.envelope', message: 'Expected {config}' } }, { status: 422 });
    const errors = [];
    if (body.config.adapter?.rank > 1024) {
      errors.push({ loc: 'adapter.rank', msg: 'Rank cannot exceed 1024' });
    }
    return HttpResponse.json({ ok: errors.length === 0, errors, warnings: [] });
  }),

  http.post('/api/plan', () => {
    const plan: Plan = {
      ok: true,
      errors: [],
      warnings: [{ code: 'vram.tight', msg: 'VRAM usage is close to 24GB peak.' }],
      images: 240,
      items: 480,
      captioned: 228,
      steps_per_epoch: 500,
      total_steps: 2000,
      epochs: 4,
      buckets: [{ w: 1024, h: 1024, items: 100, batches: 50 }],
      params: { training_mode: 'adapter', trainable: 14500000, base: 2000000000, adapted_layers: 280, by_algo: { lokr: 280 } },
      memory: {
        weights_mb: 4200, swapped_mb: 0, text_encoder_mb: 0, adapter_mb: 64, optimizer_mb: 128, gradients_mb: 64, heuristic: true,
        estimate_scope: 'per_device', communication_mb_estimate: 0, optimizer_workspace_mb_estimate: 0,
        activations_mb_by_bucket: [{ w: 1024, h: 1024, mb: 6100 }],
        peak_mb_estimate: 18400, gpu_total_mb: 24576,
        suggestions: ['Enable memory.block_swap=4 to reduce peak VRAM.'],
      },
      text_encoding: 'cached',
      eta_estimate_s: 3600,
    };
    return HttpResponse.json(plan);
  }),

  http.get('/api/artifacts', ({ request }) => {
    const url = new URL(request.url);
    const projectId = url.searchParams.get('project_id');
    const items = projectId ? mockArtifacts.filter((a) => a.project_id === projectId) : mockArtifacts;
    return HttpResponse.json(items);
  }),

  http.post('/api/artifacts/:id/convert', async ({ params, request }) => {
    const body = (await request.json()) as { format: string };
    const src = mockArtifacts.find((a) => a.id === params.id) || mockArtifacts[0];
    const converted: Artifact = {
      ...src,
      id: `art_${Date.now().toString(36)}`,
      name: src.name.replace('.safetensors', `-${body.format}.safetensors`),
      created_at: Date.now() / 1000,
    };
    mockArtifacts.push(converted);
    return HttpResponse.json(converted);
  }),

  http.get('/api/presets', () => HttpResponse.json(customPresets)),

  // ---- Models ----
  http.get('/api/families', () => HttpResponse.json(mockFamilies)),

  http.get('/api/families/:name', ({ params }) => {
    const fam = mockFamilies.find((f) => f.name === params.name);
    return fam ? HttpResponse.json(fam) : new HttpResponse(null, { status: 404 });
  }),

  http.get('/api/models/browse-root', ({ request }) => {
    const kind = new URL(request.url).searchParams.get('kind');
    const category = kind === 'vae' ? 'vae' : kind === 'dit' ? 'diffusion_models' : 'text_encoders';
    return HttpResponse.json({path: `/models/${category}`});
  }),
  http.get('/api/models', () => HttpResponse.json(mockModels)),

  http.post('/api/models', async ({ request }) => {
    const body = (await request.json()) as any;
    const asset: ModelAsset = {
      id: `m_${Date.now().toString(36)}`,
      family: body.family,
      kind: body.kind,
      path: body.path,
      size: 0,
      dtype: body.dtype ?? null,
      exists: true,
      is_default: body.is_default ?? false,
      purpose: body.purpose ?? 'training',
      created_at: Date.now() / 1000,
    } as ModelAsset;
    mockModels.push(asset);
    return HttpResponse.json(asset);
  }),

  http.post('/api/models/scan', () => HttpResponse.json({ added: 0 })),

  http.delete('/api/models/:id', ({ params }) => {
    const idx = mockModels.findIndex((m) => m.id === params.id);
    if (idx >= 0) mockModels.splice(idx, 1);
    return HttpResponse.json({ ok: true });
  }),

  // ---- Settings ----
  http.get('/api/settings', () => HttpResponse.json(mockSettings)),

  http.put('/api/settings', async ({ request }) => {
    const body = (await request.json()) as Partial<Settings>;
    mockSettings = {
      paths: { ...mockSettings.paths, ...(body.paths || {}) },
      server: { ...mockSettings.server, ...(body.server || {}) },
      ui: { ...mockSettings.ui, ...(body.ui || {}) },
    };
    return HttpResponse.json(mockSettings);
  }),
];
