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
  DatasetImagesPage,
  ModelAsset,
} from '../api/types';

let mockQueueSettings: QueueSettings = {
  held: false,
  max_concurrent: 1,
};

let mockSettings: Settings = {
  paths: {
    data_root: '/Volumes/Service/Dev/data',
    cache_dir: '/Volumes/Service/Dev/cache',
    models_dir: '/Volumes/Service/Dev/models',
    output_dir: '/Volumes/Service/Dev/output',
  },
  server: { host: '127.0.0.1', port: 8765 },
  ui: { language: 'zh-CN', theme: 'system' },
};

const mockProjects: Project[] = [
  {
    id: 'proj_01',
    name: 'Anima Anime Style',
    note: 'Fine-tuning with anime illustration dataset',
    created_at: 1789000000,
    updated_at: 1789000000,
    archived: false,
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
    created_at: 1789000000,
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

export const handlers = [
  http.get('/api/system/stats', () => {
    const stats: SystemStats = {
      cpu_pct: 12.5,
      ram: { used_mb: 16384, total_mb: 65536 },
      disks: [{ path: '/data', used_gb: 250, total_gb: 1000 }],
      gpus: [
        { index: 0, name: 'NVIDIA RTX 4090', util_pct: 78, mem_used_mb: 18432, mem_total_mb: 24576, temp_c: 65 },
      ],
    };
    return HttpResponse.json(stats);
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
      id: `p_${Date.now().toString(36)}`,
      name: body.name,
      note: body.note || '',
      created_at: Date.now() / 1000,
      updated_at: Date.now() / 1000,
      archived: false,
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

  http.post('/api/projects/:id/datasets', async ({ request }) => {
    const body = (await request.json()) as any;
    return HttpResponse.json({ ...mockDatasetInfo, source: { ...mockDatasetInfo.source, ...body, id: 'ds_new', project_id: 'proj_01' } });
  }),

  http.get('/api/projects/:id/config', () => {
    return HttpResponse.json({
      model: { family: 'toy', dtype: 'fp32' },
      dataset: { resolutions: [64], batch_size: 2 },
    });
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

  http.get('/api/datasets/:id/images', ({ request }) => {
    const url = new URL(request.url);
    const page = Number(url.searchParams.get('page') || '1');
    const pageSize = Number(url.searchParams.get('page_size') || '60');
    const q = (url.searchParams.get('q') || '').toLowerCase();
    const filtered = q
      ? mockImages.filter((img) => (mockCaptions[img.hash] || '').toLowerCase().includes(q))
      : mockImages;
    const start = (page - 1) * pageSize;
    const items = filtered.slice(start, start + pageSize).map((img) => ({
      ...img,
      caption: mockCaptions[img.hash] ?? img.caption,
    }));
    const resp: DatasetImagesPage = { items, total: filtered.length, page, page_size: pageSize };
    return HttpResponse.json(resp);
  }),

  http.get('/api/datasets/:id/images/:hash/caption', ({ params }) => {
    return HttpResponse.json({ caption: mockCaptions[String(params.hash)] || '' });
  }),

  http.put('/api/datasets/:id/images/:hash/caption', async ({ params, request }) => {
    const body = (await request.json()) as { caption: string };
    mockCaptions[String(params.hash)] = body.caption;
    return HttpResponse.json({ caption: body.caption });
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

  http.get('/api/queue/settings', () => HttpResponse.json(mockQueueSettings)),

  http.put('/api/queue/settings', async ({ request }) => {
    const body = (await request.json()) as Partial<QueueSettings>;
    mockQueueSettings = { ...mockQueueSettings, ...body };
    return HttpResponse.json(mockQueueSettings);
  }),

  http.get('/api/jobs', ({ request }) => {
    const url = new URL(request.url);
    const projectId = url.searchParams.get('project_id');
    const items = projectId ? mockJobs.filter((j) => j.project_id === projectId) : mockJobs;
    const resp: JobListResponse = { items, total: items.length, page: 1, page_size: 50 };
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
        step: 100, prompt_index: 0, prompt: '1girl, anime', seed: 42,
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
        size: 85000000, created_at: 1789000000, artifact_id: 'art_01',
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
    const resp: JobLogResponse = { lines: lines.slice(offset), next_offset: lines.length };
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
    const errors = [];
    if (body?.adapter?.rank > 1024) {
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
      params: { trainable: 14500000, base: 2000000000, adapted_layers: 280, by_algo: { lokr: 280 } },
      memory: {
        weights_mb: 4200, swapped_mb: 0, text_encoder_mb: 0, adapter_mb: 64, optimizer_mb: 128, heuristic: true,
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

  http.get('/api/presets', () => {
    const presets: Preset[] = [
      {
        name: 'toy-smoke', description: 'CPU 玩具模型冒烟测试',
        config: { model: { family: 'toy', dtype: 'fp32' }, dataset: { resolutions: [64], bucket_step: 16, batch_size: 2 }, loop: { epochs: 1, mixed_precision: 'no' } },
        builtin: true, updated_at: null,
      },
    ];
    return HttpResponse.json(presets);
  }),

  // ---- Models ----
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
