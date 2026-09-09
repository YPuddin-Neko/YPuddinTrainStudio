import { http, HttpResponse } from 'msw';
import {
  SystemStats,
  Project,
  Job,
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
} from '../api/types';

let mockQueueSettings: QueueSettings = {
  held: false,
  max_concurrent: 1,
};

let mockJobs: Job[] = [
  {
    id: 'job_01',
    type: 'train',
    name: 'chara-v1',
    project_id: 'proj_01',
    status: 'running',
    priority: 10,
    scheduled_at: null,
    created_at: new Date().toISOString(),
    started_at: new Date().toISOString(),
    finished_at: null,
    progress: {
      phase: 'training',
      step: 450,
      total_steps: 2000,
      epoch: 1,
      eta_s: 3600,
      it_s: 2.1,
      vram_peak_mb: 18400,
    },
    latest: {
      loss: 0.085,
      loss_ema: 0.089,
      lr: { default: 0.0001 },
    },
    error: null,
    resume_from: null,
    artifact_ids: ['art_01'],
  },
  {
    id: 'job_02',
    type: 'cache',
    name: 'dataset-cache',
    project_id: 'proj_01',
    status: 'completed',
    priority: 5,
    scheduled_at: null,
    created_at: new Date(Date.now() - 3600000).toISOString(),
    started_at: new Date(Date.now() - 3600000).toISOString(),
    finished_at: new Date(Date.now() - 3500000).toISOString(),
    progress: {
      phase: 'finalizing',
      step: 100,
      total_steps: 100,
      epoch: 1,
      eta_s: 0,
      it_s: 15.0,
      vram_peak_mb: 4200,
    },
    latest: {
      loss: 0,
      loss_ema: 0,
      lr: {},
    },
    error: null,
    resume_from: null,
    artifact_ids: [],
  },
];

export const handlers = [
  // 1. 系统统计
  http.get('/api/system/stats', () => {
    const stats: SystemStats = {
      cpu_pct: 12.5,
      ram: { used_mb: 16384, total_mb: 65536 },
      disks: [{ path: '/data', used_gb: 250, total_gb: 1000 }],
      gpus: [
        {
          index: 0,
          name: 'NVIDIA RTX 4090',
          util_pct: 78,
          mem_used_mb: 18432,
          mem_total_mb: 24576,
          temp_c: 65,
        },
      ],
    };
    return HttpResponse.json(stats);
  }),

  // 2. 文件系统浏览
  http.get('/api/fs/list', ({ request }) => {
    const url = new URL(request.url);
    const path = url.searchParams.get('path') || '/';
    const resp: FsListResponse = {
      path,
      parent: path === '/' ? null : path.substring(0, path.lastIndexOf('/')) || '/',
      entries: [
        { name: 'datasets', is_dir: true, size: 4096, mtime: '2026-09-01T10:00:00Z' },
        { name: 'models', is_dir: true, size: 4096, mtime: '2026-09-01T10:00:00Z' },
        { name: 'config.toml', is_dir: false, size: 1024, mtime: '2026-09-05T12:00:00Z' },
        { name: 'data_reg.safetensors', is_dir: false, size: 204800, mtime: '2026-09-08T15:00:00Z' },
      ],
    };
    return HttpResponse.json(resp);
  }),

  // 3. 项目
  http.get('/api/projects', () => {
    const projects: Project[] = [
      {
        id: 'proj_01',
        name: 'Anima Anime Style',
        note: 'Fine-tuning with anime illustration dataset',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        archived: false,
        dataset_ids: ['ds_01'],
        stats: { jobs: 2, artifacts: 1 },
      },
    ];
    return HttpResponse.json(projects);
  }),

  http.get('/api/projects/:id', ({ params }) => {
    const project: Project = {
      id: String(params.id),
      name: 'Anima Anime Style',
      note: 'Fine-tuning with anime illustration dataset',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      archived: false,
      dataset_ids: ['ds_01'],
      stats: { jobs: 2, artifacts: 1 },
    };
    return HttpResponse.json(project);
  }),

  // 4. 队列设置与任务列表
  http.get('/api/queue/settings', () => {
    return HttpResponse.json(mockQueueSettings);
  }),

  http.put('/api/queue/settings', async ({ request }) => {
    const body = (await request.json()) as Partial<QueueSettings>;
    mockQueueSettings = { ...mockQueueSettings, ...body };
    return HttpResponse.json(mockQueueSettings);
  }),

  http.get('/api/jobs', () => {
    return HttpResponse.json(mockJobs);
  }),

  http.get('/api/jobs/:id', ({ params }) => {
    const job = mockJobs.find((j) => j.id === params.id) || mockJobs[0];
    return HttpResponse.json(job);
  }),

  http.patch('/api/jobs/:id', async ({ params, request }) => {
    const body = (await request.json()) as any;
    mockJobs = mockJobs.map((j) => (j.id === params.id ? { ...j, ...body } : j));
    const updated = mockJobs.find((j) => j.id === params.id);
    return HttpResponse.json(updated);
  }),

  http.delete('/api/jobs/:id', ({ params }) => {
    mockJobs = mockJobs.filter((j) => j.id !== params.id);
    return HttpResponse.json({ ok: true });
  }),

  // 任务动作 (pause, resume, cancel, save, retry)
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
    const job = mockJobs.find((j) => j.id === params.id);
    return HttpResponse.json(job);
  }),

  http.post('/api/jobs/:id/retry', ({ params }) => {
    const job = mockJobs.find((j) => j.id === params.id);
    return HttpResponse.json(job);
  }),

  // 5. 任务指标与监控
  http.get('/api/jobs/:id/metrics', () => {
    const steps: number[] = [];
    const loss: number[] = [];
    const loss_ema: number[] = [];
    const grad_norm: number[] = [];
    const vram_mb: number[] = [];
    const it_s: number[] = [];
    const default_lr: number[] = [];

    // 生成 500 个模拟数据点
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
      steps,
      loss,
      loss_ema,
      lr: { default: default_lr },
      grad_norm,
      vram_mb,
      it_s,
      validation: [
        { step: 100, per_t: { '0.1': 0.25, '0.5': 0.18, '0.9': 0.12 }, mean: 0.183 },
        { step: 200, per_t: { '0.1': 0.21, '0.5': 0.15, '0.9': 0.09 }, mean: 0.150 },
        { step: 300, per_t: { '0.1': 0.19, '0.5': 0.13, '0.9': 0.08 }, mean: 0.133 },
        { step: 400, per_t: { '0.1': 0.17, '0.5': 0.11, '0.9': 0.07 }, mean: 0.116 },
      ],
    };
    return HttpResponse.json(metrics);
  }),

  http.get('/api/jobs/:id/samples', () => {
    const samples: JobSample[] = [
      {
        step: 100,
        prompt_index: 0,
        prompt: '1girl, beautiful anime portrait, sunny day',
        seed: 42,
        url: 'https://images.unsplash.com/photo-1578632767115-351597cf2477?w=512&auto=format&fit=crop',
        width: 512,
        height: 512,
        created_at: new Date().toISOString(),
      },
      {
        step: 200,
        prompt_index: 0,
        prompt: '1girl, beautiful anime portrait, sunny day',
        seed: 42,
        url: 'https://images.unsplash.com/photo-1534447677768-be436bb09401?w=512&auto=format&fit=crop',
        width: 512,
        height: 512,
        created_at: new Date().toISOString(),
      },
      {
        step: 300,
        prompt_index: 0,
        prompt: '1girl, beautiful anime portrait, sunny day',
        seed: 42,
        url: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=512&auto=format&fit=crop',
        width: 512,
        height: 512,
        created_at: new Date().toISOString(),
      },
    ];
    return HttpResponse.json(samples);
  }),

  http.get('/api/jobs/:id/checkpoints', () => {
    const checkpoints: JobCheckpoint[] = [
      {
        step: 200,
        kind: 'weights',
        path: '/models/checkpoints/chara-v1-step200.safetensors',
        size: 85000000,
        created_at: new Date().toISOString(),
        artifact_id: 'art_01',
      },
      {
        step: 400,
        kind: 'full',
        path: '/models/checkpoints/chara-v1-step400.pt',
        size: 350000000,
        created_at: new Date().toISOString(),
      },
    ];
    return HttpResponse.json(checkpoints);
  }),

  http.get('/api/jobs/:id/log', ({ request }) => {
    const url = new URL(request.url);
    const offset = Number(url.searchParams.get('offset') || '0');
    const lines = [
      { ts: '12:00:01', level: 'info' as const, msg: 'Initializing trainer and loading model weights...' },
      { ts: '12:00:03', level: 'info' as const, msg: 'Discovered 1500 cached latents.' },
      { ts: '12:00:05', level: 'info' as const, msg: 'Starting training loop at epoch 1...' },
      { ts: '12:00:10', level: 'info' as const, msg: 'Step 100/2000 - loss: 0.142, lr: 0.000100' },
      { ts: '12:00:15', level: 'warn' as const, msg: 'VRAM usage close to peak threshold (18.4 GB).' },
      { ts: '12:00:20', level: 'info' as const, msg: 'Step 200/2000 - checkpoint saved.' },
    ];
    const resp: JobLogResponse = {
      lines: lines.slice(offset),
      next_offset: lines.length,
    };
    return HttpResponse.json(resp);
  }),

  http.get('/api/jobs/:id/config', () => {
    return HttpResponse.json({
      model: { name: 'Anima-Default', dtype: 'bf16' },
      adapter: { algo: 'lokr', rank: 16, alpha: 16 },
      optimizer: { type: 'adamw8bit', lr: 0.0001 },
    });
  }),

  // 6. 配置规划与校验
  http.post('/api/config/validate', async ({ request }) => {
    const body = (await request.json()) as any;
    const errors = [];
    const warnings = [];
    if (body?.adapter?.rank > 1024) {
      errors.push({ loc: 'adapter.rank', msg: 'Rank cannot exceed 1024' });
    }
    if (body?.memory?.fp8_base === false && body?.memory?.block_swap === 0) {
      warnings.push({ code: 'vram.tight', msg: 'Training with fp8=false and block_swap=0 may require >24GB VRAM.' });
    }
    return HttpResponse.json({
      ok: errors.length === 0,
      errors,
      warnings,
    });
  }),

  http.post('/api/plan', () => {
    const plan: Plan = {
      ok: true,
      errors: [],
      warnings: [{ code: 'vram.tight', msg: 'VRAM usage is close to 24GB peak.' }],
      steps_per_epoch: 500,
      total_steps: 2000,
      epochs: 4,
      buckets: [{ w: 1024, h: 1024, images: 100, batches: 50 }],
      params: { trainable: 14500000, base: 2000000000 },
      memory: {
        weights_mb: 4200,
        adapter_mb: 64,
        optimizer_mb: 128,
        activations_mb_by_bucket: [{ w: 1024, h: 1024, mb: 6100 }],
        peak_mb_estimate: 18400,
        gpu_total_mb: 24576,
        suggestions: ['Enable memory.block_swap=4 to reduce peak VRAM.'],
      },
      text_encoding: 'cached',
      eta_estimate_s: 3600,
    };
    return HttpResponse.json(plan);
  }),

  // 7. 产物与预设
  http.get('/api/artifacts', () => {
    const artifacts: Artifact[] = [
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
        created_at: new Date().toISOString(),
        metadata: {},
      },
    ];
    return HttpResponse.json(artifacts);
  }),

  http.get('/api/presets', () => {
    const presets: Preset[] = [
      {
        name: 'Anima-LoKr-Default',
        description: 'Default preset for Anima LoKr adapter',
        config: { adapter: { algo: 'lokr', rank: 16, factor: -1 } },
        builtin: true,
        updated_at: new Date().toISOString(),
      },
      {
        name: 'Anima-LoRA-Standard',
        description: 'Standard LoRA config (rank 32, alpha 32)',
        config: { adapter: { algo: 'lora', rank: 32, alpha: 32 } },
        builtin: true,
        updated_at: new Date().toISOString(),
      },
    ];
    return HttpResponse.json(presets);
  }),

  http.get('/api/settings', () => {
    const settings: Settings = {
      paths: {
        data_root: '/Volumes/Service/Dev/data',
        cache_dir: '/Volumes/Service/Dev/cache',
        models_dir: '/Volumes/Service/Dev/models',
        output_dir: '/Volumes/Service/Dev/output',
      },
      server: {
        host: '127.0.0.1',
        port: 8765,
      },
      ui: {
        language: 'zh-CN',
        theme: 'system',
      },
    };
    return HttpResponse.json(settings);
  }),
];
