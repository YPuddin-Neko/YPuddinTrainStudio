import { http, HttpResponse } from 'msw';
import { SystemStats, Project, Job, Artifact, Plan, Settings } from '../api/types';

export const handlers = [
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

  http.get('/api/jobs', () => {
    const jobs: Job[] = [
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
        artifact_ids: [],
      },
    ];
    return HttpResponse.json(jobs);
  }),

  http.post('/api/plan', () => {
    const plan: Plan = {
      ok: true,
      errors: [],
      warnings: [],
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
        peak_mb_estimate: 12500,
        gpu_total_mb: 24576,
        suggestions: [],
      },
      text_encoding: 'cached',
      eta_estimate_s: 3800,
    };
    return HttpResponse.json(plan);
  }),

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
