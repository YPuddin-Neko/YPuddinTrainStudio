import { Job } from '../api/types';

/**
 * mock 任务共享存储：MSW handlers 与 mock SSE 生成器共用同一份引用，
 * 使事件流可以跟随任意 job_id（包括 POST /jobs 新建的任务）。
 * 注意：只允许原地变更（push/splice/字段赋值），不要整体重赋值。
 */
export const mockJobs: Job[] = [
  {
    id: 'job_01',
    type: 'train',
    name: 'chara-v1',
    project_id: 'proj_01',
    status: 'running',
    priority: 10,
    scheduled_at: null,
    created_at: 1789000000,
    started_at: 1789000000,
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
    latest: { loss: 0.085, loss_ema: 0.089, lr: { default: 0.0001 } },
    error: null,
    resume_from: null,
    artifact_ids: ['art_01'],
    run_dir: null,
    pid: null,
    exit_code: null,
  },
  {
    id: 'job_02',
    type: 'cache',
    name: 'dataset-cache',
    project_id: 'proj_01',
    status: 'completed',
    priority: 5,
    scheduled_at: null,
    created_at: 1788996400,
    started_at: 1788996400,
    finished_at: 1788996500,
    progress: {
      phase: 'finalizing',
      step: 100,
      total_steps: 100,
      epoch: 1,
      eta_s: 0,
      it_s: 15.0,
      vram_peak_mb: 4200,
    },
    latest: { loss: 0, loss_ema: 0, lr: {} },
    error: null,
    resume_from: null,
    artifact_ids: [],
    run_dir: null,
    pid: null,
    exit_code: 0,
  },
];

export function findMockJob(id: string): Job | undefined {
  return mockJobs.find((j) => j.id === id);
}

export function updateMockJob(id: string, patch: Partial<Job>): Job | undefined {
  const job = findMockJob(id);
  if (job) Object.assign(job, patch);
  return job;
}

export function removeMockJob(id: string): void {
  const idx = mockJobs.findIndex((j) => j.id === id);
  if (idx >= 0) mockJobs.splice(idx, 1);
}

export function runningMockJobs(): Job[] {
  return mockJobs.filter((j) => j.status === 'running');
}
