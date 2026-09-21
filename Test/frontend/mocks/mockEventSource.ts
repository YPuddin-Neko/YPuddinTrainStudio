import { EVENT_TYPES } from '../../../frontend/src/events/eventTypes';
import { runningMockJobs, findMockJob } from './mockStore';

/**
 * 开发态 mock SSE 事件源。
 * MSW 无法拦截 EventSource，因此在 mock 模式下（DEV 且未显式关闭 MSW）
 * 用本生成器模拟持续事件流：system.stats / job.step / job.validation /
 * job.sample_progress / job.log 自增推送，让页面在纯前端环境看到"活的"曲线。
 */

type RawListener = (ev: { data: string; lastEventId: string }) => void;

export function createMockEventSource() {
  const listeners = new Map<string, Set<RawListener>>();
  const timers: ReturnType<typeof setInterval>[] = [];
  let seq = 0;
  // 每个 job 独立的步进状态，支持任意 job_id（含 POST /jobs 新建的）
  const perJob = new Map<string, { step: number; loss: number }>();
  const stateOf = (jobId: string) => {
    if (!perJob.has(jobId)) perJob.set(jobId, { step: findMockJob(jobId)?.progress?.step || 0, loss: 0.42 });
    return perJob.get(jobId)!;
  };

  const emit = (type: string, data: any) => {
    const set = listeners.get(type);
    if (!set || set.size === 0) return;
    const ev = { data: JSON.stringify(data), lastEventId: String(++seq) };
    set.forEach((cb) => cb(ev));
  };

  // system.stats 每 2s
  timers.push(
    setInterval(() => {
      emit(EVENT_TYPES.SYSTEM_STATS, {
        cpu_pct: 10 + Math.random() * 30,
        ram: { used_mb: 16384 + Math.random() * 2048, total_mb: 65536 },
        disks: [{ path: '/data', used_gb: 250, total_gb: 1000 }],
        gpus: [
          {
            index: 0, kind: 'cuda', name: 'NVIDIA RTX 4090',
            util_pct: Math.round(60 + Math.random() * 30),
            mem_used_mb: 16000 + Math.random() * 3000,
            mem_total_mb: 24576, temp_c: 62 + Math.random() * 8,
          },
        ],
      });
    }, 2000)
  );

  // job.step 每 1s（遍历所有 running 状态的 mock 任务，支持任意 job_id）
  timers.push(
    setInterval(() => {
      for (const job of runningMockJobs()) {
        const st = stateOf(job.id);
        st.step += 1;
        st.loss = Math.max(0.02, st.loss * 0.998 + (Math.random() - 0.5) * 0.01);
        const total = job.progress?.total_steps || 2000;
        job.progress = { ...job.progress, step: st.step, epoch: Math.floor(st.step / 500) };
        job.latest = { ...job.latest, loss: st.loss, loss_ema: st.loss * 0.99 };
        emit(EVENT_TYPES.JOB_STEP, {
          job_id: job.id,
          step: st.step,
          epoch: Math.floor(st.step / 500),
          loss: st.loss,
          loss_ema: st.loss * 0.99,
          lr: { default: 0.0001 },
          grad_norm: 0.4 + Math.random() * 0.3,
          it_s: 2.0 + (Math.random() - 0.5) * 0.2,
          vram_mb: 16000 + Math.random() * 2000,
          eta_s: Math.max(0, total - st.step),
        });
      }
    }, 1000)
  );

  // job.validation 每 5s
  timers.push(
    setInterval(() => {
      for (const job of runningMockJobs()) {
        const st = stateOf(job.id);
        const mean = 0.2 * Math.exp(-st.step / 3000) + 0.05;
        emit(EVENT_TYPES.JOB_VALIDATION, {
          job_id: job.id,
          step: st.step,
          per_t: { '0.1': mean * 1.4, '0.5': mean, '0.9': mean * 0.7 },
          mean,
        });
      }
    }, 5000)
  );

  // job.log 每 2s
  timers.push(
    setInterval(() => {
      for (const job of runningMockJobs()) {
        const st = stateOf(job.id);
        emit(EVENT_TYPES.JOB_LOG, {
          job_id: job.id,
          lines: [{ ts: Date.now() / 1000, level: 'info', msg: `mock step ${st.step} loss=${st.loss.toFixed(4)}` }],
          next_offset: st.step,
        });
      }
    }, 2000)
  );

  // job.sample_progress 每 800ms：2 张 prompt × 20 步循环自增，走完一轮歇 3s 再开始（对每个 running 任务）
  let spDone = 0;
  let spPromptIndex = 0;
  let spPause = 0;
  const SP_TOTAL = 20;
  const SP_PROMPTS = 2;
  timers.push(
    setInterval(() => {
      if (spPause > 0) {
        spPause -= 1;
        return;
      }
      for (const job of runningMockJobs()) {
        const st = stateOf(job.id);
        emit(EVENT_TYPES.JOB_SAMPLE_PROGRESS, {
          job_id: job.id,
          step: st.step,
          prompt_index: spPromptIndex,
          prompts: SP_PROMPTS,
          done: spDone,
          total: SP_TOTAL,
        });
      }
      spDone += 1;
      if (spDone > SP_TOTAL) {
        spDone = 0;
        spPromptIndex += 1;
        if (spPromptIndex >= SP_PROMPTS) {
          spPromptIndex = 0;
          spPause = 4; // 一轮预览结束，停顿后再循环
        }
      }
    }, 800)
  );

  return {
    addEventListener(type: string, cb: RawListener) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)!.add(cb);
    },
    close() {
      timers.forEach(clearInterval);
      listeners.clear();
    },
    // EventSource 兼容字段
    set onopen(_: any) {},
    set onerror(_: any) {},
  };
}

export function shouldUseMockEvents(): boolean {
  return import.meta.env.DEV && import.meta.env.VITE_USE_MOCK === 'true';
}
