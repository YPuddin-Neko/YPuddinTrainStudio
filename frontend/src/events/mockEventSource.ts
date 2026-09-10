import { EVENT_TYPES } from './eventTypes';

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
  let step = 500;
  let loss = 0.42;

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
            index: 0, name: 'NVIDIA RTX 4090',
            util_pct: Math.round(60 + Math.random() * 30),
            mem_used_mb: 16000 + Math.random() * 3000,
            mem_total_mb: 24576, temp_c: 62 + Math.random() * 8,
          },
        ],
      });
    }, 2000)
  );

  // job.step 每 1s（模拟 job_01 训练中）
  timers.push(
    setInterval(() => {
      step += 1;
      loss = Math.max(0.02, loss * 0.998 + (Math.random() - 0.5) * 0.01);
      emit(EVENT_TYPES.JOB_STEP, {
        job_id: 'job_01',
        step,
        epoch: Math.floor(step / 500),
        loss,
        loss_ema: loss * 0.99,
        lr: { default: 0.0001 },
        grad_norm: 0.4 + Math.random() * 0.3,
        it_s: 2.0 + (Math.random() - 0.5) * 0.2,
        vram_mb: 16000 + Math.random() * 2000,
        eta_s: 2000 - step,
      });
    }, 1000)
  );

  // job.validation 每 5s
  timers.push(
    setInterval(() => {
      const mean = 0.2 * Math.exp(-step / 3000) + 0.05;
      emit(EVENT_TYPES.JOB_VALIDATION, {
        job_id: 'job_01',
        step,
        per_t: { '0.1': mean * 1.4, '0.5': mean, '0.9': mean * 0.7 },
        mean,
      });
    }, 5000)
  );

  // job.log 每 2s
  timers.push(
    setInterval(() => {
      emit(EVENT_TYPES.JOB_LOG, {
        job_id: 'job_01',
        lines: [{ ts: Date.now() / 1000, level: 'info', msg: `mock step ${step} loss=${loss.toFixed(4)}` }],
        next_offset: step,
      });
    }, 2000)
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
  return import.meta.env.DEV && import.meta.env.VITE_USE_MOCK !== 'false';
}
