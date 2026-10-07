import type { TtsCheckpoint, TtsSampleBody } from '../../api/tts';

export type SampleAttempt = { sourceJobId: string; checkpoint: TtsCheckpoint; key: string; body: TtsSampleBody; blocked: boolean; message?: string };
const prefix = 'tts-sample-request:v1:';
export const sampleRequestKey = (jobId: string) => `${prefix}${jobId}`;
export function readSampleAttempt(jobId: string): SampleAttempt | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(sampleRequestKey(jobId)) || 'null') as SampleAttempt | null;
    if (!value || value.sourceJobId !== jobId || value.checkpoint.source_job_id !== jobId
      || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value.key)
      || value.body.checkpoint_id !== value.checkpoint.id || value.body.checkpoint_revision !== value.checkpoint.revision
      || typeof value.body.text !== 'string' || !value.body.text.trim() || value.body.text.length > 4000
      || typeof value.body.reference_audio !== 'string' || typeof value.body.reference_text !== 'string'
      || !!value.body.reference_audio.trim() !== !!value.body.reference_text.trim()
      || !Number.isInteger(value.body.seed) || value.body.seed < 0 || value.body.seed >= 4294967296
      || !Number.isFinite(value.body.cfg_value) || value.body.cfg_value < 0 || value.body.cfg_value > 20
      || !Number.isInteger(value.body.inference_timesteps) || value.body.inference_timesteps < 1 || value.body.inference_timesteps > 100
      || !Array.isArray(value.body.gpu_devices) || value.body.gpu_devices.length > 1 || value.body.gpu_devices.some(device => !/^cuda:\d+$/.test(device))) return null;
    return value;
  } catch { return null; }
}
export function pendingSampleAttempts(scope: { jobId: string } | { projectId: string; versionId: string }): SampleAttempt[] {
  try {
    if ('jobId' in scope) { const attempt = readSampleAttempt(scope.jobId); return attempt ? [attempt] : []; }
    return Object.keys(sessionStorage).filter(key => key.startsWith(prefix)).flatMap(key => {
      const attempt = readSampleAttempt(key.slice(prefix.length));
      return attempt?.checkpoint.project_id === scope.projectId && attempt.checkpoint.version_id === scope.versionId ? [attempt] : [];
    });
  } catch { return []; }
}
