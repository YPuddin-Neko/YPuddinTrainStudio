import React from 'react';
import { apiClient } from '../api/client';
import { ApiError } from '../api/types';
import type { DatasetUploadFile } from './datasetFiles';
import { projectUrl } from './projectVersions';
import { createProgressId, type DatasetImportProgressSnapshot } from './useDatasetImportProgress';
import {
  completeUpload, createUploadSession, deleteUploadSession, isNetworkFailure, readUploadSession, sendUploadFiles, sessionEndpoint, waitFor,
  type DatasetUploadOptions, type DatasetUploadProgress, type DatasetUploadResult, type UploadSessionStatus,
} from './uploadDataset';

/** Where an upload goes: a version's training or regularization images, or one existing dataset. */
export interface UploadTarget { projectId: string; versionId?: string; isReg?: boolean; datasetId?: string }
export const uploadTargetKey = ({ projectId, versionId, isReg, datasetId }: UploadTarget) => `${projectId}/${versionId || ''}/${datasetId || (isReg ? 'reg' : 'train')}`;

export type UploadPhase = 'preparing' | 'uploading' | 'importing' | 'waiting' | 'completed' | 'failed' | 'interrupted';
/** Why an upload stopped; pages word the message. */
export type UploadProblem =
  | 'send' // the service answered, but the files could not be read or sent (moved or deleted)
  | 'connection' // the service could not be reached
  | 'unconfirmed' // the import may have finished, but its result could not be read
  | 'expired' // the service no longer has the upload: idle for an hour, or restarted
  | 'mismatch' // files chosen to continue an interrupted upload differ from it
  | 'rejected' // the files themselves cannot be imported
  | 'failed'; // the import failed; the same upload can be imported again
export interface UploadError { problem: UploadProblem; code: string; message: string; error?: unknown; files?: string[] }
export interface UploadEntry {
  id: string;
  key: string;
  target: UploadTarget;
  subject: string;
  link: string;
  phase: UploadPhase;
  sessionId?: string;
  bytesDone: number;
  bytesTotal: number;
  filesDone: number;
  filesTotal: number;
  /** Bytes per second while uploading. */
  rate: number | null;
  /** The service's import phase once every byte has arrived. */
  server?: DatasetImportProgressSnapshot | null;
  /** The busy state an import waits for, such as version.indexing. */
  waiting?: string;
  result?: DatasetUploadResult;
  error?: UploadError;
  /** Continues without choosing the files again. */
  canRetry: boolean;
  /** Needs the same files chosen again: they belong to a page that was closed. */
  needsFiles: boolean;
  /** Running in another tab of this browser. */
  remote: boolean;
  startedAt: number;
  finishedAt?: number;
  /** When the service drops an interrupted upload. */
  expiresAt?: number;
  acknowledged?: boolean;
}
type ManifestFile = { name: string; size: number; modified: number };
interface StoredUpload {
  id: string;
  target: UploadTarget;
  subject: string;
  options: DatasetUploadOptions;
  sessionId: string;
  phase: 'uploading' | 'importing' | 'failed';
  bytesDone: number;
  bytesTotal: number;
  filesDone: number;
  startedAt: number;
  tab: string;
  beat: number;
}
export type ResumeCheck = { ok: true } | { ok: false; missing: string[]; changed: string[] };

const STORAGE_KEY = 'studio.dataset-uploads';
// Each upload's file list is written once under its own key; progress records are rewritten often.
const manifestKey = (id: string) => `${STORAGE_KEY}.files.${id}`;
const BEAT_MS = 4000;
// A tab that has not written its heartbeat for this long has been closed or reloaded.
const LIVE_MS = 15000;
const ACTIVE: UploadPhase[] = ['preparing', 'uploading', 'importing', 'waiting'];
// Busy states that end on their own; the import waits for them instead of failing.
const WAIT_CODES = new Set(['version.busy', 'version.indexing', 'service.restarting']);
// Mirrors server/upload_sessions.py: the files cannot be imported, so the service released them.
const REJECTED_CODES = new Set(['upload.image', 'upload.caption', 'upload.no_images', 'upload.orphan_sidecar', 'upload.duplicate', 'upload.zip_entry', 'upload.too_large', 'upload.too_many', 'upload.path', 'upload.file_type', 'upload.mixed_zip', 'upload.invalid']);
export const isActiveUpload = (entry: UploadEntry) => ACTIVE.includes(entry.phase);

function readStored(): Record<string, StoredUpload> {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    return value?.v === 1 && value.uploads && typeof value.uploads === 'object' ? value.uploads : {};
  } catch { return {}; }
}
function readManifest(id: string): ManifestFile[] | null {
  try {
    const value = JSON.parse(localStorage.getItem(manifestKey(id)) || 'null');
    return Array.isArray(value) ? value : null;
  } catch { return null; }
}
function writeManifest(id: string, manifest: ManifestFile[] | null) {
  try {
    if (manifest) localStorage.setItem(manifestKey(id), JSON.stringify(manifest));
    else localStorage.removeItem(manifestKey(id));
  } catch { /* Without storage an interrupted upload is not offered again. */ }
}
function writeStored(change: (uploads: Record<string, StoredUpload>) => void) {
  try {
    const uploads = readStored();
    change(uploads);
    if (Object.keys(uploads).length) localStorage.setItem(STORAGE_KEY, JSON.stringify({ v: 1, uploads }));
    else localStorage.removeItem(STORAGE_KEY);
  } catch { /* Without storage an interrupted upload is not offered again; uploading still works. */ }
}
/** Survives reloads of the same tab, so a reloaded page treats its own uploads as interrupted at once. */
function tabId() {
  const fresh = () => createProgressId() || `tab-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  try {
    const saved = sessionStorage.getItem(`${STORAGE_KEY}.tab`);
    if (saved) return saved;
    const id = fresh();
    sessionStorage.setItem(`${STORAGE_KEY}.tab`, id);
    return id;
  } catch { return fresh(); }
}
const manifestOf = (files: DatasetUploadFile[]): ManifestFile[] => files.map(({ file, relativePath }) => ({ name: relativePath, size: file.size, modified: file.lastModified }));
export function uploadLink({ projectId, versionId, isReg, datasetId }: UploadTarget) {
  return datasetId ? `/datasets/${encodeURIComponent(datasetId)}` : `${projectUrl(projectId, versionId, 'data')}&data_step=${isReg ? 'reg' : 'datasets'}`;
}
async function serviceReachable() {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 3000);
  try { await apiClient.get('/health', { silent: true, signal: controller.signal }); return true; }
  catch { return false; }
  finally { window.clearTimeout(timer); }
}

/** Dataset uploads run here, outside any page: switching stages or pages keeps them going. */
export class DatasetUploads {
  readonly tab = tabId();
  private entries = new Map<string, UploadEntry>();
  private files = new Map<string, DatasetUploadFile[]>();
  private stored = new Map<string, Omit<StoredUpload, 'phase' | 'bytesDone' | 'filesDone' | 'tab' | 'beat'> & { manifest: ManifestFile[] }>();
  private runs = new Map<string, AbortController>();
  private polls = new Map<string, number>();
  private listeners = new Set<() => void>();
  private snapshot: UploadEntry[] = [];
  private loaded = false;
  private flush: number | undefined;
  private saved = new Map<string, number>();
  private beat: number | undefined;
  private guarded = false;

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    this.load();
    return () => { this.listeners.delete(listener); };
  };
  getSnapshot = () => this.snapshot;
  get(id: string) { return this.entries.get(id); }

  /** Upload chosen files; returns the upload's id, which is also its import progress id. */
  start(target: UploadTarget, files: DatasetUploadFile[], options: DatasetUploadOptions, subject: string): string {
    this.load();
    const key = uploadTargetKey(target);
    // A finished upload for the same place makes way for the new one.
    for (const entry of this.entries.values()) if (entry.key === key && ['completed', 'failed'].includes(entry.phase) && !entry.sessionId) this.entries.delete(entry.id);
    const id = createProgressId() || `upload-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    // The upload's own id becomes its progress id when the session is created.
    const rest = { ...options };
    delete rest.progress_id;
    this.files.set(id, files);
    this.stored.set(id, { id, target, subject, options: rest, manifest: manifestOf(files), sessionId: '', bytesTotal: files.reduce((sum, { file }) => sum + file.size, 0), startedAt: Date.now() });
    this.entries.set(id, {
      id, key, target, subject, link: uploadLink(target), phase: 'preparing', bytesDone: 0, bytesTotal: this.stored.get(id)!.bytesTotal,
      filesDone: 0, filesTotal: files.length, rate: null, canRetry: false, needsFiles: false, remote: false, startedAt: Date.now(),
    });
    this.emit();
    void this.run(id, 'start');
    return id;
  }

  /** Stop and release the staged files. An import already running on the service cannot be stopped. */
  async cancel(id: string): Promise<boolean> {
    const entry = this.entries.get(id);
    if (!entry || entry.remote || entry.phase === 'importing') return false;
    this.runs.get(id)?.abort();
    if (entry.sessionId) {
      try { await deleteUploadSession(entry.target.projectId, entry.sessionId); }
      catch (error) {
        if (error instanceof ApiError && error.code === 'upload.finalizing') {
          // The last request had already started the import: show how it ends.
          this.update(id, { phase: 'importing' });
          void this.run(id, 'import');
          return false;
        }
      }
    }
    this.forget(id);
    return true;
  }

  /** Continue a stopped upload without choosing its files again. */
  retry(id: string) {
    const entry = this.entries.get(id);
    if (!entry || entry.remote || this.runs.has(id) || !entry.canRetry) return;
    if (!entry.sessionId) { this.restart(id); return; }
    void this.run(id, this.files.has(id) ? 'resume' : 'import');
  }

  /** Continue an interrupted upload with the same files chosen again. */
  resume(id: string, files: DatasetUploadFile[]): ResumeCheck {
    const entry = this.entries.get(id);
    const stored = this.stored.get(id);
    if (!entry || !stored || entry.remote || this.runs.has(id)) return { ok: false, missing: [], changed: [] };
    const chosen = new Map(files.map(item => [item.relativePath, item]));
    const missing: string[] = [], changed: string[] = [];
    const ordered = stored.manifest.flatMap(item => {
      const match = chosen.get(item.name);
      if (!match) missing.push(item.name);
      else if (match.file.size !== item.size || match.file.lastModified !== item.modified) changed.push(item.name);
      return match ? [match] : [];
    });
    if (missing.length || changed.length) {
      this.update(id, { error: { problem: 'mismatch', code: 'upload.mismatch', message: '', files: [...changed, ...missing] } });
      return { ok: false, missing, changed };
    }
    this.files.set(id, ordered);
    if (!entry.sessionId) { this.restart(id); return { ok: true }; }
    void this.run(id, 'resume');
    return { ok: true };
  }

  /** Give up an interrupted or failed upload and release what the service still holds. */
  discard(id: string) {
    const entry = this.entries.get(id);
    if (!entry || entry.remote || this.runs.has(id)) return;
    if (entry.sessionId) void deleteUploadSession(entry.target.projectId, entry.sessionId).catch(() => {});
    this.forget(id);
  }

  /** Remove a finished upload from the lists. */
  dismiss(id: string) {
    const entry = this.entries.get(id);
    if (!entry || isActiveUpload(entry)) return;
    if (entry.phase === 'interrupted' || entry.sessionId) { this.discard(id); return; }
    this.forget(id);
  }

  /** A page showed this upload's result; it is not offered again once that page closes. */
  acknowledge(id: string) {
    const entry = this.entries.get(id);
    if (entry && !entry.acknowledged) this.update(id, { acknowledged: true });
  }

  /** Forget every upload of this page; used when a test or a sign-out starts over. */
  reset() {
    for (const controller of this.runs.values()) controller.abort();
    for (const timer of this.polls.values()) window.clearTimeout(timer);
    this.entries.clear(); this.files.clear(); this.stored.clear(); this.runs.clear(); this.polls.clear(); this.saved.clear();
    window.clearTimeout(this.flush); window.clearInterval(this.beat);
    this.flush = this.beat = undefined;
    if (this.loaded) { window.removeEventListener('storage', this.onStorage); window.removeEventListener('pagehide', this.release); }
    this.loaded = false;
    this.snapshot = [];
    this.guard();
  }

  private load() {
    if (this.loaded || typeof window === 'undefined') return;
    this.loaded = true;
    window.addEventListener('storage', this.onStorage);
    window.addEventListener('pagehide', this.release);
    for (const record of Object.values(readStored())) this.adopt(record);
    this.emit();
  }

  private adopt(record: StoredUpload) {
    if (!record?.id || !record.sessionId || this.entries.has(record.id)) return;
    const manifest = readManifest(record.id);
    if (!manifest) return;
    const { phase, bytesDone, filesDone, tab, beat, ...stored } = record;
    this.stored.set(record.id, { ...stored, manifest });
    const live = tab !== this.tab && Date.now() - beat < LIVE_MS;
    this.entries.set(record.id, {
      id: record.id, key: uploadTargetKey(record.target), target: record.target, subject: record.subject, link: uploadLink(record.target),
      phase: live ? phase === 'importing' ? 'importing' : phase === 'failed' ? 'failed' : 'uploading' : 'interrupted', sessionId: record.sessionId,
      bytesDone, bytesTotal: record.bytesTotal, filesDone, filesTotal: manifest.length, rate: null,
      canRetry: false, needsFiles: !live, remote: live, startedAt: record.startedAt,
    });
    if (!live) void this.check(record.id, phase);
    this.heartbeat();
  }

  /** Learn how an upload left by a closed page stands on the service. */
  private async check(id: string, phase: StoredUpload['phase']) {
    const entry = this.entries.get(id);
    if (!entry?.sessionId) return;
    let status: UploadSessionStatus;
    try { status = await readUploadSession(entry.target.projectId, entry.sessionId); }
    catch (error) {
      if (!(error instanceof ApiError) || error.status !== 404 || this.entries.get(id)?.phase !== 'interrupted') return;
      this.dropRecord(id);
      // Without its record the service cannot say whether an import that had started finished.
      this.update(id, { phase: 'failed', sessionId: undefined, canRetry: false, needsFiles: phase !== 'importing', error: { problem: phase === 'importing' ? 'unconfirmed' : 'expired', code: error.code, message: error.message, error } });
      return;
    }
    if (this.entries.get(id)?.phase === 'interrupted') this.apply(id, status);
  }

  private apply(id: string, status: UploadSessionStatus) {
    const entry = this.entries.get(id)!;
    if (status.state === 'completed' && status.result) { this.completed(id, status.result as DatasetUploadResult); return; }
    if (status.state === 'finalizing') { this.update(id, { phase: 'importing', needsFiles: false }); void this.run(id, 'import'); return; }
    if (status.state === 'failed') {
      this.dropRecord(id);
      this.update(id, { phase: 'failed', canRetry: false, needsFiles: false, error: { problem: 'rejected', code: status.error?.code || 'upload.failed', message: status.error?.message || '' } });
      return;
    }
    const received = status.received.reduce((sum, value) => sum + value, 0);
    const manifest = this.stored.get(id)?.manifest || [];
    const ready = status.state === 'ready';
    this.update(id, {
      phase: 'interrupted', bytesDone: received, filesDone: manifest.filter((item, index) => status.received[index] >= item.size).length,
      canRetry: ready, needsFiles: !ready, expiresAt: Date.now() + status.expires_in * 1000,
      error: status.error ? { problem: 'failed', code: status.error.code, message: status.error.message } : entry.error,
    });
  }

  private async run(id: string, mode: 'start' | 'resume' | 'import') {
    const entry = this.entries.get(id);
    if (!entry || this.runs.has(id)) return;
    const unconfirmed = entry.error?.problem === 'unconfirmed';
    const controller = new AbortController();
    this.runs.set(id, controller);
    const { signal } = controller;
    const { projectId } = entry.target;
    let stage: 'create' | 'send' | 'import' = mode === 'import' ? 'import' : 'create';
    this.update(id, { remote: false, needsFiles: false, canRetry: false, error: undefined, phase: mode === 'import' ? 'importing' : entry.phase === 'preparing' ? 'preparing' : 'uploading' });
    try {
      let sessionId = entry.sessionId;
      if (mode !== 'import') {
        const files = this.files.get(id)!;
        let received = files.map(() => 0);
        let chunkBytes: number;
        if (!sessionId) {
          // Not aborted with the upload: an id the service already issued must still be released.
          const session = await createUploadSession(projectId, files, { ...this.stored.get(id)!.options, progress_id: id }, new AbortController().signal);
          sessionId = session.id;
          chunkBytes = session.chunk_bytes;
          if (signal.aborted) { void deleteUploadSession(projectId, sessionId).catch(() => {}); return; }
          this.update(id, { sessionId });
        } else {
          const status = await readUploadSession(projectId, sessionId, signal);
          if (status.state !== 'receiving' && status.state !== 'ready') { this.runs.delete(id); this.apply(id, status); return; }
          received = status.received;
          chunkBytes = status.chunk_bytes;
        }
        stage = 'send';
        this.update(id, { phase: 'uploading' });
        this.save(id, true);
        await sendUploadFiles({ endpoint: sessionEndpoint(projectId, sessionId), files, received, chunkBytes, signal, verify: mode === 'resume', onProgress: progress => this.progress(id, progress) });
      }
      stage = 'import';
      this.update(id, { phase: 'importing', rate: null, bytesDone: entry.bytesTotal, filesDone: entry.filesTotal });
      this.save(id, true);
      this.poll(id, controller);
      const result = await this.importWhenFree(id, sessionId!, signal);
      this.completed(id, result);
    } catch (error) {
      if (!signal.aborted) await this.failed(id, error, stage, unconfirmed);
    } finally {
      if (this.runs.get(id) === controller) {
        this.runs.delete(id);
        window.clearTimeout(this.polls.get(id));
        this.polls.delete(id);
      }
      this.guard();
    }
  }

  private async importWhenFree(id: string, sessionId: string, signal: AbortSignal): Promise<DatasetUploadResult> {
    const { projectId } = this.entries.get(id)!.target;
    for (let attempt = 1; ; attempt++) {
      try {
        return await completeUpload(projectId, sessionId, signal, () => { if (this.entries.get(id)?.phase !== 'importing') this.update(id, { phase: 'importing', waiting: undefined }); });
      } catch (error) {
        // Another import or an index run holds the version; about ten minutes before giving up.
        if (!(error instanceof ApiError) || !WAIT_CODES.has(error.code) || attempt > 60) throw error;
        this.update(id, { phase: 'waiting', waiting: error.code });
        await waitFor(signal, Math.min(10000, 2000 * attempt));
        this.update(id, { phase: 'importing', waiting: undefined });
      }
    }
  }

  private async failed(id: string, error: unknown, stage: 'create' | 'send' | 'import', unconfirmed = false) {
    const entry = this.entries.get(id);
    if (!entry) return;
    const code = error instanceof ApiError ? error.code : '';
    const message = error instanceof Error ? error.message : String(error);
    const network = isNetworkFailure(error) || error instanceof TypeError;
    let problem: UploadProblem = 'failed';
    let keep = !!entry.sessionId;
    if (code === 'upload.session_not_found') {
      // An import whose result was unknown may have finished before the service forgot it.
      problem = unconfirmed || stage === 'import' ? 'unconfirmed' : 'expired';
      keep = false;
    } else if (code === 'upload.chunk_conflict') problem = 'mismatch';
    else if (code === 'upload.result_unconfirmed') problem = 'unconfirmed';
    else if (network) problem = stage === 'send' && await serviceReachable() ? 'send' : 'connection';
    else if (REJECTED_CODES.has(code) || error instanceof ApiError && (error.status === 404 || error.details?.retryable === false || ['upload.archived', 'upload.finished', 'upload.update_required'].includes(code))) {
      problem = 'rejected';
      keep = false;
    }
    if (!this.entries.has(id)) return;
    if (!keep) {
      if (entry.sessionId && problem !== 'expired') void deleteUploadSession(entry.target.projectId, entry.sessionId).catch(() => {});
      this.dropRecord(id);
    } else this.save(id, true, 'failed');
    const files = this.files.has(id);
    const missing = this.entries.get(id)!.bytesDone < entry.bytesTotal;
    this.update(id, {
      phase: 'failed', waiting: undefined, rate: null, finishedAt: Date.now(), sessionId: keep ? entry.sessionId : undefined,
      canRetry: problem === 'rejected' || problem === 'mismatch' || problem === 'unconfirmed' && !keep ? false : keep ? files || !missing : files,
      needsFiles: problem === 'mismatch' || keep && !files && missing,
      error: { problem, code, message, error },
    });
  }

  private completed(id: string, result: DatasetUploadResult) {
    const entry = this.entries.get(id)!;
    if (entry.sessionId) void deleteUploadSession(entry.target.projectId, entry.sessionId).catch(() => {});
    this.dropRecord(id);
    this.files.delete(id);
    this.stored.delete(id);
    this.update(id, { phase: 'completed', result, sessionId: undefined, waiting: undefined, error: undefined, rate: null, bytesDone: entry.bytesTotal, filesDone: entry.filesTotal, finishedAt: Date.now(), canRetry: false, needsFiles: false, remote: false });
  }

  /** Start over with a new upload of the same files, when the service no longer has the old one. */
  private restart(id: string) {
    const entry = this.entries.get(id);
    const stored = this.stored.get(id);
    const files = this.files.get(id);
    if (!entry || !stored || !files) return;
    this.forget(id);
    this.start(entry.target, files, stored.options, entry.subject);
  }

  private forget(id: string) {
    this.runs.get(id)?.abort();
    this.dropRecord(id);
    this.entries.delete(id); this.files.delete(id); this.stored.delete(id); this.saved.delete(id);
    this.emit();
  }

  private progress(id: string, progress: DatasetUploadProgress) {
    const entry = this.entries.get(id);
    if (!entry) return;
    this.entries.set(id, { ...entry, bytesDone: progress.bytesDone, filesDone: progress.filesDone, rate: progress.bytesPerSecond ?? entry.rate });
    this.save(id);
    // Byte counts change many times a second; pages redraw at most ten times.
    if (this.flush === undefined) this.flush = window.setTimeout(() => { this.flush = undefined; this.emit(); }, 100);
  }

  /** The service's own import phases, shown once every byte has arrived. */
  private poll(id: string, run: AbortController) {
    const next = async () => {
      const current = this.entries.get(id);
      if (!current || !['importing', 'waiting'].includes(current.phase) || this.runs.get(id) !== run) return;
      try {
        const snapshot = await apiClient.get<DatasetImportProgressSnapshot>(`/projects/${current.target.projectId}/datasets/import-progress/${id}`, { silent: true });
        if (snapshot.id === id && ['importing', 'waiting'].includes(this.entries.get(id)?.phase || '')) this.update(id, { server: snapshot });
      } catch { /* The import's own response decides the result. */ }
      if (this.runs.get(id) === run) this.polls.set(id, window.setTimeout(() => void next(), 1000));
    };
    void next();
  }

  private save(id: string, now = false, phase?: StoredUpload['phase']) {
    const entry = this.entries.get(id);
    const stored = this.stored.get(id);
    if (!entry?.sessionId || !stored || entry.remote) return;
    if (!now && Date.now() - (this.saved.get(id) || 0) < 2000) return;
    if (!this.saved.has(id)) writeManifest(id, stored.manifest);
    this.saved.set(id, Date.now());
    const value: StoredUpload = {
      id, target: stored.target, subject: stored.subject, options: stored.options, bytesTotal: stored.bytesTotal, startedAt: stored.startedAt, sessionId: entry.sessionId, phase: phase || (entry.phase === 'importing' || entry.phase === 'waiting' ? 'importing' : entry.phase === 'failed' ? 'failed' : 'uploading'),
      bytesDone: entry.bytesDone, filesDone: entry.filesDone, tab: this.tab, beat: Date.now(),
    };
    writeStored(uploads => { uploads[id] = value; });
    this.heartbeat();
  }

  private dropRecord(id: string) {
    this.saved.delete(id);
    if (readStored()[id]) writeStored(uploads => { delete uploads[id]; });
    writeManifest(id, null);
  }

  /** Other tabs see this tab's uploads as running while it keeps writing; a closed tab's go quiet. */
  private heartbeat() {
    const needed = [...this.entries.values()].some(entry => entry.remote || !entry.remote && entry.sessionId && isActiveUpload(entry));
    if (needed && this.beat === undefined) {
      this.beat = window.setInterval(() => {
        for (const entry of this.entries.values()) if (!entry.remote && entry.sessionId && isActiveUpload(entry)) this.save(entry.id, true);
        this.onStorage();
      }, BEAT_MS);
    } else if (!needed && this.beat !== undefined) {
      window.clearInterval(this.beat);
      this.beat = undefined;
    }
  }

  /** Leaving the page stops its uploads: other tabs may continue them right away. */
  private release = () => {
    const mine = [...this.entries.values()].filter(entry => !entry.remote && entry.sessionId && isActiveUpload(entry));
    if (mine.length) writeStored(uploads => { for (const entry of mine) if (uploads[entry.id]) uploads[entry.id].beat = 0; });
  };

  private onStorage = (event?: StorageEvent) => {
    if (event && event.key !== STORAGE_KEY && event.key !== null) return;
    const records = readStored();
    let changed = false;
    for (const entry of [...this.entries.values()]) {
      const record = records[entry.id];
      if (entry.remote) {
        if (!record) { this.entries.delete(entry.id); this.stored.delete(entry.id); changed = true; continue; }
        if (Date.now() - record.beat >= LIVE_MS) {
          // Its tab closed: the upload can be continued here.
          this.entries.set(entry.id, { ...entry, remote: false, phase: 'interrupted', needsFiles: true, bytesDone: record.bytesDone, filesDone: record.filesDone });
          void this.check(entry.id, record.phase);
        } else this.entries.set(entry.id, { ...entry, phase: record.phase === 'importing' ? 'importing' : record.phase === 'failed' ? 'failed' : 'uploading', bytesDone: record.bytesDone, filesDone: record.filesDone });
        changed = true;
      } else if (entry.phase === 'interrupted' && record && record.tab !== this.tab && Date.now() - record.beat < LIVE_MS) {
        // Continued in another tab.
        this.entries.set(entry.id, { ...entry, remote: true, needsFiles: false, phase: 'uploading', bytesDone: record.bytesDone, filesDone: record.filesDone });
        changed = true;
      } else if (entry.phase === 'interrupted' && !record) {
        this.entries.delete(entry.id); this.stored.delete(entry.id); changed = true;
      }
    }
    for (const record of Object.values(records)) if (!this.entries.has(record.id)) { this.adopt(record); changed = true; }
    if (changed) this.emit();
    this.heartbeat();
  };

  private update(id: string, changes: Partial<UploadEntry>) {
    const entry = this.entries.get(id);
    if (!entry) return;
    this.entries.set(id, { ...entry, ...changes });
    this.emit();
  }

  private emit() {
    window.clearTimeout(this.flush);
    this.flush = undefined;
    this.snapshot = [...this.entries.values()];
    this.guard();
    this.listeners.forEach(listener => listener());
  }

  private unload = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ''; };

  /** Closing or reloading the tab would stop an upload: the browser asks first. */
  private guard() {
    const sending = [...this.entries.values()].some(entry => !entry.remote && ['preparing', 'uploading', 'waiting'].includes(entry.phase));
    if (sending && !this.guarded) window.addEventListener('beforeunload', this.unload);
    else if (!sending && this.guarded) window.removeEventListener('beforeunload', this.unload);
    this.guarded = sending;
  }
}

export const datasetUploads = new DatasetUploads();

export function useDatasetUploads(): UploadEntry[] {
  return React.useSyncExternalStore(datasetUploads.subscribe, datasetUploads.getSnapshot, datasetUploads.getSnapshot);
}

/** The upload a page shows for its target: running first, then one that needs attention, then the latest result. */
export function useTargetUpload(target: UploadTarget): UploadEntry | undefined {
  const uploads = useDatasetUploads();
  const key = uploadTargetKey(target);
  const own = uploads.filter(entry => entry.key === key);
  return own.find(isActiveUpload) || own.find(entry => entry.phase === 'interrupted' || entry.phase === 'failed') || own.filter(entry => entry.phase === 'completed').sort((a, b) => (b.finishedAt || 0) - (a.finishedAt || 0))[0];
}
