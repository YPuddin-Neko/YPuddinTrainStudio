import React from 'react';
import { Loader2, MemoryStick } from 'lucide-react';
import { apiClient } from '../../api/client';
import { EVENT_TYPES } from '../../events/eventTypes';
import { useEventStream } from '../../events/useEventStream';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import type { KeptModel } from './xyzTypes';

/** Unloads the base models model-test workers keep for the next comparison. */
export default function ReleaseModelsButton() {
  const text = useWorkspaceText();
  const [models, setModels] = React.useState<KeptModel[] | null>(null);
  const [releasing, setReleasing] = React.useState(false);
  const [notice, setNotice] = React.useState<{ tone: 'done' | 'error'; message: string } | null>(null);
  React.useEffect(() => {
    const controller = new AbortController();
    void apiClient.get<{ models: KeptModel[] }>('/xyz/models', { signal: controller.signal, silent: true })
      .then(result => { if (!controller.signal.aborted) setModels(result.models); })
      .catch(() => { /* the next change arrives as an event */ });
    return () => controller.abort();
  }, []);
  useEventStream<{ models: KeptModel[] }>(EVENT_TYPES.XYZ_MODELS, data => setModels(data.models));
  React.useEffect(() => {
    if (notice?.tone !== 'done') return;
    const timer = setTimeout(() => setNotice(null), 4000);
    return () => clearTimeout(timer);
  }, [notice]);
  const idle = (models || []).filter(model => !model.busy);
  const title = idle.length ? text(`卸载 ${idle.map(model => model.label).join('、')}`, `Unload ${idle.map(model => model.label).join(', ')}`)
    : models?.some(model => model.busy) ? text('模型测试正在生成，完成后才能释放', 'A model test is generating; it can be released once it finishes')
      : text('没有已加载的模型', 'No model is loaded');
  const release = async () => {
    setReleasing(true); setNotice(null);
    try {
      const result = await apiClient.post<{ models: KeptModel[] }>('/xyz/models/release', {}, { silent: true });
      setModels(result.models);
      setNotice({ tone: 'done', message: text('已释放显存', 'VRAM released') });
    } catch (failure) { setNotice({ tone: 'error', message: formatApiError(failure) }); }
    finally { setReleasing(false); }
  };
  return <div className="sampling-release">
    {notice && <span className="sampling-release-note" data-tone={notice.tone} role={notice.tone === 'error' ? 'alert' : 'status'}>{notice.message}</span>}
    <button type="button" className="ui-btn" disabled={!idle.length || releasing} title={title} onClick={() => void release()}>
      {releasing ? <Loader2 size={14} className="animate-spin" aria-hidden="true"/> : <MemoryStick size={14} aria-hidden="true"/>}{releasing ? text('释放中…', 'Releasing…') : text('释放显存', 'Free VRAM')}
    </button>
  </div>;
}
