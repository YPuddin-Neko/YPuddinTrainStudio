import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useLocation } from 'react-router-dom';
import { FolderOpen, ImagePlus, Loader2, RefreshCw, X } from 'lucide-react';
import { apiClient } from '../../api/client';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { projectUrl } from '../../utils/projectVersions';
import StudioSelect from '../StudioSelect';
import ProjectDataImport from '../../pages/ProjectDetail/ProjectDataImport';
import type { CredentialStates } from '../../pages/Settings/AccessKeys';
import './regularization.css';

type Source = 'ai' | 'danbooru' | 'gelbooru';
export type RegularizationTask = {
  id: string; source: Source; status: string; phase: string; done: number; total: number;
  logs: string[]; error: string | null; created_at: number; can_cancel: boolean;
  dataset_id: string | null; path: string | null; images: number; duplicates: number;
};
type Snapshot = { path: string; images: number; operations: RegularizationTask[] };
type Props = { projectId: string; versionId: string; readOnly?: boolean; onChanged: () => void };
const active = (task: RegularizationTask) => ['queued', 'running', 'cancelling'].includes(task.status);

export default function RegularizationPanel({ projectId, versionId, readOnly = false, onChanged }: Props) {
  const text = useWorkspaceText();
  const location = useLocation();
  const endpoint = `/projects/${projectId}/versions/${versionId}/regularization`;
  const [source, setSource] = useState<Source>('ai');
  const [prompt, setPrompt] = useState('');
  const [count, setCount] = useState(20);
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [steps, setSteps] = useState(25);
  const [cfg, setCfg] = useState(4);
  const [seed, setSeed] = useState(0);
  const [negative, setNegative] = useState('');
  const [weight, setWeight] = useState(1);
  const [repeats, setRepeats] = useState(1);
  const [excluded, setExcluded] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const notified = useRef(new Set<string>());
  const loaded = useRef(false);
  const changed = useRef(onChanged); changed.current = onChanged;
  useEffect(() => {
    loaded.current = false; notified.current.clear();
    setPrompt(''); setError('');
  }, [projectId, versionId]);
  const query = useQuery({ queryKey: ['regularization', projectId, versionId],
    queryFn: () => apiClient.get<Snapshot>(endpoint, {silent:true}),
    refetchInterval: q => q.state.data?.operations.some(active) ? 1200 : false,
  });
  const credentials = useQuery({ queryKey: ['credentials'], enabled: source !== 'ai',
    queryFn: () => apiClient.get<CredentialStates>('/credentials', { silent: true }), staleTime: 0,
  });
  const refreshCredentials = credentials.refetch;
  useEffect(() => {
    const refresh = () => { if (source !== 'ai') void refreshCredentials(); };
    window.addEventListener('credentials.changed', refresh);
    return () => window.removeEventListener('credentials.changed', refresh);
  }, [source, refreshCredentials]);
  useEffect(() => {
    const tasks = query.data?.operations;
    if (!tasks) return;
    const completed = tasks.filter(task => task.status === 'completed');
    if (loaded.current && completed.some(task => !notified.current.has(task.id))) changed.current();
    completed.forEach(task => notified.current.add(task.id));
    loaded.current = true;
  }, [query.data]);
  const current = query.data?.operations.find(active);
  const locked = readOnly || submitting || !!current;
  const credentialsBlocked = source !== 'ai' && (credentials.isPending || !!credentials.error || (source === 'gelbooru' && !credentials.data?.gelbooru.configured));
  const start = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!prompt.trim() || locked || credentialsBlocked) return;
    setSubmitting(true); setError('');
    try {
      await apiClient.post<RegularizationTask>(endpoint, {source,prompt:prompt.trim(),count,width,height,steps,cfg,seed,negative,
        prior_weight:weight,repeats,excluded_tags:excluded.split(',').map(s=>s.trim()).filter(Boolean)}, {silent:true});
      await query.refetch();
    } catch (err) { setError(formatApiError(err)); }
    finally { setSubmitting(false); }
  };
  const cancel = async (id: string) => {
    setSubmitting(true); setError('');
    try { await apiClient.post(`/regularization/${id}/cancel`, {}, {silent:true}); await query.refetch(); }
    catch (err) { setError(formatApiError(err)); }
    finally { setSubmitting(false); }
  };
  const statusName = (status: string) => ({queued:text('等待开始','Queued'),running:text('准备中','Preparing'),cancelling:text('正在取消','Cancelling'),completed:text('已加入正则集','Added to regularization data'),failed:text('失败','Failed'),cancelled:text('已取消','Cancelled')}[status] || status);

  return <section className="regularization-panel" aria-label={text('正则图','Regularization images')}>
    <div className="reg-heading"><div><h3>{text('正则图','Regularization images')}</h3><p>{text('可选的先验保持数据，用于保留底模对通用类别的表现。','Optional prior-preservation images help retain the base model’s general class knowledge.')}</p></div><span className="reg-count">{text(`已有 ${query.data?.images ?? 0} 张`,`${query.data?.images ?? 0} images`)}</span></div>
    {query.data?.path && <div className="reg-path"><FolderOpen size={14}/><code>{query.data.path}</code></div>}
    {(error || query.error) && <div role="alert" className="reg-error">{error || formatApiError(query.error)}{query.error && <button type="button" onClick={()=>void query.refetch()}><RefreshCw size={13}/>{text('重试','Retry')}</button>}</div>}
    {query.isPending ? <p role="status" className="reg-note"><Loader2 size={14}/>{text('读取正则集…','Loading regularization data…')}</p> : <>
    <form onSubmit={event=>void start(event)}>
      <div className="reg-form-grid">
        <label>{text('图片来源','Image source')}<StudioSelect aria-label={text('图片来源','Image source')} value={source} disabled={locked}
          onValueChange={value=>setSource(value as Source)}
          options={[{value:'ai',label:text('本地底模生成','Generate with base model')},{value:'danbooru',label:'Danbooru'},{value:'gelbooru',label:'Gelbooru'}]}/></label>
        <label>{text('图片数量','Image count')}<input type="number" min={1} max={200} step={1} value={count} disabled={locked} onChange={event=>setCount(Number(event.target.value))}/></label>
      </div>
      <label className="reg-prompt">{source === 'ai' ? text('类别提示词','Class prompt') : text('站点检索标签','Site search tags')}
        <textarea rows={2} required maxLength={4000} value={prompt} disabled={locked} onChange={event=>setPrompt(event.target.value)} placeholder={source === 'ai' ? text('例如：a photo of a dog；不要写训练主体的触发词','For example: a photo of a dog; omit the subject’s trigger word') : text('例如：dog solo；用空格分隔站点标签','For example: dog solo; separate site tags with spaces')}/>
      </label>
      <p className="reg-note">{source === 'ai' ? text('使用当前版本的底模权重生成，不加载 LoRA；请先在“模型”步骤设置完整权重。','Uses this version’s base weights without LoRA. Configure the weights in the Model step first.') : text('只收集站点标记为全年龄的图片，保存原站点标签并去重。需遵守来源站点的使用规则。','Collects only images rated safe by the source, saves source tags and skips duplicates. Follow the source site’s usage rules.')}</p>
      <details className="reg-options"><summary>{text('生成与训练选项','Generation and training options')}</summary><div className="reg-form-grid reg-grid-three">
        {source === 'ai' && <>
          <label>{text('宽度','Width')}<input type="number" min={64} max={2048} step={32} value={width} disabled={locked} onChange={event=>setWidth(Number(event.target.value))}/></label>
          <label>{text('高度','Height')}<input type="number" min={64} max={2048} step={32} value={height} disabled={locked} onChange={event=>setHeight(Number(event.target.value))}/></label>
          <label>{text('采样步数','Sampling steps')}<input type="number" min={1} max={100} value={steps} disabled={locked} onChange={event=>setSteps(Number(event.target.value))}/></label>
          <label>CFG<input type="number" min={0} max={20} step={0.5} value={cfg} disabled={locked} onChange={event=>setCfg(Number(event.target.value))}/></label>
          <label>{text('种子','Seed')}<input type="number" min={0} value={seed} disabled={locked} onChange={event=>setSeed(Number(event.target.value))}/></label>
        </>}
        <label>{text('正则损失权重','Prior loss weight')}<input type="number" min={0} max={100} step={0.1} value={weight} disabled={locked} onChange={event=>setWeight(Number(event.target.value))}/></label>
        <label>{text('重复次数','Repeats')}<input type="number" min={1} max={100} value={repeats} disabled={locked} onChange={event=>setRepeats(Number(event.target.value))}/></label>
      </div>
        {source === 'ai' ? <label>{text('负面提示词','Negative prompt')}<input value={negative} disabled={locked} onChange={event=>setNegative(event.target.value)}/></label> : <label>{text('排除标签','Excluded tags')}<input value={excluded} disabled={locked} onChange={event=>setExcluded(event.target.value)} placeholder={text('用逗号分隔','Separate with commas')}/></label>}
        <p className="reg-note">{text('正则图加入训练来源，不参与自动验证集划分；图片数量、重复次数和权重共同影响训练。','These images join training sources and are excluded from automatic validation splits. Count, repeats and weight all affect training.')}</p>
      </details>
      {source !== 'ai' && <div className="reg-options"><p className="reg-note" role="status">{credentials.error ? text('无法读取站点密钥状态。','Could not load site-key status.') : credentials.isPending ? text('读取站点密钥状态…','Loading site-key status…') : credentials.data?.[source]?.configured ? text(`${source} 访问密钥已配置`,`${source} access keys configured`) : source === 'gelbooru' ? text('Gelbooru 需要先配置用户 ID 和 API Key。','Configure the Gelbooru user ID and API key first.') : text('Danbooru 尚未配置密钥，将使用匿名访问。','Danbooru keys are not configured; anonymous access will be used.')}</p><Link className="studio-link" state={{ backgroundLocation: location.state?.backgroundLocation ?? location }} to={`/settings/environment?tab=credentials#credentials-${source}`}>{text('管理访问密钥','Manage access keys')}</Link>{credentials.error && <button type="button" className="ml-3 underline" onClick={()=>void credentials.refetch()}>{text('重试','Retry')}</button>}</div>}
      <div className="reg-actions"><span>{readOnly ? text('当前版本只读','This version is read-only') : text('每次创建独立批次，完成后自动加入训练来源。','Each run creates a separate batch and registers it after completion.')}</span><button type="submit" className="studio-primary" disabled={locked || credentialsBlocked || !prompt.trim() || !!query.error}>{submitting || current ? <Loader2 size={14} className="animate-spin"/> : <ImagePlus size={14}/>} {source === 'ai' ? text('生成正则图','Generate images') : text('收集正则图','Collect images')}</button></div>
    </form>
    {query.data?.operations.slice(0,3).map(task=><article className="reg-task" key={task.id} aria-label={task.id}>
      <div className="reg-task-heading"><strong>{task.source === 'ai' ? text('底模生成','Base model generation') : task.source}</strong><code>{task.id}</code><span className={task.status === 'failed' ? 'reg-failed' : ''}>{statusName(task.status)}</span>{task.can_cancel && <button type="button" disabled={submitting || readOnly} onClick={()=>void cancel(task.id)} aria-label={text(`取消 ${task.id}`,`Cancel ${task.id}`)}><X size={14}/>{text('取消','Cancel')}</button>}</div>
      {active(task) && <><progress value={task.done} max={Math.max(task.total,1)} aria-label={text('正则图准备进度','Regularization progress')}/><p className="reg-note">{task.done} / {task.total}</p></>}
      {task.error && <p role="alert" className="reg-error">{task.error}</p>}
      {task.status === 'completed' && <p className="reg-note">{text(`新增 ${task.images} 张图片`,`Added ${task.images} images`)}{task.duplicates ? text(`，跳过 ${task.duplicates} 张重复图片`,`, skipped ${task.duplicates} duplicates`) : ''}</p>}
      {task.dataset_id && <Link className="studio-link" to={`${projectUrl(projectId,versionId,'data')}&data_step=captions`}>{text('查看图片与标签','View images and captions')}</Link>}
      {!!task.logs.length && <details><summary>{text('查看日志','View logs')}</summary><pre>{task.logs.join('\n')}</pre></details>}
    </article>)}
    {!readOnly && !current && <details className="reg-options reg-existing"><summary>{text('导入已有正则图','Import existing regularization images')}</summary><ProjectDataImport projectId={projectId} versionId={versionId} defaultIsReg onImported={()=>{void query.refetch();onChanged();}}/></details>}
    </>}
  </section>;
}
