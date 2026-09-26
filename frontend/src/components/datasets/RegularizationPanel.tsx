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
type GenerationPlan = {signature:string;sources:{id:string;path:string;name:string}[];top_tags:{tag:string;count:number}[];source_images:number;existing_images:number;missing_captions:number;invalid_captions:number;empty_after_exclusion:number;eligible_images:number;planned_images:number;remaining_images:number;max_batch_images:number;examples:{source_id:string;rel_path:string;prompt:string}[]};
const tagKey = (value:string) => value.replace(/_/g,' ').trim().toLocaleLowerCase().replace(/\s+/g,' ');
type Snapshot = { path: string; images: number; operations: RegularizationTask[] };
type Props = { projectId: string; versionId: string; readOnly?: boolean; onChanged: () => void };
const active = (task: RegularizationTask) => ['queued', 'running', 'cancelling'].includes(task.status);

export default function RegularizationPanel({ projectId, versionId, readOnly = false, onChanged }: Props) {
  const text = useWorkspaceText();
  const location = useLocation();
  const endpoint = `/projects/${projectId}/versions/${versionId}/regularization`;
  const [source, setSource] = useState<Source>('ai');
  const [prompt, setPrompt] = useState('');
  const [promptSource, setPromptSource] = useState<'manual'|'training_tags'>('manual');
  const [sourceRange, setSourceRange] = useState('');
  const [generationScope, setGenerationScope] = useState<'incremental'|'all'>('incremental');
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
    setPrompt(''); setError(''); setSourceRange(''); setPromptSource('manual');
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
  const fromTraining = source === 'ai' && promptSource === 'training_tags';
  const requestBody = {source,prompt:prompt.trim(),count,width,height,steps,cfg,seed,negative,prior_weight:weight,repeats,
    excluded_tags:excluded.split(',').map(value=>value.trim()).filter(Boolean),
    prompt_source:fromTraining?'training_tags':'manual',source_ids:sourceRange?[sourceRange]:[],generation_scope:generationScope};
  const requestKey = JSON.stringify(requestBody);
  const [plannedRequest, setPlannedRequest] = useState(requestKey);
  useEffect(() => { const timer=setTimeout(()=>setPlannedRequest(requestKey),250); return ()=>clearTimeout(timer); },[requestKey]);
  const completedBatches = query.data?.operations.filter(task=>task.status==='completed').map(task=>task.id).join(',') || '';
  const plan = useQuery<GenerationPlan>({queryKey:['regularization-plan',endpoint,plannedRequest,completedBatches],enabled:fromTraining && plannedRequest===requestKey && !current,
    placeholderData:(previous,previousQuery)=>previousQuery?.queryKey[1]===endpoint?previous:undefined,
    refetchOnWindowFocus:false,
    queryFn:({signal})=>apiClient.post<GenerationPlan>(`${endpoint}/plan`,JSON.parse(plannedRequest),{silent:true,signal})});
  const planPending = requestKey !== plannedRequest || plan.isFetching || plan.isPlaceholderData;
  const toggleExcluded = (tag:string) => setExcluded(previous=>{
    const tags=previous.split(',').map(value=>value.trim()).filter(Boolean);
    return (tags.some(value=>tagKey(value)===tagKey(tag))?tags.filter(value=>tagKey(value)!==tagKey(tag)):[...tags,tag]).join(', ');
  });
  const start = async (event: React.FormEvent) => {
    event.preventDefault();
    if (locked || credentialsBlocked || (fromTraining ? planPending || !!plan.error || !plan.data?.planned_images : !prompt.trim())) return;
    setSubmitting(true); setError('');
    try {
      await apiClient.post<RegularizationTask>(endpoint, {...requestBody,...(fromTraining?{plan_signature:plan.data!.signature}:{})}, {silent:true});
      await query.refetch();
    } catch (err) { setError(formatApiError(err)); if (fromTraining && (err as {code?:string})?.code === 'regularization.plan_stale') void plan.refetch(); }
    finally { setSubmitting(false); }
  };
  const cancel = async (id: string) => {
    setSubmitting(true); setError('');
    try { await apiClient.post(`/regularization/${id}/cancel`, {}, {silent:true}); await query.refetch(); }
    catch (err) { setError(formatApiError(err)); }
    finally { setSubmitting(false); }
  };
  const statusName = (status: string) => ({queued:text('等待开始','Queued'),running:text('进行中','Running'),cancelling:text('正在取消','Cancelling'),completed:text('已加入正则集','Added to regularization data'),failed:text('失败','Failed'),cancelled:text('已取消','Cancelled')}[status] || status);
  const logText = (message: string) => {
    const fixed: Record<string, string> = {
      'Preparing a new isolated regularization batch': '正在创建本批正则图',
      'Cancellation requested': '已请求取消，正在停止任务',
      'Cleanup requires attention; existing datasets were preserved': '临时文件未能全部清理，请检查日志；已有数据集已保留',
    };
    if (fixed[message]) return text(fixed[message], message);
    const loading = /^Loading (.+) base model without adapters$/.exec(message);
    if (loading) return text(`正在加载 ${loading[1]} 底模，不加载适配器`, message);
    const publishing = /^Publishing (\d+) verified image\/caption pairs$/.exec(message);
    if (publishing) return text(`正在保存 ${publishing[1]} 组已检查的图片与标签`, message);
    const added = /^Added (\d+) regularization images to this version$/.exec(message);
    if (added) return text(`已将 ${added[1]} 张正则图加入当前版本`, message);
    const searching = /^Searching (danbooru|gelbooru), page (\d+)$/.exec(message);
    if (searching) return text(`正在检索 ${searching[1]} 第 ${searching[2]} 页`, message);
    return message;
  };

  return <section className="regularization-panel" aria-label={text('正则图','Regularization images')}>
    <div className="reg-heading"><div><h3>{text('正则图','Regularization images')}</h3><p>{text('可选，用于保持底模对通用类别的表现。','Optional images that preserve general class knowledge.')}</p></div><span className="reg-count">{text(`已有 ${query.data?.images ?? 0} 张`,`${query.data?.images ?? 0} images`)}</span></div>
    {query.data?.path && <div className="reg-path"><FolderOpen size={14}/><code>{query.data.path}</code></div>}
    {(error || query.error) && <div role="alert" className="reg-error">{error || formatApiError(query.error)}{query.error && <button type="button" className="ui-btn ui-btn-sm" onClick={()=>void query.refetch()}><RefreshCw size={13}/>{text('重试','Retry')}</button>}</div>}
    {query.isPending ? <p role="status" className="reg-note"><Loader2 size={14}/>{text('读取正则集…','Loading regularization data…')}</p> : <>
    <form onSubmit={event=>void start(event)}>
      <div className="reg-form-grid">
        <label>{text('图片来源','Image source')}<StudioSelect aria-label={text('图片来源','Image source')} value={source} disabled={locked}
          onValueChange={value=>setSource(value as Source)}
          options={[{value:'ai',label:text('本地底模生成','Generate with base model')},{value:'danbooru',label:'Danbooru'},{value:'gelbooru',label:'Gelbooru'}]}/></label>
        <label>{fromTraining?text('本批最多生成','Maximum images this batch'):text('图片数量','Image count')}<input type="number" min={1} max={200} step={1} value={count} disabled={locked} onChange={event=>setCount(Number(event.target.value))}/></label>
      </div>
      {source === 'ai' && <label>{text('提示词来源','Prompt source')}<StudioSelect aria-label={text('提示词来源','Prompt source')} value={promptSource} disabled={locked} onValueChange={value=>setPromptSource(value as 'manual'|'training_tags')} options={[{value:'manual',label:text('手动填写类别提示词','Enter class prompts')},{value:'training_tags',label:text('按训练图片标签逐张生成','One prior per training image caption')}]}/></label>}
      {!fromTraining && <>      <label className="reg-prompt">{source === 'ai' ? text('类别提示词','Class prompt') : text('站点检索标签','Site search tags')}
        <textarea rows={2} required maxLength={4000} value={prompt} disabled={locked} onChange={event=>setPrompt(event.target.value)} placeholder={source === 'ai' ? text('例如：a photo of a dog；不要写训练主体的触发词','For example: a photo of a dog; omit the subject’s trigger word') : text('例如：dog solo；用空格分隔站点标签','For example: dog solo; separate site tags with spaces')}/>
      </label>
</>}
      {fromTraining && <div className="reg-training-plan">
        <div className="reg-form-grid"><label>{text('训练目录范围','Training folders')}<StudioSelect aria-label={text('训练目录范围','Training folders')} value={sourceRange} disabled={locked} onValueChange={setSourceRange} options={[{value:'',label:text('全部训练目录','All training folders')},...(plan.data?.sources || []).map(item=>({value:item.id,label:`${item.name} · ${item.path}`,displayLabel:item.name}))]}/></label>
        <label>{text('生成范围','Generation range')}<StudioSelect aria-label={text('生成范围','Generation range')} value={generationScope} disabled={locked} onValueChange={value=>setGenerationScope(value as 'all'|'incremental')} options={[{value:'incremental',label:text('增量：跳过已有来源','Incremental: skip completed sources')},{value:'all',label:text('全部：另建一个批次','All: create another batch')}]}/></label></div>
        <label>{text('排除标签','Excluded tags')}<input value={excluded} disabled={locked} onChange={event=>setExcluded(event.target.value)} placeholder={text('点击下方标签，或用逗号分隔输入','Select tags below or enter comma-separated tags')}/></label>
        <div className="reg-tag-selection"><strong>{text('训练标签频次 · 点击排除','Training tag frequency · select to exclude')}</strong><div>{plan.data?.top_tags.map(item=><button type="button" key={item.tag} aria-label={text(`排除标签：${item.tag}，${item.count} 张图片`,`Exclude tag: ${item.tag}, ${item.count} images`)} aria-pressed={requestBody.excluded_tags.some(tag=>tagKey(tag)===tagKey(item.tag))} disabled={locked} onClick={()=>toggleExcluded(item.tag)}><span>{item.tag}</span><small>{item.count}</small></button>)}</div></div>
        <p className="reg-note">{text('排除词只影响生成提示词；每批最多 200 张。','Exclusions affect generation prompts only. Maximum 200 images per batch.')}</p>
        <p className="reg-note">{generationScope==='incremental'?text('增量会跳过已有结果；调整排除词后重做请选择“全部”。','Incremental runs skip sources already generated. To regenerate after changing exclusions, select All.'):text('另建批次，保留旧结果并跳过重复图片。','Creates a new batch, preserves prior results and skips duplicates.')}</p>
        <div className="reg-plan-preview" aria-label={text('正则生成计划','Regularization generation plan')} role="region" aria-busy={planPending}>
          {planPending && <p role="status"><Loader2 size={14} className="animate-spin"/>{text('正在核对训练标签与已有来源…','Checking captions and completed sources…')}</p>}
          {plan.error && <p role="alert" className="reg-error">{formatApiError(plan.error)} <button type="button" className="ui-btn ui-btn-sm" onClick={()=>void plan.refetch()}>{text('重新预览','Preview again')}</button></p>}
          {plan.data && <><dl><div><dt>{text('范围内图片','Source images')}</dt><dd>{plan.data.source_images}</dd></div><div><dt>{text('已有来源结果','Completed sources')}</dt><dd>{plan.data.existing_images}</dd></div><div><dt>{text('缺少标签','Missing captions')}</dt><dd>{plan.data.missing_captions}</dd></div><div><dt>{text('标签读取失败','Invalid captions')}</dt><dd>{plan.data.invalid_captions}</dd></div><div><dt>{text('排除后无提示词','Empty after exclusions')}</dt><dd>{plan.data.empty_after_exclusion}</dd></div><div><dt>{text('本批待生成','Planned this batch')}</dt><dd>{plan.data.planned_images}</dd></div></dl>
          {!!plan.data.remaining_images && <p>{text(`本批之后还剩 ${plan.data.remaining_images} 张，可继续增量生成。`,`${plan.data.remaining_images} images remain after this batch; continue incrementally.`)}</p>}
          {!!plan.data.examples.length && <details><summary>{text('查看生成提示词示例','Preview generation prompts')}</summary><ul>{plan.data.examples.map((item,index)=><li key={index}><strong>{plan.data!.sources.find(source=>source.id===item.source_id)?.name}/{item.rel_path}</strong><p>{item.prompt}</p></li>)}</ul></details>}</>}
        </div>
      </div>}
      <p className="reg-note">{source === 'ai' ? text('使用本版本底模，不加载 LoRA；请先在训练参数中设置模型。','Uses this version’s base model without LoRA. Select the model in training settings first.') : text('仅收集站点标记为全年龄的图片，保留原标签并去重。','Collects images rated safe by the site, keeps source tags and skips duplicates.')}</p>
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
        <p className="reg-note">{text('正则图不参与自动验证集划分。','Regularization images are excluded from automatic validation splits.')}</p>
      </details>
      {source !== 'ai' && <div className="reg-options"><p className="reg-note" role="status">{credentials.error ? text('无法读取站点密钥状态。','Could not load site-key status.') : credentials.isPending ? <><Loader2 size={13} className="animate-spin reg-note-spinner" aria-hidden="true"/>{text('读取站点密钥状态…','Loading site-key status…')}</> : credentials.data?.[source]?.configured ? text(`${source} 访问密钥已配置`,`${source} access keys configured`) : source === 'gelbooru' ? text('Gelbooru 需要先配置用户 ID 和 API Key。','Configure the Gelbooru user ID and API key first.') : text('Danbooru 尚未配置密钥，将使用匿名访问。','Danbooru keys are not configured; anonymous access will be used.')}</p><Link className="ui-link" state={{ backgroundLocation: location.state?.backgroundLocation ?? location }} to={`/settings/environment?tab=credentials#credentials-${source}`}>{text('管理访问密钥','Manage access keys')}</Link>{credentials.error && <button type="button" className="ui-link ml-3" onClick={()=>void credentials.refetch()}>{text('重试','Retry')}</button>}</div>}
      <div className="reg-actions"><span>{readOnly ? text('当前版本只读','This version is read-only') : text('完成后自动加入正则集。','Added to regularization data when complete.')}</span><button type="submit" className="ui-btn ui-btn-primary" disabled={locked || credentialsBlocked || !!query.error || (fromTraining ? planPending || !!plan.error || !plan.data?.planned_images : !prompt.trim())}>{submitting || current ? <Loader2 size={14} className="animate-spin"/> : <ImagePlus size={14}/>} {source === 'ai' ? text('生成正则图','Generate images') : text('收集正则图','Collect images')}</button></div>
    </form>
    {query.data?.operations.slice(0,3).map(task=><article className="reg-task" key={task.id} aria-label={task.id}>
      <div className="reg-task-heading"><strong>{task.source === 'ai' ? text('底模生成','Base model generation') : task.source}</strong><code>{task.id}</code><span className={task.status === 'failed' ? 'reg-failed' : ''}>{statusName(task.status)}</span>{task.can_cancel && <button type="button" className="ui-btn ui-btn-sm" disabled={submitting || readOnly} onClick={()=>void cancel(task.id)} aria-label={text(`取消 ${task.id}`,`Cancel ${task.id}`)}><X size={14}/>{text('取消','Cancel')}</button>}</div>
      {active(task) && <><progress value={task.done} max={Math.max(task.total,1)} aria-label={text('正则图进度','Regularization progress')}/><p className="reg-note">{task.done} / {task.total}</p></>}
      {task.error && <p role="alert" className="reg-error">{task.error}</p>}
      {task.status === 'completed' && <p className="reg-note">{text(`新增 ${task.images} 张图片`,`Added ${task.images} images`)}{task.duplicates ? text(`，跳过 ${task.duplicates} 张重复图片`,`, skipped ${task.duplicates} duplicates`) : ''}</p>}
      {task.dataset_id && <Link className="ui-link" to={`${projectUrl(projectId,versionId,'data')}&data_step=captions`}>{text('查看图片与标签','View images and captions')}</Link>}
      {!!task.logs.length && <details><summary>{text('查看日志','View logs')}</summary><pre>{task.logs.map(logText).join('\n')}</pre></details>}
    </article>)}
    {!readOnly && !current && <details className="reg-options reg-existing"><summary>{text('导入已有正则图','Import existing regularization images')}</summary><ProjectDataImport projectId={projectId} versionId={versionId} defaultIsReg onImported={()=>{void query.refetch();onChanged();}}/></details>}
    </>}
  </section>;
}
