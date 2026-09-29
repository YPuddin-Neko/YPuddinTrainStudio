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

type Source = 'ai' | 'danbooru' | 'gelbooru' | 'e621' | 'rule34';
export type RegularizationTask = {
  id: string; source: Source; status: string; phase: string; done: number; total: number;
  logs: string[]; error: string | null; created_at: number; can_cancel: boolean;
  dataset_id: string | null; path: string | null; images: number; duplicates: number;
};
type MatchPlan = {source:string;sources:{id:string;path:string;name:string}[];source_images:number;captioned_images:number;missing_captions:number;invalid_captions:number;top_tags:{tag:string;count:number}[];search_tags:{tag:string;share:number}[];searchable_tags:number;unsearchable_tags:string[];aspect:{low:number;median:number;high:number};size:{width:number;height:number};existing_images:number;suggested_count:number;tag_limit:number|null;tag_limit_known:boolean};
type Estimate = {source:string;count:number|null;terms:string[];local_exclusions:string[];tag_limit:number|null};
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
  // Sites follow the training tags by default; a manual search is the other way.
  const [siteMode, setSiteMode] = useState<'manual'|'training_tags'>('training_tags');
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
  const credentialsBlocked = source !== 'ai' && (credentials.isPending || !!credentials.error || (['gelbooru', 'rule34'].includes(source) && !credentials.data?.[source]?.configured));
  const fromTraining = source === 'ai' && promptSource === 'training_tags';
  const matching = source !== 'ai' && siteMode === 'training_tags';
  const manualSite = source !== 'ai' && siteMode === 'manual';
  const requestBody = {source,prompt:prompt.trim(),count,width,height,steps,cfg,seed,negative,prior_weight:weight,repeats,
    excluded_tags:excluded.split(',').map(value=>value.trim()).filter(Boolean),
    prompt_source:fromTraining || matching?'training_tags':'manual',source_ids:sourceRange?[sourceRange]:[],generation_scope:generationScope};
  const requestKey = JSON.stringify(requestBody);
  const [plannedRequest, setPlannedRequest] = useState(requestKey);
  useEffect(() => { const timer=setTimeout(()=>setPlannedRequest(requestKey),250); return ()=>clearTimeout(timer); },[requestKey]);
  const completedBatches = query.data?.operations.filter(task=>task.status==='completed').map(task=>task.id).join(',') || '';
  const plan = useQuery<GenerationPlan>({queryKey:['regularization-plan',endpoint,plannedRequest,completedBatches],enabled:fromTraining && plannedRequest===requestKey && !current,
    placeholderData:(previous,previousQuery)=>previousQuery?.queryKey[1]===endpoint?previous:undefined,
    refetchOnWindowFocus:false,
    queryFn:({signal})=>apiClient.post<GenerationPlan>(`${endpoint}/plan`,JSON.parse(plannedRequest),{silent:true,signal})});
  const planPending = requestKey !== plannedRequest || plan.isFetching || plan.isPlaceholderData;
  // What following the training tags would search for; the count does not change it.
  const matchKey = JSON.stringify({source,prompt_source:'training_tags',source_ids:requestBody.source_ids,excluded_tags:requestBody.excluded_tags});
  const [plannedMatch, setPlannedMatch] = useState(matchKey);
  useEffect(() => { const timer=setTimeout(()=>setPlannedMatch(matchKey),250); return ()=>clearTimeout(timer); },[matchKey]);
  const match = useQuery<MatchPlan>({queryKey:['regularization-match',endpoint,plannedMatch,completedBatches],enabled:matching && plannedMatch===matchKey && !current,
    placeholderData:(previous,previousQuery)=>previousQuery?.queryKey[1]===endpoint?previous:undefined, refetchOnWindowFocus:false,
    queryFn:({signal})=>apiClient.post<MatchPlan>(`${endpoint}/match`,JSON.parse(plannedMatch),{silent:true,signal})});
  const matchPending = matchKey !== plannedMatch || match.isFetching || match.isPlaceholderData;
  const matchReady = !!match.data && !matchPending && !match.error && match.data.captioned_images > 0 && match.data.searchable_tags > 0;
  // How many posts a manual search finds on the site, asked a moment after typing stops.
  const estimateKey = JSON.stringify({source,prompt:requestBody.prompt,excluded_tags:requestBody.excluded_tags});
  const [plannedEstimate, setPlannedEstimate] = useState(estimateKey);
  useEffect(() => { const timer=setTimeout(()=>setPlannedEstimate(estimateKey),600); return ()=>clearTimeout(timer); },[estimateKey]);
  const estimate = useQuery<Estimate>({queryKey:['regularization-estimate',endpoint,plannedEstimate],enabled:manualSite && !!requestBody.prompt && plannedEstimate===estimateKey && !credentialsBlocked,
    retry:false, refetchOnWindowFocus:false, staleTime:60_000,
    queryFn:({signal})=>apiClient.post<Estimate>(`${endpoint}/estimate`,JSON.parse(plannedEstimate),{silent:true,signal})});
  const estimatePending = estimateKey !== plannedEstimate || estimate.isFetching;
  const percent = (share:number) => `${Math.round(share*100)}%`;
  const ratio = (value:number) => Number(value.toFixed(2)).toString();
  const toggleExcluded = (tag:string) => setExcluded(previous=>{
    const tags=previous.split(',').map(value=>value.trim()).filter(Boolean);
    return (tags.some(value=>tagKey(value)===tagKey(tag))?tags.filter(value=>tagKey(value)!==tagKey(tag)):[...tags,tag]).join(', ');
  });
  const start = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!ready) return;
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
  const ready = !locked && !credentialsBlocked && !query.error && (fromTraining ? !planPending && !plan.error && !!plan.data?.planned_images : matching ? matchReady : !!prompt.trim());
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
    const searching = /^Searching (danbooru|gelbooru|e621|rule34), page (\d+)$/.exec(message);
    if (searching) return text(`正在检索 ${searching[1]} 第 ${searching[2]} 页`, message);
    const matched = /^Searching (danbooru|gelbooru|e621|rule34) for (.+) \(page (\d+)\)$/.exec(message);
    if (matched) return text(`正在检索 ${matched[1]}：${matched[2]}（第 ${matched[3]} 页）`, message);
    const found = /^Found (\d+) of (\d+) matching images; adding them$/.exec(message);
    if (found) return text(`找到 ${found[1]} 张符合条件的图片（目标 ${found[2]} 张），已全部加入`, message);
    const skipped = /^Skipped (\d+) files that were not usable images$/.exec(message);
    if (skipped) return text(`跳过 ${skipped[1]} 个无法使用的文件`, message);
    const local = /^Checking (\d+) excluded tags locally beyond the site's tag limit$/.exec(message);
    if (local) return text(`有 ${local[1]} 个排除标签超出站点的标签上限，下载时逐张检查`, message);
    const slow = /^(Danbooru|Gelbooru|e621|Rule34) asked to slow down; retrying in (\d+) s$/.exec(message);
    if (slow) return text(`${slow[1]} 要求放慢请求，${slow[2]} 秒后重试`, message);
    const busy = /^(Danbooru|Gelbooru|e621|Rule34) is busy \(HTTP (\d+)\); retrying in (\d+) s$/.exec(message);
    if (busy) return text(`${busy[1]} 暂时繁忙（HTTP ${busy[2]}），${busy[3]} 秒后重试`, message);
    if (message === 'Reading the training captions and image sizes') return text('正在读取训练标签和图片尺寸', message);
    return message;
  };
  const errorText = (message: string) => {
    const fixed: Record<string, string> = {
      'No new matching safe images were found. Change the search tags or excluded tags; no batch was added.': '没有找到新的符合条件的全年龄图片，请调整检索标签或排除标签；未添加任何图片。',
      'No new non-duplicate images were produced': '新图片都与已有图片重复，未添加任何图片。',
      'The training captions have no searchable tags left; check the captions and excluded tags': '训练标签里没有可检索的站点标签，请检查标签或排除设置。',
      'Regularization download exceeded its byte limit': '下载量超过本批上限，未添加任何图片。',
    };
    if (fixed[message]) return text(fixed[message], message);
    const http = /^(danbooru|gelbooru|e621|rule34) returned HTTP (\d+); check site credentials or retry later$/.exec(message);
    if (http) return text(`${http[1]} 返回 HTTP ${http[2]}，请检查访问密钥或稍后重试。`, message);
    const offline = /^Could not read (danbooru|gelbooru|e621|rule34); check connectivity or retry later$/.exec(message);
    if (offline) return text(`无法连接 ${offline[1]}，请检查网络或代理设置后重试。`, message);
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
          options={[{value:'ai',label:text('本地底模生成','Generate with base model')},{value:'danbooru',label:'Danbooru'},{value:'gelbooru',label:'Gelbooru'},{value:'e621',label:'e621'},{value:'rule34',label:'Rule34'}]}/></label>
        <label>{fromTraining?text('本批最多生成','Maximum images this batch'):matching?text('本批最多收集','Maximum images this batch'):text('图片数量','Image count')}<input type="number" min={1} max={200} step={1} value={count} disabled={locked} onChange={event=>setCount(Number(event.target.value))}/></label>
      </div>
      {source === 'ai' ? <label>{text('提示词来源','Prompt source')}<StudioSelect aria-label={text('提示词来源','Prompt source')} value={promptSource} disabled={locked} onValueChange={value=>setPromptSource(value as 'manual'|'training_tags')} options={[{value:'manual',label:text('手动填写类别提示词','Enter class prompts')},{value:'training_tags',label:text('按训练图片标签逐张生成','One prior per training image caption')}]}/></label>
        : <label>{text('检索方式','Search method')}<StudioSelect aria-label={text('检索方式','Search method')} value={siteMode} disabled={locked} onValueChange={value=>setSiteMode(value as 'manual'|'training_tags')} options={[{value:'training_tags',label:text('按训练标签自动匹配','Follow the training tags')},{value:'manual',label:text('手动输入检索标签','Enter search tags')}]}/></label>}
      {!fromTraining && !matching && <>      <label className="reg-prompt">{source === 'ai' ? text('类别提示词','Class prompt') : text('站点检索标签','Site search tags')}
        <textarea rows={2} required maxLength={4000} value={prompt} disabled={locked} onChange={event=>setPrompt(event.target.value)} placeholder={source === 'ai' ? text('例如：a photo of a dog；不要写训练主体的触发词','For example: a photo of a dog; omit the subject’s trigger word') : text('例如：dog solo；用空格分隔站点标签','For example: dog solo; separate site tags with spaces')}/>
      </label>
</>}
      {manualSite && !!requestBody.prompt && <p className="reg-note reg-estimate" role="status" aria-live="polite">{estimatePending ? <><Loader2 size={13} className="animate-spin reg-note-spinner" aria-hidden="true"/>{text('正在查询站点匹配数…','Checking matches on the site…')}</>
        : estimate.error ? <span className="reg-error">{formatApiError(estimate.error)}</span>
          : estimate.data && <span title={estimate.data.terms.join(' ')}>{estimate.data.count == null ? text('站点没有给出确切数量，通常是符合条件的图片很多','The site gave no exact count, usually because many images match') : text(`站点约有 ${estimate.data.count.toLocaleString()} 张符合条件的全年龄图片`,`About ${estimate.data.count.toLocaleString()} matching safe images on the site`)}{estimate.data.tag_limit != null && text(`；当前账号每次最多检索 ${estimate.data.tag_limit} 个标签（rating 等条件不计入）`,`; this account searches up to ${estimate.data.tag_limit} tags at a time (rating and similar conditions excluded)`)}{!!estimate.data.local_exclusions.length && text(`，另有 ${estimate.data.local_exclusions.length} 个排除标签超出上限，下载时逐张检查`,`; ${estimate.data.local_exclusions.length} more exclusions are checked on each image`)}</span>}</p>}
      {matching && <div className="reg-training-plan">
        <label>{text('训练目录范围','Training folders')}<StudioSelect aria-label={text('训练目录范围','Training folders')} value={sourceRange} disabled={locked} onValueChange={setSourceRange} options={[{value:'',label:text('全部训练目录','All training folders')},...(match.data?.sources || []).map(item=>({value:item.id,label:`${item.name} · ${item.path}`,displayLabel:item.name}))]}/></label>
        <label>{text('排除标签','Excluded tags')}<input value={excluded} disabled={locked} onChange={event=>setExcluded(event.target.value)} placeholder={text('点击下方标签，或用逗号分隔输入','Select tags below or enter comma-separated tags')}/></label>
        <div className="reg-tag-selection"><strong>{text('训练标签频次 · 点击排除','Training tag frequency · select to exclude')}</strong><div>{match.data?.top_tags.map(item=><button type="button" key={item.tag} aria-label={text(`排除标签：${item.tag}，${item.count} 张图片`,`Exclude tag: ${item.tag}, ${item.count} images`)} aria-pressed={requestBody.excluded_tags.some(tag=>tagKey(tag)===tagKey(item.tag))} disabled={locked} onClick={()=>toggleExcluded(item.tag)}><span>{item.tag}</span><small>{item.count}</small></button>)}</div></div>
        <p className="reg-note">{text('排除的标签不会被检索，带有这些标签的图片也不会下载。请排除角色名等训练主体的标签；触发词会自动排除。','Excluded tags are never searched, and images carrying them are skipped. Exclude the subject’s own tags such as its name; the trigger word is excluded automatically.')}</p>
        <div className="reg-plan-preview" aria-label={text('站点匹配计划','Site matching plan')} role="region" aria-busy={matchPending}>
          {matchPending && <p role="status"><Loader2 size={14} className="animate-spin"/>{text('正在统计训练标签与图片尺寸…','Reading training tags and image sizes…')}</p>}
          {match.error && <p role="alert" className="reg-error">{formatApiError(match.error)} <button type="button" className="ui-btn ui-btn-sm" onClick={()=>void match.refetch()}>{text('重新统计','Read again')}</button></p>}
          {match.data && <><dl><div><dt>{text('范围内图片','Source images')}</dt><dd>{match.data.source_images}</dd></div><div><dt>{text('有标签的图片','Captioned images')}</dt><dd>{match.data.captioned_images}</dd></div><div><dt>{text('已有正则图','Existing priors')}</dt><dd>{match.data.existing_images}</dd></div><div><dt>{text('可检索标签','Searchable tags')}</dt><dd>{match.data.searchable_tags}</dd></div><div><dt>{text('画面宽高比','Aspect ratio')}</dt><dd>{ratio(match.data.aspect.low)}–{ratio(match.data.aspect.high)}</dd></div><div><dt>{text('常见尺寸','Typical size')}</dt><dd>{match.data.size.width} × {match.data.size.height}</dd></div></dl>
          {!!match.data.search_tags.length && <p>{text('优先检索：','Searched first: ')}{match.data.search_tags.slice(0,8).map(item=>`${item.tag} ${percent(item.share)}`).join(text('、',', '))}</p>}
          {source === 'danbooru' && <p>{!match.data.tag_limit_known ? text('无法确认 Danbooru 账号等级，按每次 2 个标签检索。','Could not read the Danbooru account level; searches use 2 tags at a time.') : match.data.tag_limit == null ? text('当前 Danbooru 账号检索不限标签数。','This Danbooru account has no tag limit per search.') : text(`当前 Danbooru 账号每次最多检索 ${match.data.tag_limit} 个标签，会自动组合。`,`This Danbooru account searches up to ${match.data.tag_limit} tags at a time; tags are combined automatically.`)}</p>}
          {!!match.data.unsearchable_tags.length && <p>{text(`不参与检索（不是站点标签）：${match.data.unsearchable_tags.slice(0,8).join('、')}`,`Not searched (not site tags): ${match.data.unsearchable_tags.slice(0,8).join(', ')}`)}</p>}
          {!match.data.captioned_images ? <p className="reg-error">{text('范围内的训练图片还没有标签，先完成打标再收集。','The training images in range have no captions yet; caption them first.')}</p>
            : !match.data.searchable_tags ? <p className="reg-error">{text('排除后没有可检索的标签，请减少排除标签。','No searchable tags remain after exclusions; exclude fewer tags.')}</p>
              : match.data.suggested_count > 0 ? <p className="reg-plan-suggestion">{text(`建议本批 ${match.data.suggested_count} 张，使正则图与训练图片数量持平。`,`Suggested: ${match.data.suggested_count} images, matching the number of training images.`)}{count !== match.data.suggested_count && <button type="button" className="ui-link" disabled={locked} onClick={()=>setCount(match.data!.suggested_count)}>{text('使用建议数量','Use the suggestion')}</button>}</p>
                : <p>{text('正则图数量已达到训练图片数，再收集会继续补齐标签占比。','Priors already match the number of training images; more images keep refining the tag shares.')}</p>}</>}
        </div>
      </div>}
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
      <p className="reg-note">{source === 'ai' ? text('使用本版本底模，不加载 LoRA；请先在训练参数中设置模型。','Uses this version’s base model without LoRA. Select the model in training settings first.') : matching ? text('按训练标签的占比检索站点标记为全年龄的图片，优先挑选宽高比和尺寸接近训练图片的；不下载带排除标签的图片、训练图片本身和已收集过的图片。','Searches safe-rated images in proportion to the training tags, preferring shapes and sizes close to the training images; skips images with excluded tags, the training images themselves and images already collected.') : text('仅收集站点标记为全年龄的图片，保留原标签并去重；已收集过的图片不会重复下载。','Collects images rated safe by the site, keeps source tags and skips duplicates; images already collected are not downloaded again.')}</p>
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
        {source === 'ai' ? <label>{text('负面提示词','Negative prompt')}<input value={negative} disabled={locked} onChange={event=>setNegative(event.target.value)}/></label> : manualSite && <label>{text('排除标签','Excluded tags')}<input value={excluded} disabled={locked} onChange={event=>setExcluded(event.target.value)} placeholder={text('用逗号分隔','Separate with commas')}/></label>}
        <p className="reg-note">{text('正则图不参与自动验证集划分。','Regularization images are excluded from automatic validation splits.')}</p>
      </details>
      {source !== 'ai' && <div className="reg-options"><p className="reg-note" role="status">{credentials.error ? text('无法读取站点密钥状态。','Could not load site-key status.') : credentials.isPending ? <><Loader2 size={13} className="animate-spin reg-note-spinner" aria-hidden="true"/>{text('读取站点密钥状态…','Loading site-key status…')}</> : credentials.data?.[source]?.configured ? text(`${source} 访问密钥已配置`,`${source} access keys configured`) : ['gelbooru', 'rule34'].includes(source) ? text(`${source === 'rule34' ? 'Rule34' : 'Gelbooru'} 需要先配置用户 ID 和 API Key。`, 'Configure the user ID and API key first.') : text(`${source === 'e621' ? 'e621' : 'Danbooru'} 将使用匿名访问。`, 'Anonymous access will be used.')}</p><Link className="ui-link" state={{ backgroundLocation: location.state?.backgroundLocation ?? location }} to={`/settings/environment?tab=credentials#credentials-${source}`}>{text('管理访问密钥','Manage access keys')}</Link>{credentials.error && <button type="button" className="ui-link ml-3" onClick={()=>void credentials.refetch()}>{text('重试','Retry')}</button>}</div>}
      <div className="reg-actions"><span>{readOnly ? text('当前版本只读','This version is read-only') : text('完成后自动加入正则集。','Added to regularization data when complete.')}</span><button type="submit" className="ui-btn ui-btn-primary" disabled={!ready}>{submitting || current ? <Loader2 size={14} className="animate-spin"/> : <ImagePlus size={14}/>} {source === 'ai' ? text('生成正则图','Generate images') : text('收集正则图','Collect images')}</button></div>
    </form>
    {query.data?.operations.slice(0,3).map(task=><article className="reg-task" key={task.id} aria-label={task.id}>
      <div className="reg-task-heading"><strong>{task.source === 'ai' ? text('底模生成','Base model generation') : task.source}</strong><code>{task.id}</code><span className={task.status === 'failed' ? 'reg-failed' : ''}>{statusName(task.status)}</span>{task.can_cancel && <button type="button" className="ui-btn ui-btn-sm" disabled={submitting || readOnly} onClick={()=>void cancel(task.id)} aria-label={text(`取消 ${task.id}`,`Cancel ${task.id}`)}><X size={14}/>{text('取消','Cancel')}</button>}</div>
      {active(task) && <><progress value={task.done} max={Math.max(task.total,1)} aria-label={text('正则图进度','Regularization progress')}/><p className="reg-note">{task.done} / {task.total}</p></>}
      {task.error && <p role="alert" className="reg-error">{errorText(task.error)}</p>}
      {task.status === 'completed' && <p className="reg-note">{text(`新增 ${task.images} 张图片`,`Added ${task.images} images`)}{task.duplicates ? text(`，跳过 ${task.duplicates} 张重复图片`,`, skipped ${task.duplicates} duplicates`) : ''}{task.source !== 'ai' && task.images < task.total ? text(`（目标 ${task.total} 张，站点上没有更多符合条件的图片）`,` (target ${task.total}; the site had no more matching images)`) : ''}</p>}
      {task.dataset_id && <Link className="ui-link" to={`${projectUrl(projectId,versionId,'data')}&data_step=captions`}>{text('查看图片与标签','View images and captions')}</Link>}
      {!!task.logs.length && <details><summary>{text('查看日志','View logs')}</summary><pre>{task.logs.map(logText).join('\n')}</pre></details>}
    </article>)}
    {!readOnly && !current && <details className="reg-options reg-existing"><summary>{text('导入已有正则图','Import existing regularization images')}</summary><ProjectDataImport projectId={projectId} versionId={versionId} defaultIsReg onImported={()=>{void query.refetch();onChanged();}}/></details>}
    </>}
  </section>;
}
