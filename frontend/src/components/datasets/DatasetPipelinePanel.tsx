import { useEffect, useState, type ReactNode } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { CheckCircle2, ScanLine, Loader2, RotateCcw, ArrowRight, RefreshCw } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { modelConfigUrl, projectUrl } from '../../utils/projectVersions';
import { presentConfigIssues, presentPlanWarning } from '../../utils/configPresentation';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import './dataset-pipeline.css';
import CaptionViewer from './CaptionViewer';
import CaptionWorkspace from './CaptionWorkspace';
import RegularizationPanel from './RegularizationPanel';
import StudioSelect from '../StudioSelect';
import DatasetOperationProgress from './DatasetOperationProgress';
import { useWorkspaceHeight } from '../projects/useWorkspaceHeight';

export type PipelineImageRef = { dataset_id: string; rel_path: string };
type Issue = { severity: 'error' | 'warning'; code: string; message: string; path?: string };
type InspectedImage = { dataset_id: string | null; rel_path: string; path: string; hash: string | null; width: number | null; height: number | null; caption: string; has_mask: boolean; roles: string[]; issues: Issue[]; editable: boolean };
type Inspection = { images: InspectedImage[]; duplicate_groups: number[][]; errors: number; warnings: number; captioned: number; masks: number; source_issues: Issue[] };
export type PipelineOperation = { id: string; action: string; status: string; phase: string; created_at?: number; done: number; total: number; error: string | null; job_id: string | null; can_undo: boolean; can_cancel: boolean; result: { changed_files?: number; rolled_back?: boolean; undone_by?: string }; logs: { time: number; message: string }[] };
type Plan = { ok: boolean; errors: { loc?: string; msg: string }[]; warnings: { code?: string; msg: string }[]; images?: number; items?: number; buckets?: { w: number; h: number; items: number; batches: number }[]; native?: { downscaled: number; sizes: number; forward_groups: number; logical_batches: number } };
export type PipelineSnapshot = { signature: string; inspection: Inspection | null; plan: Plan | null; operations: PipelineOperation[]; busy: boolean; archived: boolean; stale: boolean; ready_to_train: boolean; prepared_job_id: string | null };
type Props = { projectId: string; versionId: string; config?: Record<string, any>; readOnly?: boolean; importPanel: ReactNode; datasetList: ReactNode; onChanged: () => void };
const terminal = (status: string) => ['completed', 'failed', 'cancelled'].includes(status);
const keyOf = (image: { dataset_id: string | null; rel_path: string }) => `${image.dataset_id}/${image.rel_path}`;

export default function DatasetPipelinePanel({ projectId, versionId, readOnly = false, importPanel, datasetList, onChanged }: Props) {
  const text = useWorkspaceText();
  const navigationRef = useWorkspaceHeight('--pipeline-nav-height');
  const navigate = useNavigate();
  const [params,setParams] = useSearchParams();
  const stageStorage = `studio.pipeline.stage.${projectId}.${versionId}`;
  let remembered = 'import';
  try { remembered = sessionStorage.getItem(stageStorage) || 'import'; } catch { /* Storage is optional. */ }
  const requested = params.get('data_step') || remembered;
  const normalizedStage = requested === 'preprocess' ? 'paint' : requested;
  const stage = ['import','inspect','paint','captions','reg','prepare'].includes(normalizedStage) ? normalizedStage : 'import';
  useEffect(() => { try { sessionStorage.setItem(stageStorage,stage); } catch { /* URL remains authoritative. */ } },[stage,stageStorage]);
  useEffect(() => {
    const revealStage = () => {
      const list = navigationRef.current?.querySelector<HTMLElement>('.pipeline-stages');
      const current = list?.querySelector<HTMLElement>('[aria-current]');
      if (!list || !current) return;
      const viewport = list.getBoundingClientRect();
      const item = current.getBoundingClientRect();
      if (item.left < viewport.left) list.scrollLeft -= viewport.left - item.left;
      else if (item.right > viewport.right) list.scrollLeft += item.right - viewport.right;
    };
    revealStage();
    window.addEventListener('resize', revealStage);
    return () => window.removeEventListener('resize', revealStage);
  }, [stage, navigationRef]);
  const setStage = (value:string) => setParams(previous => { const next = new URLSearchParams(previous); next.set('data_step',value); return next; });
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState('all');
  const [limit, setLimit] = useState(60);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [clock, setClock] = useState(() => Date.now());
  const query = useQuery({
    queryKey: ['dataset-pipeline', projectId, versionId],
    queryFn: () => apiClient.get<PipelineSnapshot>(`/projects/${projectId}/versions/${versionId}/pipeline`, { silent: true }),
    refetchInterval: q => q.state.data?.busy || q.state.data?.operations.some(op => !terminal(op.status)) ? 1200 : false,
  });
  const refresh = () => { void query.refetch(); onChanged(); };
  useEventStream(EVENT_TYPES.DATASET_CHANGED, () => { void query.refetch(); });
  useEventStream(EVENT_TYPES.JOB_STATE, () => { void query.refetch(); });
  const snapshot = query.data;
  const report = snapshot?.inspection;
  const images = report?.images || [];
  const active = snapshot?.operations.find(op => !terminal(op.status));
  const activeId = active?.id;
  useEffect(() => {
    if (!activeId) return;
    const timer = window.setInterval(() => setClock(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [activeId]);
  const locked = readOnly || !!snapshot?.archived || !!snapshot?.busy || !!active || submitting;
  const selection = images.filter(image => image.editable && image.dataset_id && selected.has(keyOf(image))).map(image => ({dataset_id: image.dataset_id!, rel_path: image.rel_path}));
  const visible = images.filter(image => filter === 'all' || (filter === 'errors' ? image.issues.some(issue => issue.severity === 'error') : filter === 'missing_caption' ? !image.caption : image.issues.some(issue => issue.code === filter)));
  const actionName = (action: string) => ({inspect:text('数据检查','Data inspection'),exclude:text('排除素材','Exclude images'),restore:text('恢复原始文件','Restore originals'),preprocess:text('裁剪 / 缩放','Crop / resize'),captions:text('批量标签','Batch captions'),paint:text('图像涂抹与遮罩','Image painting & masks'),tag:text('历史自动标注','Legacy automatic tagging'),prepare:text('提前生成缓存','Prepare caches')}[action] || action);
  const statusName = (status: string) => ({queued:text('等待','Queued'),running:text('进行中','Running'),cancelling:text('取消中','Cancelling'),completed:text('完成','Completed'),failed:text('失败','Failed'),cancelled:text('已取消','Cancelled'),staging:text('准备绘制文件','Preparing painted files'),inspecting:text('检查图片、标签与遮罩','Checking images, captions and masks'),preprocessing:text('生成处理结果','Processing images'),captions:text('生成标签','Preparing captions'),applying:text('保存文件与备份','Saving files and backups'),planning:text('检查训练配置','Checking training configuration'),cache:text('编码与缓存','Encoding and caching'),excluding:text('备份并排除','Backing up and excluding')}[status] || status);
  const issueName = (issue: Issue) => ({small_image:text('短边小于 256px','Short side below 256px'),missing_caption:text('缺少标签','Missing caption'),duplicate:text('相同内容重复图','Exact duplicate'),unreadable_image:text('图片无法完整解码','Cannot decode image'),mask_size:text('遮罩尺寸与图片不同','Mask dimensions differ'),unreadable_mask:text('遮罩无法读取','Cannot read mask'),caption_encoding:text('标签读取或格式错误','Cannot read or parse caption'),empty_dataset:text('请先导入训练图片','Import training images first'),missing_source:text('素材目录不存在','Source folder is missing')}[issue.code] || issue.message);
  const perform = async (body: Record<string, unknown>, endpoint?: string) => {
    setError(''); setSubmitting(true);
    try {
      await apiClient.post<PipelineOperation>(endpoint || `/projects/${projectId}/versions/${versionId}/pipeline/operations`, body, {silent:true});
      await query.refetch(); onChanged();
    } catch (e) { setError(formatApiError(e)); }
    finally { setSubmitting(false); }
  };
  const copyForEditing = async () => {
    setSubmitting(true); setError('');
    try {
      const created = await apiClient.post<{id:string}>(`/projects/${projectId}/versions`, {name:text('素材处理副本','Data preparation copy'), source_version_id:versionId, data_mode:'copy'}, {silent:true});
      onChanged(); navigate(projectUrl(projectId,created.id,'data'));
    } catch (e) {setError(formatApiError(e));} finally {setSubmitting(false);}
  };
  const tabs = [
    ['import', text('导入','Import'), null],
    ['inspect', text('检查与筛选','Inspect & curate'), report ? `${images.length}` : null],
    ['paint', text('涂抹与遮罩','Paint & masks'), text('可选','Optional')],
    ['captions', text('标签编辑','Caption editor'), report ? `${report.captioned}/${images.length}` : null],
    ['reg', text('正则图','Regularization'), text('可选','Optional')],
    ['prepare', text('训练缓存','Training caches'), text('可选','Optional')],
  ];
  return <div className="dataset-pipeline" data-testid="dataset-pipeline">
    <div className="pipeline-navigation" ref={navigationRef}>
    <nav className="pipeline-stages" aria-label={text('数据集流水线','Dataset pipeline')}>
      {tabs.map(([id, label, badge]) => <button key={id} type="button" aria-current={stage === id ? 'step' : undefined} title={badge ? `${label} · ${badge}` : undefined} onClick={() => setStage(id!)}><strong>{label}</strong>{id === 'prepare' && <small>{badge}</small>}</button>)}
    </nav>
    <div className="pipeline-stage-navigation"><Link to={projectUrl(projectId,versionId,'train')}>{text('训练参数','Training settings')}<ArrowRight size={14}/></Link></div>
    </div>
    {(error || query.error) && <div role="alert" className="workspace-message error">{error || formatApiError(query.error)}<button onClick={() => void query.refetch()}>{text('重新读取','Reload')}</button></div>}
    {snapshot?.stale && (stage === 'inspect' || stage === 'prepare') && <p className="pipeline-note">{text('图片、标签、遮罩或数据配置已变化，请重新检查。旧操作记录和备份仍保留。','Images, captions, masks or data settings changed. Run inspection again; operation history and backups are retained.')}</p>}
    {active && <DatasetOperationProgress label={actionName(active.action)} phaseText={statusName(active.phase)} done={active.done} total={active.total || null} state="active" elapsed={active.created_at ? Math.max(0, clock / 1000 - active.created_at) : null} detail={active.total > 0 ? text(`当前阶段 ${active.done} / ${active.total} 项`,`Current phase: ${active.done} / ${active.total} items`) : text('正在准备当前阶段…','Preparing this phase…')} actions={<>{active.can_cancel && <button disabled={submitting} onClick={() => void perform({}, `/dataset-pipeline/operations/${active.id}/cancel`)}>{text('取消','Cancel')}</button>}{active.job_id && <Link to={`/jobs/${active.job_id}`}>{text('任务日志','Job log')}</Link>}</>}/>}
    {query.isPending && <p role="status"><Loader2 size={14} className="animate-spin"/>{text('读取数据状态…','Loading dataset status…')}</p>}
    {stage === 'import' ? <div className={readOnly ? '' : 'version-data-layout'}>{!readOnly && <fieldset disabled={locked}>{importPanel}</fieldset>}{datasetList}</div> : stage === 'paint' ? <CaptionViewer key={`${projectId}/${versionId}/paint`} projectId={projectId} versionId={versionId} readOnly={readOnly || snapshot?.archived || locked} editing initialDatasetId={params.get('dataset') || ''}/> : stage === 'reg' ? <RegularizationPanel projectId={projectId} versionId={versionId} readOnly={readOnly || snapshot?.archived} onChanged={onChanged}/> : <>
      {stage !== 'captions' && <div className="pipeline-toolbar"><div><h3 className="sr-only">{tabs.find(tab => tab[0] === stage)?.[1]}</h3><p>{stage === 'inspect' ? text('查找损坏文件、缺失标签、重复图与遮罩尺寸问题。检查只读取数据。','Find damaged files, missing captions, exact duplicates and mask size issues. Inspection only reads data.') : stage === 'captions' ? text('选择图片查看完整的已有标签；可按文件名或标签搜索。','Select an image to read its existing caption. Search by filename or caption.') : text('开始训练会自动准备缓存。也可以在这里提前生成，供相同配置的训练复用。','Training prepares caches automatically. You can also generate them here in advance for training with matching settings.')}</p></div>{stage === 'inspect' && <button className="pipeline-primary" disabled={locked} onClick={() => void perform({action: 'inspect'})}><ScanLine size={15}/>{text('检查数据','Inspect data')}</button>}<button aria-label={text('刷新流水线','Refresh pipeline')} onClick={refresh}><RefreshCw size={15}/></button>{stage !== 'captions' && report && <div className="pipeline-summary"><span>{images.length} {text('个图像文件','image files')} · {images.filter(image => !image.issues.some(issue => issue.code === 'unreadable_image')).length} {text('张可解码','decodable images')}</span><span className={report.errors ? 'pipeline-error-text' : ''}>{report.errors} {text('项错误','errors')}</span><span>{report.warnings} {text('项提示','warnings')}</span><span>{report.duplicate_groups.length} {text('组重复图','duplicate groups')}</span><span>{report.masks} {text('张遮罩','masks')}</span></div>}</div>}

      {stage === 'inspect' && <details className="pipeline-check-scope"><summary>{text('检查哪些内容','What is checked')}</summary><ul><li>{text('图片能否完整解码，是否有短边小于 256px 的小图。','Whether images fully decode, and whether either side is below 256px.')}</li><li>{text('TXT / JSON 标签能否读取和解析，是否为空。','Whether TXT / JSON captions can be read and parsed, and whether they are empty.')}</li><li>{text('遮罩能否读取，尺寸是否与图片一致。','Whether masks are readable and match their images in size.')}</li><li>{text('按文件内容识别完全相同的重复图片。','Find byte-for-byte duplicate images.')}</li><li>{text('数据目录是否存在，是否包含可用图片。','Whether source folders exist and contain images.')}</li></ul><p>{text('这里不评价画面美感或标签语义。只有你执行“排除”才会移动文件，可从底部操作记录恢复。','This does not score aesthetics or caption accuracy. Files move only when you choose Exclude, and can be restored from the history below.')}</p></details>}
      {stage !== 'captions' && report?.images.some(image => !image.editable) && <p className="pipeline-note">{text('旧版或外部引用素材先复制到独立新版本，之后可安全筛选与处理。','Copy legacy or external sources into an independent version before editing.')} <button disabled={locked} onClick={() => void copyForEditing()}>{text('复制为可处理的新版本','Copy into an editable version')}</button></p>}
      {report?.source_issues.map((issue,index) => <p role="alert" className="pipeline-error-text" key={index}>{issueName(issue)} {issue.path}</p>)}
      {stage === 'captions' ? <CaptionWorkspace key={`${projectId}/${versionId}/${params.get('dataset') || ''}`} projectId={projectId} versionId={versionId} initialDatasetId={params.get('dataset') || ''} readOnly={locked} onChanged={onChanged}/> : stage === 'prepare' ? <div className="pipeline-preparation"><div className="pipeline-cache-explanation"><h3>{text('缓存里保存什么？','What is cached?')}</h3><p>{text('图像缓存：VAE 将图片转换为训练用的 latent，后续训练可直接读取。','Image cache: the VAE converts images into latents that training can read directly.')}</p><p>{text('文本缓存：选择预先编码时，保存标签和采样提示词的编码结果；随训练编码时则按需计算。','Text cache: pre-encoding stores encoded captions and sample prompts; encoding during training computes them as needed.')}</p><p>{text('缓存按图片、标签、尺寸和编码器配置识别，已有匹配内容会复用。','Matching cache entries are reused based on images, captions, dimensions and encoder settings.')}</p></div><div className="pipeline-prepare-actions"><button className="pipeline-primary" disabled={locked} onClick={() => void perform({action:'prepare'})}>{submitting ? <Loader2 size={15} className="animate-spin"/> : <CheckCircle2 size={15}/>}{text('提前生成缓存（可选）','Generate caches in advance (optional)')}</button><Link to={modelConfigUrl(projectId,versionId)}>{text('选择训练模型','Choose training model')}</Link><Link to={projectUrl(projectId,versionId,'train')}>{text('配置分辨率与训练参数','Configure resolution and training')}</Link></div>{snapshot?.ready_to_train && <div role="status" className="pipeline-ready"><CheckCircle2 size={18}/>{text('此版本当前配置的训练缓存已完成。','Training caches are ready for the current version configuration.')}<Link to={projectUrl(projectId,versionId,'train')}>{text('进入训练','Continue to training')}<ArrowRight size={14}/></Link></div>}{snapshot?.prepared_job_id && <Link to={`/jobs/${snapshot.prepared_job_id}`}>{text('查看缓存任务与日志','View cache job and logs')}</Link>}{snapshot?.plan && <><div className="pipeline-summary"><span>{snapshot.plan.images} {text('张训练图片','training images')}</span><span>{snapshot.plan.buckets?.length} {text(snapshot.plan.native ? '种原生尺寸' : '个分桶',snapshot.plan.native ? 'native shapes' : 'buckets')}</span>{snapshot.plan.native && <span>{snapshot.plan.native.downscaled} {text('张因预算缩小','images downscaled for budget')}</span>}</div>{presentConfigIssues(snapshot.plan.errors, text('zh','en') === 'en').map((issue,index) => <details className="pipeline-error-text" key={index}><summary>{issue.label} · {issue.message}</summary><p>{issue.detail}</p></details>)}{snapshot.plan.warnings.map((issue,index) => <p className="pipeline-note" key={index}>{text(presentPlanWarning(issue.code || '',issue.msg),issue.msg)}</p>)}<div className="pipeline-bucket-list">{snapshot.plan.buckets?.map(bucket => <span key={`${bucket.w}x${bucket.h}`}><strong>{bucket.w} × {bucket.h}</strong> · {bucket.items} {text('项','items')} · {bucket.batches} {text('前向组','forward groups')}</span>)}</div></>}</div> : report ? <><div className="pipeline-selection"><label>{text('显示','Show')}<StudioSelect aria-label={text('显示','Show')} value={filter} onValueChange={value => {setFilter(value);setLimit(60);}} options={[
        {value:'all',label:text('全部图片','All images')},{value:'errors',label:text('有错误','With errors')},{value:'duplicate',label:text('重复图片','Duplicates')},{value:'missing_caption',label:text('缺少标签','Missing captions')},{value:'small_image',label:text('小尺寸图片','Small images')}
      ]}/></label><button disabled={locked} onClick={() => setSelected(new Set(visible.filter(image => image.editable).map(keyOf)))}>{text('选择当前筛选结果','Select filtered images')}</button><button disabled={locked} onClick={() => setSelected(new Set())}>{text('清空选择','Clear selection')}</button>{stage === 'inspect' && <><button disabled={locked || !report.duplicate_groups.length} onClick={() => setSelected(new Set(report.duplicate_groups.flatMap(group => [...group].sort((a,b) => Number(images[b].has_mask)-Number(images[a].has_mask) || Number(!!images[b].caption)-Number(!!images[a].caption) || images[a].path.localeCompare(images[b].path)).slice(1).map(index => images[index])).filter(image => image.editable).map(keyOf)))}>{text('选择重复副本（每组保留一张）','Select duplicates (keep one per group)')}</button><button disabled={locked || !selection.length} onClick={() => void perform({action:'exclude',images:selection})}>{text(`排除 ${selection.length} 张选中图片`,`Exclude ${selection.length} selected images`)}</button></>}<strong>{text(`已选 ${selection.length} 张`,`Selected ${selection.length}`)}</strong></div><div className="pipeline-image-browser" role="region" aria-label={text('待处理图片','Images to curate')} tabIndex={0}><div className="pipeline-image-grid">{visible.slice(0,limit).map(image => <article key={keyOf(image)} className={selected.has(keyOf(image)) ? 'selected' : ''}><label><input type="checkbox" aria-label={text(`选择 ${image.rel_path}`,`Select ${image.rel_path}`)} disabled={locked || !image.editable} checked={selected.has(keyOf(image))} onChange={() => setSelected(previous => {const next = new Set(previous); if (next.has(keyOf(image))) next.delete(keyOf(image)); else next.add(keyOf(image)); return next;})}/>{image.hash && image.dataset_id && !image.issues.some(issue => issue.code === 'unreadable_image') ? <img src={apiUrl(`/datasets/${image.dataset_id}/images/${image.hash}/thumb?size=256`)} alt={image.rel_path} loading="lazy"/> : <span className="pipeline-no-image">{text('无法预览','No preview')}</span>}</label><div><strong title={image.path}>{image.rel_path}</strong><small>{image.width || '—'} × {image.height || '—'} · {image.has_mask ? text('有遮罩','Mask') : text('无遮罩','No mask')}{image.roles.includes('validation') ? ` · ${text('验证集','Validation')}` : ''}</small><p className="pipeline-caption" title={image.caption}>{image.caption || text('暂无标签','No caption')}</p><div className="pipeline-issues">{image.issues.map((issue,index) => <span key={index} className={issue.severity} title={issue.message}>{issueName(issue)}</span>)}</div>{!image.editable && <small>{text('此来源仅可检查，复制到新版本后可批量处理','Inspection only; copy into a new version to edit')}</small>}</div></article>)}</div>{visible.length > limit && <button className="pipeline-load-more" onClick={() => setLimit(limit + 60)}>{text(`继续显示（剩余 ${visible.length-limit} 张）`,`Show more (${visible.length-limit} remaining)`)}</button>}</div></> : <div className="pipeline-empty"><ScanLine size={24}/><strong>{text('先检查本版本的数据','Inspect this version first')}</strong><p>{text('点击“检查数据”读取真实图片和标签，随后选择要处理的素材。','Use Inspect data to read actual images and captions, then select files to process.')}</p></div>}
    </>}

    {!!snapshot?.operations.length && <footer className="pipeline-footer"><details className="pipeline-history" open={historyOpen} onToggle={event => setHistoryOpen(event.currentTarget.open)}><summary>{text('操作记录','Operation history')} <span className="pipeline-history-count">{snapshot.operations.length}</span>{snapshot.operations.some(op => op.status === 'failed') && <span className="pipeline-error-text"> · {text('有失败记录','Failed operation')}</span>}</summary><div className="pipeline-history-list">{snapshot.operations.map(op => <div className="pipeline-history-row" key={op.id}><div className="pipeline-history-title"><strong>{actionName(op.action)}</strong><span>{statusName(op.status)}</span>{op.result.changed_files !== undefined && <small>{op.result.changed_files} {text('个文件','files')}</small>}{op.can_undo && <button disabled={locked} onClick={() => void perform({action:'restore',restore_operation_id:op.id})}><RotateCcw size={13}/>{text('恢复此操作前的文件','Restore files before this operation')}</button>}{!['tag','captions','paint'].includes(op.action) && ['failed','cancelled'].includes(op.status) && <button disabled={locked} onClick={() => void perform({},`/dataset-pipeline/operations/${op.id}/retry`)}>{text('重试','Retry')}</button>}{op.action === 'paint' && ['failed','cancelled'].includes(op.status) && <button disabled={locked} onClick={() => setStage('paint')}>{text('返回涂抹与遮罩','Return to paint & masks')}</button>}{op.job_id && <Link to={`/jobs/${op.job_id}`}>{text('打开任务','Open job')}</Link>}</div>{op.error && <p role="alert" className="pipeline-error-text">{op.error}</p>}{op.action === 'inspect' && <p className="pipeline-note">{text('只读取数据，未修改文件。','Read-only inspection; files were not changed.')}</p>}{op.result.rolled_back && <p>{text('本次更改已回滚，原文件已恢复。','Changes from this operation were rolled back and original files restored.')}</p>}{op.result.undone_by && <small>{text('已恢复','Restored')}</small>}{!!op.logs.length && <details><summary>{text('详细日志','Detailed logs')}</summary><pre>{op.logs.map(log => `${new Date(log.time*1000).toLocaleTimeString()} ${log.message}`).join('\n')}</pre></details>}</div>)}</div></details></footer>}
  </div>;
}
