import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ScanLine, Loader2, RefreshCw, X, FolderOpen, ScanSearch, ListChecks, Wand2, Tags, PencilLine, Layers, Image as ImageIcon, CircleAlert, TriangleAlert, Copy as CopyIcon, type LucideIcon } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { projectUrl } from '../../utils/projectVersions';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import './dataset-pipeline.css';
import CaptionViewer from './CaptionViewer';
import AnimaCaptionAdvice, { type CaptionFormatAdvice } from './AnimaCaptionAdvice';
import CaptionWorkspace from './CaptionWorkspace';
import RegularizationPanel from './RegularizationPanel';
import StudioSelect from '../StudioSelect';
import DatasetOperationProgress from './DatasetOperationProgress';
import DatasetCurationPanel from './DatasetCurationPanel';
import TaggingStage from './TaggingStage';
import AutoMaskPanel from './AutoMaskPanel';
import type { WorkspaceDataset } from './ProjectDatasetCards';
import { useEnterAnimation } from '../../utils/motion';
import { useWorkspaceHeight } from '../projects/useWorkspaceHeight';

export type PipelineImageRef = { dataset_id: string; rel_path: string };
type Issue = { severity: 'error' | 'warning'; code: string; message: string; path?: string };
type InspectedImage = { dataset_id: string | null; rel_path: string; path: string; hash: string | null; width: number | null; height: number | null; caption: string; caption_format?: CaptionFormatAdvice; has_mask: boolean; has_alpha?: boolean | null; has_alpha_channel?: boolean | null; image_mode?: string | null; transparency_source?: 'alpha' | 'palette' | 'color_key' | 'metadata' | null; has_transparency?: boolean | null; transparent_pixels?: number; min_alpha?: number; training_enabled?: boolean; roles: string[]; issues: Issue[]; editable: boolean };
type Inspection = { caption_profile?: 'anima' | null; images: InspectedImage[]; duplicate_groups: number[][]; errors: number; warnings: number; captioned: number; masks: number; alpha_images?: number; alpha_channel_images?: number; transparency_metadata_images?: number; transparent_images?: number; source_issues: Issue[] };
export type PipelineOperation = { id: string; action: string; status: string; phase: string; created_at?: number; done: number; total: number; error: string | null; job_id: string | null; can_undo: boolean; can_cancel: boolean; result: { changed_files?: number; rolled_back?: boolean; undone_by?: string; skipped?: number; stopped?: boolean; stopped_reason?: string; failed_files?: number; failures?: { rel_path: string; error: string }[]; images?: number; with_heads?: number; heads?: number; unreadable?: number; applied_by?: string; dismissed?: boolean; stale_images?: number; proposal_id?: string }; logs: { time: number; message: string }[] };
type Plan = { ok: boolean; errors: { loc?: string; msg: string }[]; warnings: { code?: string; msg: string }[]; images?: number; items?: number; buckets?: { w: number; h: number; items: number; batches: number }[]; native?: { downscaled: number; sizes: number; forward_groups: number; logical_batches: number } };
export type PipelineSnapshot = { signature: string; inspection: Inspection | null; plan: Plan | null; operations: PipelineOperation[]; busy: boolean; archived: boolean; stale: boolean; ready_to_train: boolean; prepared_job_id: string | null };
type Props = { projectId: string; versionId: string; config?: Record<string, any>; readOnly?: boolean; datasets?: WorkspaceDataset[]; importPanel: ReactNode; datasetList: ReactNode; onChanged: () => void };
const STAGES = ['datasets', 'inspect', 'curate', 'preprocess', 'tagging', 'captions', 'reg'];
// Links from before the page was reorganised keep landing on the matching stage.
const STAGE_ALIASES: Record<string, string> = { import: 'datasets', paint: 'preprocess' };
const terminal = (status: string) => ['completed', 'failed', 'cancelled'].includes(status);
const PANEL_STAGE: Record<string, string> = { autotag: 'tagging', vlmtag: 'tagging', assisttag: 'tagging', automask: 'preprocess', detectheads: 'preprocess' };

export default function DatasetPipelinePanel({ projectId, versionId, readOnly = false, datasets = [], importPanel, datasetList, onChanged }: Props) {
  const text = useWorkspaceText();
  const navigationRef = useWorkspaceHeight('--pipeline-nav-height');
  const navigate = useNavigate();
  const [params,setParams] = useSearchParams();
  const stageStorage = `studio.pipeline.stage.${projectId}.${versionId}`;
  let remembered = 'datasets';
  try { remembered = sessionStorage.getItem(stageStorage) || 'datasets'; } catch { /* Storage is optional. */ }
  const requested = params.get('data_step') || remembered;
  const normalizedStage = STAGE_ALIASES[requested] || requested;
  const stage = STAGES.includes(normalizedStage) ? normalizedStage : 'datasets';
  const stageBody = useEnterAnimation<HTMLDivElement>(stage, { skipFirst: true });
  useEffect(() => { try { sessionStorage.setItem(stageStorage,stage); } catch { /* URL remains authoritative. */ } },[stage,stageStorage]);
  useEffect(() => {
    const revealStage = () => {
      const list = navigationRef.current?.querySelector<HTMLElement>('.dataset-stages');
      const current = list?.querySelector<HTMLElement>('[aria-current]');
      if (!list || !current) return;
      const viewport = list.getBoundingClientRect();
      const item = current.getBoundingClientRect();
      // Keep the current stage clear of the faded edges.
      const edge = list.scrollWidth > list.clientWidth ? 28 : 0;
      if (item.left < viewport.left + edge) list.scrollLeft -= viewport.left + edge - item.left;
      else if (item.right > viewport.right - edge) list.scrollLeft += item.right - viewport.right + edge;
    };
    // Faded edges show that more stages sit off screen on narrow windows.
    const list = navigationRef.current?.querySelector<HTMLElement>('.dataset-stages');
    const markOverflow = () => {
      if (!list) return;
      list.dataset.before = String(list.scrollLeft > 1);
      list.dataset.after = String(list.scrollLeft + list.clientWidth < list.scrollWidth - 1);
    };
    const update = () => { revealStage(); markOverflow(); };
    update();
    list?.addEventListener('scroll', markOverflow, { passive: true });
    window.addEventListener('resize', update);
    return () => { list?.removeEventListener('scroll', markOverflow); window.removeEventListener('resize', update); };
  }, [stage, navigationRef]);
  const setStage = (value:string) => setParams(previous => { const next = new URLSearchParams(previous); next.set('data_step',value); return next; });
  const [filter, setFilter] = useState('all');
  useEffect(() => { setFilter('all'); }, [projectId, versionId]);
  const [limit, setLimit] = useState(60);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [dismissed, setDismissed] = useState('');
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
  // A finished run changed captions or masks: image lists elsewhere on the page must reload.
  const client = useQueryClient();
  const previousActive = useRef(activeId);
  useEffect(() => {
    if (previousActive.current && previousActive.current !== activeId) {
      void client.invalidateQueries({ queryKey: ['caption-images', projectId, versionId] });
      void client.invalidateQueries({ queryKey: ['caption-datasets', projectId, versionId] });
    }
    previousActive.current = activeId;
  }, [activeId, client, projectId, versionId]);
  const locked = readOnly || !!snapshot?.archived || !!snapshot?.busy || !!active || submitting;
  // The newest operation, when it failed, is reported once until dismissed or replaced.
  const failed = snapshot?.operations[0]?.status === 'failed' && snapshot.operations[0].id !== dismissed ? snapshot.operations[0] : undefined;
  const visible = images.filter(image => filter === 'all' || (filter === 'unused' ? image.training_enabled === false : filter === 'training' ? image.training_enabled !== false : filter === 'has_alpha_channel' ? image.has_alpha_channel === true : filter === 'has_alpha' ? image.has_alpha === true : filter === 'transparent_image' ? image.has_transparency === true : filter === 'has_mask' ? image.has_mask : filter === 'errors' ? image.issues.some(issue => issue.severity === 'error') : filter === 'warnings' ? image.issues.some(issue => issue.severity === 'warning') : filter === 'missing_caption' ? !image.caption : image.issues.some(issue => issue.code === filter)));
  const actionName = (action: string) => ({inspect:text('数据集检查','Inspection'),exclude:text('排除素材','Exclude images'),restore:text('恢复原始文件','Restore originals'),preprocess:text('裁剪 / 缩放','Crop / resize'),captions:text('批量标签','Batch captions'),paint:text('图像涂抹与遮罩','Image painting & masks'),tag:text('历史自动标注','Legacy automatic tagging'),autotag:text('自动打标','Automatic tagging'),automask:text('自动遮罩','Automatic masks'),detectheads:text('检测头部','Head detection'),vlmtag:text('视觉大模型打标','Vision model tagging'),assisttag:text('辅助打标','Assisted tagging'),prepare:text('提前生成缓存','Prepare caches')}[action] || action);
  const statusName = (status: string) => ({queued:text('等待','Queued'),running:text('进行中','Running'),cancelling:text('取消中','Cancelling'),completed:text('完成','Completed'),failed:text('失败','Failed'),cancelled:text('已取消','Cancelled'),staging:text('准备绘制文件','Preparing painted files'),inspecting:text('检查图片、标签与遮罩','Checking images, captions and masks'),preprocessing:text('生成处理结果','Processing images'),captions:text('生成标签','Preparing captions'),applying:text('保存文件与备份','Saving files and backups'),planning:text('检查训练配置','Checking training configuration'),cache:text('编码与缓存','Encoding and caching'),excluding:text('备份并排除','Backing up and excluding'),tagging:text('打标中','Tagging'),detecting:text('检测头部','Detecting heads'),masking:text('写入遮罩','Writing masks'),requesting:text('等待模型回复','Waiting for the model')}[status] || status);
  const issueName = (issue: Issue) => ({anima_artist_prefix:text('画师名前建议加 @','Artist name: consider @ prefix'),anima_tag_spacing:text('普通标签建议用空格','Ordinary tags: consider spaces'),anima_tag_case:text('标签建议用小写','Tags: consider lowercase'),anima_text_shuffle:text('可能含自然语言，请核对洗牌与丢弃设置','Possible prose: review shuffle and dropout'),transparent_image:text('含透明或半透明像素','Transparent or semi-transparent pixels'),small_image:text('短边小于 256px','Short side below 256px'),missing_caption:text('缺少标签','Missing caption'),duplicate:text('相同内容重复图','Exact duplicate'),unreadable_image:text('图片无法完整读取','Cannot decode image'),mask_size:text('遮罩尺寸与图片不同','Mask dimensions differ'),unreadable_mask:text('遮罩无法读取','Cannot read mask'),caption_encoding:text('标签读取或格式错误','Cannot read or parse caption'),caption_json_invalid:text('JSON 标签内容无效','Invalid JSON caption'),caption_format_unsupported:text('此 JSON 标签结构不受支持','Unsupported JSON caption structure'),caption_model_unsupported:text('当前模型不支持此标签格式','Caption format unsupported by this model'),empty_dataset:text('请先导入训练图片','Import training images first'),missing_source:text('素材目录不存在','Source folder is missing')}[issue.code] || issue.message);
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
  const tabs: [string, string, LucideIcon, number | null][] = [
    ['datasets', text('数据集管理','Datasets'), FolderOpen, null],
    ['inspect', text('数据集检查','Inspection'), ScanSearch, report?.errors || null],
    ['curate', text('数据集筛选','Curation'), ListChecks, null],
    ['preprocess', text('数据集预处理','Preprocessing'), Wand2, null],
    ['tagging', text('图片打标','Tagging'), Tags, null],
    ['captions', text('标签编辑','Caption editor'), PencilLine, null],
    ['reg', text('正则数据集','Regularization'), Layers, null],
  ];
  const latest = (action: string) => snapshot?.operations.find(op => op.action === action);
  const inspectionRun = snapshot?.operations.find(op => op.action === 'inspect' && op.status === 'completed');
  const inspectionTime = inspectionRun?.created_at ? new Date(inspectionRun.created_at * 1000).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }) : '';
  // Runs started from a panel show their progress beside that panel's button.
  const hosted = !!active && PANEL_STAGE[active.action] === stage;
  const progress = active ? <DatasetOperationProgress label={actionName(active.action)} phaseText={statusName(active.phase)} done={active.done} total={active.total || null} state="active" elapsed={active.created_at ? Math.max(0, clock / 1000 - active.created_at) : null} detail={active.total > 0 ? text(`当前阶段 ${active.done} / ${active.total} 项`,`Current phase: ${active.done} / ${active.total} items`) : text('正在准备当前阶段…','Preparing this phase…')} actions={<>{active.can_cancel && <button type="button" className="ui-btn ui-btn-sm" disabled={submitting} title={['vlmtag','assisttag'].includes(active.action) && active.phase === 'requesting' ? text('已回复的图片会保留','Images already answered are kept') : undefined} onClick={() => void perform({}, `/dataset-pipeline/operations/${active.id}/cancel`)}>{['vlmtag','assisttag'].includes(active.action) && active.phase === 'requesting' ? text('停止','Stop') : text('取消','Cancel')}</button>}{active.job_id && <Link className="ui-link" to={`/jobs/${active.job_id}`}>{text('任务日志','Job log')}</Link>}</>}/> : null;
  const startOperation = async (body: Record<string, unknown>) => {
    await apiClient.post<PipelineOperation>(`/projects/${projectId}/versions/${versionId}/pipeline/operations`, body, {silent:true});
    await query.refetch(); onChanged();
  };
  const undo = (id: string) => void perform({action:'restore', restore_operation_id:id});
  return <div className="dataset-pipeline" data-testid="dataset-pipeline">
    <div className="pipeline-navigation" ref={navigationRef}>
    <nav className="dataset-stages" aria-label={text('训练数据处理','Dataset pipeline')}>
      {tabs.map(([id, label, Icon, errors]) => <button key={id} type="button" aria-current={stage === id ? 'step' : undefined} onClick={() => setStage(id)}><Icon size={15} aria-hidden="true"/><span>{label}</span>{errors ? <span className="dataset-stage-badge" aria-label={text(`${errors} 项错误`, `${errors} errors`)}>{errors}</span> : null}</button>)}
    </nav>
    </div>
    {(error || query.error) && <div role="alert" className="workspace-message error">{error || formatApiError(query.error)}<button type="button" className="ui-btn ui-btn-sm" onClick={() => void query.refetch()}>{text('重新读取','Reload')}</button></div>}
    {snapshot?.stale && (stage === 'inspect' || stage === 'prepare') && <p className="pipeline-note">{text('检查结果已过期，请重新检查数据。','The inspection is out of date. Inspect the data again.')}</p>}
    {failed && <div role="alert" className="workspace-message error pipeline-failure"><span>{text(`${actionName(failed.action)}失败`, `${actionName(failed.action)} failed`)}{failed.error ? `：${failed.error}` : ''}{failed.result.rolled_back ? text('。更改已回滚，原文件已恢复。', '. Changes were rolled back and the original files restored.') : ''}</span>{!['tag','captions','paint'].includes(failed.action) && <button type="button" className="ui-btn ui-btn-sm" disabled={locked} onClick={() => void perform({}, `/dataset-pipeline/operations/${failed.id}/retry`)}>{text('重试','Retry')}</button>}<button type="button" className="ui-btn ui-btn-quiet ui-btn-icon ui-btn-sm" aria-label={text('关闭提示','Dismiss')} onClick={() => setDismissed(failed.id)}><X size={14}/></button></div>}
    {active && !hosted && progress}
    {query.isPending && <p role="status"><Loader2 size={14} className="animate-spin"/>{text('读取数据状态…','Loading dataset status…')}</p>}
    <div className="pipeline-stage-body" ref={stageBody}>
    {stage === 'datasets' ? <div className={readOnly ? '' : 'version-data-layout'}>{!readOnly && <fieldset disabled={locked}>{importPanel}</fieldset>}{datasetList}</div> : stage === 'curate' ? <DatasetCurationPanel datasets={datasets} readOnly={locked} onChanged={onChanged} onAddImages={() => setStage('datasets')}/> : stage === 'preprocess' ? <div className="pipeline-preprocess"><AutoMaskPanel projectId={projectId} versionId={versionId} locked={locked} latest={latest('automask')} detection={latest('detectheads')} running={hosted ? progress : null} onStart={startOperation} onUndo={undo} onRefresh={refresh}/><CaptionViewer key={`${projectId}/${versionId}/paint`} projectId={projectId} versionId={versionId} readOnly={readOnly || snapshot?.archived || locked} editing initialDatasetId={params.get('dataset') || ''}/></div> : stage === 'tagging' ? <TaggingStage projectId={projectId} versionId={versionId} locked={locked} latest={latest} active={hosted ? active : undefined} progress={hosted ? progress : null} onStart={startOperation} onUndo={undo} onReview={() => setStage('captions')}/> : stage === 'reg' ? <RegularizationPanel projectId={projectId} versionId={versionId} readOnly={readOnly || snapshot?.archived} onChanged={onChanged}/> : <>
      {stage !== 'captions' && <div className="pipeline-toolbar"><h3 className="sr-only">{tabs.find(tab => tab[0] === stage)?.[1]}</h3><button type="button" className="ui-btn ui-btn-primary" disabled={locked} onClick={() => void perform({action: 'inspect'})}><ScanLine size={15}/>{text('检查数据','Inspect data')}</button><button type="button" className="ui-btn ui-btn-icon" aria-label={text('刷新数据状态','Refresh pipeline')} title={text('刷新数据状态','Refresh pipeline')} onClick={refresh}><RefreshCw size={15}/></button>{stage === 'inspect' && inspectionTime && <span className="pipeline-inspection-last">{text('上次检查 ' + inspectionTime + ' · ' + (snapshot?.stale ? '数据已变化' : '数据未变化'), 'Checked ' + inspectionTime + ' · ' + (snapshot?.stale ? 'Data changed' : 'No data changes'))}</span>}</div>}

      {stage !== 'captions' && report?.images.some(image => !image.editable) && <p className="pipeline-note">{text('外部或旧版素材需先复制到新版本，才能编辑。','Copy external or legacy images to a new version before editing.')} <button type="button" className="ui-link" disabled={locked} onClick={() => void copyForEditing()}>{text('复制为可处理的新版本','Copy into an editable version')}</button></p>}
      {stage === 'inspect' && report && <InspectionSummary report={report} images={images} filter={filter} onFilter={value => { setFilter(value); setLimit(60); }}/>}
      {stage === 'inspect' && report?.caption_profile === 'anima' && <AnimaCaptionAdvice images={images} onEdit={() => setStage('captions')}/>}
      {report?.source_issues.map((issue,index) => <p role="alert" className="pipeline-error-text" key={index}>{issueName(issue)} {issue.path}</p>)}
      {stage === 'captions' ? <CaptionWorkspace key={`${projectId}/${versionId}/${params.get('dataset') || ''}`} projectId={projectId} versionId={versionId} initialDatasetId={params.get('dataset') || ''} readOnly={locked} onChanged={onChanged}/> : report ? <><div className="pipeline-inspection-toolbar">
          <label>{text('显示','Show')}<StudioSelect aria-label={text('显示','Show')} value={filter} onValueChange={value => {setFilter(value);setLimit(60);}} options={[
            {value:'all',label:text('全部图片','All images')},{value:'training',label:text('参与训练','In training')},{value:'unused',label:text('暂不训练','Held out')},{value:'errors',label:text('有错误','With errors')},{value:'warnings',label:text('有提示','With warnings')},{value:'duplicate',label:text('重复内容','Exact duplicates')},{value:'missing_caption',label:text('缺少标签','Missing captions')},{value:'small_image',label:text('小尺寸图片','Small images')},{value:'has_mask',label:text('有遮罩','With masks')},{value:'transparent_image',label:text('含透明像素','With transparent pixels')},{value:'has_alpha',label:text('带透明信息','With transparency information')},...(report.alpha_channel_images !== undefined ? [{value:'has_alpha_channel',label:text('含 Alpha 通道','With an alpha channel')}] : [])
          ]}/></label>
          <span className="pipeline-visible-count">{text('显示 ' + visible.length + ' / ' + images.length + ' 张','Showing ' + visible.length + ' / ' + images.length)}</span>
        </div><div className="pipeline-image-browser" role="region" aria-label={text('检查结果图片','Inspected images')} tabIndex={0}><div className="pipeline-image-grid">{visible.slice(0,limit).map(image => <article key={image.dataset_id + '/' + image.rel_path}><div className="pipeline-image-preview">{image.hash && image.dataset_id && !image.issues.some(issue => issue.code === 'unreadable_image') ? <img src={apiUrl(`/datasets/${image.dataset_id}/images/${image.hash}/thumb?size=256`)} alt={image.rel_path} loading="lazy"/> : <span className="pipeline-no-image">{text('无法预览','No preview')}</span>}</div><div><strong title={image.path}>{image.rel_path}</strong>{image.training_enabled === false && <small>{text('暂不训练','Held out')}</small>}<small>{image.width || '—'} × {image.height || '—'} · {image.has_mask ? text('有遮罩','Mask') : text('无遮罩','No mask')}{(filter === 'has_alpha' || filter === 'has_alpha_channel') && image.has_alpha === true ? ` · ${image.image_mode ? `${image.image_mode} · ` : ''}${image.transparency_source === 'alpha' ? text('Alpha 通道','Alpha channel') : image.transparency_source === 'color_key' ? text('色键透明标记','Color-key transparency') : image.transparency_source === 'palette' ? text('调色板透明标记','Palette transparency') : text('透明信息','Transparency information')}${image.has_transparency === false ? text('（无透明像素）',' (no transparent pixels)') : ''}` : ''}{image.roles.includes('validation') ? ` · ${text('验证集','Validation')}` : ''}</small>{image.has_transparency && image.transparent_pixels !== undefined && <small>{text(`${image.transparent_pixels.toLocaleString()} 个透明或半透明像素`, `${image.transparent_pixels.toLocaleString()} transparent or semi-transparent pixels`)} · {text('最低不透明度', 'Minimum opacity')} {((image.min_alpha ?? 0) / 255 * 100).toFixed(1)}%</small>}<p className="pipeline-caption" title={image.caption}>{image.caption || text('暂无标签','No caption')}</p><div className="pipeline-issues">{image.issues.map((issue,index) => issue.code.startsWith('caption_') ? <details key={index} className={issue.severity}><summary>{issueName(issue)}{issue.path && ` · ${issue.path}`}</summary><p>{issue.code === 'caption_format_unsupported' ? text('请使用 tags / nl 标签结构，或在标签格式中明确选择 TXT。原文件未修改。', 'Use a supported tags / nl structure, or select TXT explicitly. The original file is unchanged.') : text('请核对以下具体原因，修复标签文件后重新检查。', 'Review the reason below, fix the caption, then inspect again.')}</p><pre>{issue.message}</pre></details> : <span key={index} className={issue.severity} title={issue.message || issueName(issue)}>{issueName(issue)}</span>)}</div>{!image.editable && <small>{text('此来源仅可检查，复制到新版本后可批量处理','Inspection only; copy into a new version to edit')}</small>}</div></article>)}</div>{visible.length > limit && <button type="button" className="ui-btn ui-btn-sm pipeline-load-more" onClick={() => setLimit(limit + 60)}>{text(`继续显示（剩余 ${visible.length-limit} 张）`,`Show more (${visible.length-limit} remaining)`)}</button>}</div></> : <div className="pipeline-empty"><ScanLine size={24}/></div>}
    </>}
    </div>

  </div>;
}
type InspectionSummaryProps = {
  report: Inspection;
  images: InspectedImage[];
  filter: string;
  onFilter: (value: string) => void;
};

function InspectionSummary({ report, images, filter, onFilter }: InspectionSummaryProps) {
  const text = useWorkspaceText();
  const readable = images.filter(image => !image.issues.some(issue => issue.code === 'unreadable_image')).length;
  const unreadable = images.filter(image => image.issues.some(issue => issue.code === 'unreadable_image')).length;
  const smallImages = images.filter(image => image.issues.some(issue => issue.code === 'small_image')).length;
  const transparentCount = report.transparent_images;
  const duplicateImages = report.duplicate_groups.reduce((total, group) => total + group.length, 0);
  const fullyOpaqueAlpha = images.filter(image => image.has_alpha_channel === true && image.has_transparency === false).length;
  const cards = [
    { key: 'all', icon: ImageIcon, label: text('图像文件', 'Image files'), value: images.length, detail: text(readable + ' 张可正常读取', readable + ' readable'), tone: 'neutral' },
    { key: 'errors', icon: CircleAlert, label: text('错误', 'Errors'), value: report.errors, detail: text(unreadable + ' 张无法读取', unreadable + ' unreadable'), tone: report.errors ? 'danger' : 'neutral' },
    { key: 'warnings', icon: TriangleAlert, label: text('提示', 'Warnings'), value: report.warnings, detail: text(smallImages + ' 张尺寸过小' + (transparentCount ? ' · ' + transparentCount + ' 张透明像素' : ''), smallImages + ' small' + (transparentCount ? ' · ' + transparentCount + ' transparent' : '')), tone: report.warnings ? 'warning' : 'neutral' },
    { key: 'duplicate', icon: CopyIcon, label: text('重复图', 'Duplicates'), value: report.duplicate_groups.length + text(' 组', ' groups'), detail: text('共 ' + duplicateImages + ' 张相同内容', duplicateImages + ' identical images'), tone: report.duplicate_groups.length ? 'warning' : 'neutral' },
    { key: 'has_mask', icon: Layers, label: text('遮罩', 'Masks'), value: report.masks, detail: text(report.masks + ' 张有独立遮罩', report.masks + ' with masks'), tone: 'neutral' },
    { key: 'transparent_image', icon: ImageIcon, label: text('透明像素', 'Transparent pixels'), value: transparentCount ?? '—', detail: transparentCount === undefined ? text('重新检查后显示', 'Reinspect to calculate') : text(transparentCount + ' 张含透明或半透明像素', transparentCount + ' with transparent pixels'), tone: transparentCount ? 'warning' : 'neutral' },
  ];
  return <section className="pipeline-inspection-summary" aria-label={text('检查概况', 'Inspection summary')}>
    <div className="pipeline-inspection-stat-grid">
      {cards.map(card => <button key={card.key} type="button" className={'pipeline-inspection-stat ' + card.tone} aria-pressed={filter === card.key} onClick={() => onFilter(card.key)}>
        <span className="pipeline-inspection-stat-label"><card.icon size={14} aria-hidden="true"/>{card.label}</span>
        <strong>{card.value}</strong>
        <small>{card.detail}</small>
      </button>)}
    </div>
    {report.alpha_images !== undefined && <details className="pipeline-check-scope" open>
      <summary><span>{text('通道与透明信息', 'Channels and transparency')}</span><small>{text('Alpha 通道 ' + (report.alpha_channel_images ?? report.alpha_images) + ' 张 · 其中 ' + fullyOpaqueAlpha + ' 张全不透明', 'Alpha channel ' + (report.alpha_channel_images ?? report.alpha_images) + ' · ' + fullyOpaqueAlpha + ' fully opaque')}</small></summary>
      <div><p>{text('Alpha 通道：' + (report.alpha_channel_images ?? report.alpha_images) + ' 张，其中 ' + fullyOpaqueAlpha + ' 张全不透明。', 'Alpha channel: ' + (report.alpha_channel_images ?? report.alpha_images) + '; ' + fullyOpaqueAlpha + ' fully opaque.')}</p>{!!report.transparency_metadata_images && <p>{text('仅带透明标记：' + report.transparency_metadata_images + ' 张。', 'Transparency metadata only: ' + report.transparency_metadata_images + '.')}</p>}</div>
    </details>}
  </section>;
}
