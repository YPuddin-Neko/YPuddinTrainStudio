import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useLocation } from 'react-router-dom';
import { Download, Loader2, X } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { components } from '../../api/generated';
import type { DatasetInfo } from '../../api/types';
import type { CredentialStates } from '../../pages/Settings/AccessKeys';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import CaptionFormatSelect from '../CaptionFormatSelect';
import ConfigHelp from '../ConfigHelp';
import StudioSelect from '../StudioSelect';
import DatasetLink from './DatasetLink';
import TagSearchInput, { type SiteName } from './TagSearchInput';
import './site-download.css';

type Task = components['schemas']['SiteDownloadTask'];
type Estimate = components['schemas']['SiteDownloadEstimate'];
type Rating = 'general' | 'sensitive' | 'questionable' | 'explicit';
const RATINGS: Rating[] = ['general', 'sensitive', 'questionable', 'explicit'];
const active = (task: Task) => !['completed', 'failed', 'cancelled'].includes(task.status);

/**
 * Training images from Danbooru or Gelbooru: a new dataset of this version, or more images for
 * `targetDataset`. The download runs on the training service, so it goes on when the page closes.
 */
export default function SiteDownloadImport({ projectId, versionId, targetDataset, captionFormats, onImported, onBusyChange }: {
  projectId: string; versionId: string; targetDataset?: DatasetInfo; captionFormats?: readonly string[];
  onImported: () => void; onBusyChange?: (busy: boolean) => void;
}) {
  const text = useWorkspaceText();
  const location = useLocation();
  const ids = React.useId();
  const endpoint = `/projects/${projectId}/versions/${versionId}/site-downloads`;
  const [source, setSource] = React.useState<SiteName>('danbooru');
  const [tags, setTags] = React.useState('');
  const [count, setCount] = React.useState(100);
  const [ratings, setRatings] = React.useState<Rating[]>(['general']);
  const [order, setOrder] = React.useState<'score' | 'newest'>('score');
  const [minScore, setMinScore] = React.useState('');
  const [minSide, setMinSide] = React.useState(512);
  const [excluded, setExcluded] = React.useState('');
  const [name, setName] = React.useState('');
  const [captionExt, setCaptionExt] = React.useState('auto');
  const [repeats, setRepeats] = React.useState(1);
  const [submitting, setSubmitting] = React.useState(false);
  const [error, setError] = React.useState('');
  const snapshot = useQuery({
    queryKey: ['site-downloads', projectId, versionId],
    queryFn: () => apiClient.get<{ operations: Task[] }>(endpoint, { silent: true }),
    refetchInterval: query => query.state.data?.operations.some(active) ? 1200 : false,
  });
  const credentials = useQuery({ queryKey: ['credentials'], queryFn: () => apiClient.get<CredentialStates>('/credentials', { silent: true }), staleTime: 0 });
  const refreshCredentials = credentials.refetch;
  React.useEffect(() => {
    const refresh = () => void refreshCredentials();
    window.addEventListener('credentials.changed', refresh);
    return () => window.removeEventListener('credentials.changed', refresh);
  }, [refreshCredentials]);
  const targetId = targetDataset?.source.id;
  const tasks = (snapshot.data?.operations || []).filter(task => (task.target_dataset_id ?? undefined) === targetId);
  const running = snapshot.data?.operations.find(active);
  const current = tasks.find(active);
  // Finished downloads add datasets and images; the page reloads them once per task.
  const notified = React.useRef<Set<string> | null>(null);
  const imported = React.useRef(onImported); imported.current = onImported;
  React.useEffect(() => {
    if (!snapshot.data) return;
    const done = snapshot.data.operations.filter(task => task.status === 'completed').map(task => task.id);
    if (notified.current && done.some(id => !notified.current!.has(id))) imported.current();
    notified.current = new Set(done);
  }, [snapshot.data]);
  React.useEffect(() => { onBusyChange?.(submitting || !!current); return () => onBusyChange?.(false); }, [submitting, current, onBusyChange]);

  const excludedTags = excluded.split(',').map(value => value.trim()).filter(Boolean);
  const request = {
    source, tags: tags.trim().split(/\s+/).filter(Boolean).join(' '), excluded_tags: excludedTags, count, ratings, order,
    min_score: minScore.trim() === '' ? null : Number(minScore), min_side: minSide,
    ...(targetId ? { dataset_id: targetId } : { name: name.trim(), caption_ext: captionExt, repeats }),
  };
  const credentialsBlocked = credentials.isPending || !!credentials.error || (source === 'gelbooru' && !credentials.data?.gelbooru.configured);
  const estimateKey = JSON.stringify({ source, tags: request.tags, excluded_tags: excludedTags, ratings, order, min_score: request.min_score });
  const [asked, setAsked] = React.useState(estimateKey);
  React.useEffect(() => { const timer = setTimeout(() => setAsked(estimateKey), 600); return () => clearTimeout(timer); }, [estimateKey]);
  const estimate = useQuery({
    queryKey: ['site-download-estimate', endpoint, asked], enabled: !!request.tags && asked === estimateKey && !credentialsBlocked,
    retry: false, refetchOnWindowFocus: false, staleTime: 60_000,
    queryFn: ({ signal }) => apiClient.post<Estimate>(`${endpoint}/estimate`, JSON.parse(asked), { silent: true, signal }),
  });
  const estimatePending = asked !== estimateKey || estimate.isFetching;
  const validCount = Number.isInteger(count) && count >= 1 && count <= 1000;
  const validScore = request.min_score == null || Number.isInteger(request.min_score);
  const ready = !submitting && !running && !credentialsBlocked && !!request.tags && validCount && validScore && ratings.length > 0
    && Number.isInteger(minSide) && minSide >= 0 && (!!targetId || (Number.isInteger(repeats) && repeats >= 1));

  const ratingLabel: Record<Rating, [string, string]> = { general: ['全年龄', 'General'], sensitive: ['敏感', 'Sensitive'], questionable: ['较露骨', 'Questionable'], explicit: ['露骨', 'Explicit'] };
  const errorText = (failure: unknown) => {
    const code = (failure as { code?: string })?.code;
    const message = formatApiError(failure);
    const tag = (failure as { details?: { tag?: string } })?.details?.tag;
    if (code === 'site_download.query') return tag ? text(`「${tag}」不是站点标签；分级、排序和分数请在对应选项中设置。`, message) : message === 'Excluded tags must be plain site tags' ? text('排除标签只能填写站点标签。', message) : text('请填写 1 到 12 个站点标签，用空格分隔。', message);
    return taskError(message);
  };
  const taskError = (message: string) => {
    const fixed: Record<string, string> = {
      'No new matching images were found. Change the tags, ratings or filters; nothing was added.': '没有找到新的符合条件的图片，请调整标签、分级或筛选条件；未添加任何图片。',
      'Studio stopped during the download; nothing was added. Start it again.': '下载途中训练服务停止了，未添加任何图片，请重新开始。',
      'Configure the Gelbooru user ID and API key in Settings → Access keys': 'Gelbooru 需要先在“访问密钥”中配置用户 ID 和 API Key。',
      'A dataset of this version already covers its training folder; add the images to it': '当前版本已有数据集覆盖整个训练目录，请在那个数据集里添加图片。',
      'This model does not read JSON captions; choose TXT': '当前模型不支持 JSON 标签，请选择 TXT。',
    };
    if (fixed[message]) return text(fixed[message], message);
    const http = /^(danbooru|gelbooru) returned HTTP (\d+); check site credentials or retry later$/.exec(message);
    if (http) return text(`${http[1]} 返回 HTTP ${http[2]}，请检查访问密钥或稍后重试。`, message);
    const offline = /^Could not read (danbooru|gelbooru); check connectivity or retry later$/.exec(message);
    if (offline) return text(`无法连接 ${offline[1]}，请检查网络或代理设置后重试。`, message);
    return message;
  };
  const logText = (message: string) => {
    const page = /^Searching (danbooru|gelbooru), page (\d+)$/.exec(message);
    if (page) return text(`正在检索 ${page[1]} 第 ${page[2]} 页`, message);
    const found = /^Found (\d+) of (\d+) matching images; adding them$/.exec(message);
    if (found) return text(`找到 ${found[1]} 张符合条件的图片（目标 ${found[2]} 张），已全部加入`, message);
    const skipped = /^Skipped (\d+) files that were not usable images$/.exec(message);
    if (skipped) return text(`跳过 ${skipped[1]} 个无法使用的文件`, message);
    const local = /^Checking (\d+) excluded tags locally beyond the site's tag limit$/.exec(message);
    if (local) return text(`有 ${local[1]} 个排除标签超出账号的标签上限，下载时逐张检查`, message);
    const adding = /^Adding (\d+) images to the version$/.exec(message);
    if (adding) return text(`正在把 ${adding[1]} 张图片加入当前版本`, message);
    const added = /^Added (\d+) images to (.+)$/.exec(message);
    if (added) return text(`已把 ${added[1]} 张图片加入 ${added[2]}`, message);
    const limit = /^Stopped at the 3 GiB download limit with (\d+) images$/.exec(message);
    if (limit) return text(`单次下载达到 3 GiB 上限，已下载 ${limit[1]} 张`, message);
    const slow = /^(Danbooru|Gelbooru) asked to slow down; retrying in (\d+) s$/.exec(message);
    if (slow) return text(`${slow[1]} 要求放慢请求，${slow[2]} 秒后重试`, message);
    const busy = /^(Danbooru|Gelbooru) is busy \(HTTP (\d+)\); retrying in (\d+) s$/.exec(message);
    if (busy) return text(`${busy[1]} 暂时繁忙（HTTP ${busy[2]}），${busy[3]} 秒后重试`, message);
    const fixed: Record<string, string> = {
      'Preparing the download': '正在准备下载',
      'Cancellation requested': '已请求取消',
      'Sorting by score takes one more tag than this account may search; newest posts come first': '按高分排序需要多占一个标签，当前账号的标签数不够，改为按最新排序',
    };
    return fixed[message] ? text(fixed[message], message) : message;
  };
  const statusName = (status: string) => ({ queued: text('等待开始', 'Queued'), running: text('下载中', 'Downloading'), cancelling: text('正在取消', 'Cancelling'), completed: text('已完成', 'Completed'), failed: text('失败', 'Failed'), cancelled: text('已取消', 'Cancelled') }[status] || status);

  const start = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!ready) return;
    setSubmitting(true); setError('');
    try { await apiClient.post<Task>(endpoint, request, { silent: true }); await snapshot.refetch(); }
    catch (failure) { setError(errorText(failure)); }
    finally { setSubmitting(false); }
  };
  const cancel = async (id: string) => {
    try { await apiClient.post<Task>(`/site-downloads/${id}/cancel`, {}, { silent: true }); await snapshot.refetch(); }
    catch (failure) { setError(errorText(failure)); }
  };
  const toggleRating = (rating: Rating) => setRatings(previous => previous.includes(rating) ? previous.filter(item => item !== rating) : RATINGS.filter(item => item === rating || previous.includes(item)));
  const locked = submitting || !!running;
  const hintId = `${ids}-hint`;
  const sourceLabel = source === 'danbooru' ? 'Danbooru' : 'Gelbooru';
  const defaultName = request.tags.split(' ').slice(0, 3).join('-');

  return <form className="site-download" onSubmit={event => void start(event)} aria-busy={locked}>
    <div className="site-download-grid">
      <label>{text('站点', 'Site')}<StudioSelect aria-label={text('站点', 'Site')} value={source} disabled={locked} onValueChange={value => setSource(value as SiteName)}
        options={[{ value: 'danbooru', label: 'Danbooru' }, { value: 'gelbooru', label: 'Gelbooru' }]}/></label>
      <label>{text('下载数量', 'Images to download')}<input type="number" min={1} max={1000} step={1} value={count} disabled={locked} onChange={event => setCount(Number(event.target.value))}/></label>
    </div>
    <label className="site-download-tags">{text('检索标签', 'Search tags')}
      <TagSearchInput value={tags} onChange={setTags} source={source} disabled={locked} aria-describedby={hintId} placeholder={text('例如 hatsune_miku', 'For example hatsune_miku')}/>
    </label>
    <p id={hintId} className="site-download-hint">{text('填写角色、作品或画师的站点标签，多个标签用空格分隔；输入时会列出站点上的标签。', 'Enter site tags for a character, series or artist, separated by spaces; matching site tags are listed as you type.')}</p>
    <fieldset className="site-download-ratings" disabled={locked}>
      <legend>{text('分级', 'Ratings')}</legend>
      {RATINGS.map(rating => <label key={rating}><input type="checkbox" checked={ratings.includes(rating)} onChange={() => toggleRating(rating)}/>{text(...ratingLabel[rating])}</label>)}
    </fieldset>
    {ratings.some(rating => rating !== 'general') && <p className="site-download-hint">{text('选择全年龄以外的分级时，带有 loli、shota 等未成年相关标签的图片不会下载。', 'With ratings other than General, images tagged loli, shota or other tags for minors are never downloaded.')}</p>}
    {!!request.tags && !credentialsBlocked && <p className="site-download-estimate" role="status" aria-live="polite">{estimatePending ? <><Loader2 size={13} className="animate-spin" aria-hidden="true"/>{text('正在查询站点匹配数…', 'Checking matches on the site…')}</>
      : estimate.error ? <span className="site-download-error">{errorText(estimate.error)}</span>
        : estimate.data && <span title={estimate.data.terms.join(' ')}>
          {estimate.data.count == null ? text('站点没有给出确切数量，通常是符合条件的图片很多', 'The site gave no exact count, usually because many images match') : text(`站点约有 ${estimate.data.count.toLocaleString()} 张符合条件的图片`, `About ${estimate.data.count.toLocaleString()} matching images on the site`)}
          {estimate.data.tag_limit != null && text(`；当前账号每次最多检索 ${estimate.data.tag_limit} 个标签（分级和分数不计入）`, `; this account searches up to ${estimate.data.tag_limit} tags at a time (ratings and score are free)`)}
          {!estimate.data.sorted && text('；标签数不够按高分排序，将按最新排序', '; too few tags left to sort by score, so the newest come first')}
          {!!estimate.data.local_exclusions.length && text(`；${estimate.data.local_exclusions.length} 个排除标签在下载时逐张检查`, `; ${estimate.data.local_exclusions.length} excluded tags are checked on each image`)}
        </span>}</p>}
    <details className="project-import-options site-download-options">
      <summary>{text('下载选项', 'Download options')}</summary>
      <div className="site-download-fields">
        <label>{text('排序', 'Order')}<StudioSelect aria-label={text('排序', 'Order')} value={order} disabled={locked} onValueChange={value => setOrder(value as 'score' | 'newest')}
          options={[{ value: 'score', label: text('高分优先', 'Highest score first') }, { value: 'newest', label: text('最新上传', 'Newest first') }]}/></label>
        <label>{text('最低分数', 'Minimum score')}<input type="number" step={1} value={minScore} disabled={locked} placeholder={text('不限', 'Any')} onChange={event => setMinScore(event.target.value)}/></label>
        <label>{text('最短边（像素）', 'Shortest side (px)')}<input type="number" min={0} max={8192} step={64} value={minSide} disabled={locked} onChange={event => setMinSide(Number(event.target.value))}/></label>
        <label className="site-download-wide">{text('排除标签', 'Excluded tags')}<input value={excluded} disabled={locked} placeholder={text('例如 comic, monochrome；用逗号分隔', 'For example comic, monochrome; separate with commas')} onChange={event => setExcluded(event.target.value)}/></label>
        {!targetDataset && <>
          <label className="site-download-phone-wide">{text('数据集名称', 'Dataset name')}<input value={name} maxLength={100} disabled={locked} placeholder={defaultName || text('按检索标签命名', 'Named after the tags')} onChange={event => setName(event.target.value)}/></label>
          <label className="site-download-phone-wide">{text('标签格式', 'Caption format')}<CaptionFormatSelect value={captionExt} onChange={setCaptionExt} disabled={locked} formats={captionFormats}/></label>
          <div className="site-download-field"><span className="site-download-label">{text('每张图片重复次数', 'Repeats per image')}<ConfigHelp label={text('重复次数说明', 'Repeats help')}>{text('每轮使用每张图片的次数，默认 1 次。次数越高，这组图片的训练占比越大。', 'Times each image is used per epoch, default 1. More repeats increase this dataset’s share.')}</ConfigHelp></span><input aria-label={text('每张图片重复次数', 'Repeats per image')} type="number" min={1} step={1} value={repeats} disabled={locked} onChange={event => setRepeats(Number(event.target.value))}/></div>
        </>}
      </div>
      <p className="site-download-hint">{text('图片按站点原文件保存；站点标签写成标注文件：人数在前，然后是角色、作品和画师，其余标签按站点顺序；下划线改为空格。', 'Images keep the site’s original files. Captions list the site tags: people count first, then characters, series and artists, then the other tags in the site’s order, with spaces for underscores.')}</p>
    </details>
    <div className="site-download-access">
      <p role="status">{credentials.error ? text('无法读取站点密钥状态。', 'Could not load site-key status.') : credentials.isPending ? text('读取站点密钥状态…', 'Loading site-key status…')
        : credentials.data?.[source]?.configured ? text(`${sourceLabel} 访问密钥已配置`, `${sourceLabel} access keys configured`)
          : source === 'gelbooru' ? text('Gelbooru 需要先配置用户 ID 和 API Key。', 'Configure the Gelbooru user ID and API key first.') : text('Danbooru 尚未配置密钥，将匿名访问，每次最多检索 2 个标签。', 'Danbooru keys are not configured; anonymous access searches up to 2 tags at a time.')}</p>
      <Link className="ui-link" state={{ backgroundLocation: location.state?.backgroundLocation ?? location }} to={`/settings/environment?tab=credentials#credentials-${source}`}>{text('管理访问密钥', 'Manage access keys')}</Link>
      <Link className="ui-link" state={{ backgroundLocation: location.state?.backgroundLocation ?? location }} to="/settings/preferences?section=interface#preferences-network">{text('网络代理', 'Network proxy')}</Link>
    </div>
    {error && <p role="alert" className="site-download-error">{error}</p>}
    {snapshot.error && <p role="alert" className="site-download-error">{formatApiError(snapshot.error)}</p>}
    <div className="site-download-actions">
      <span>{running && !current ? text('当前版本有另一个下载正在进行。', 'Another download is running for this version.') : targetDataset ? text('下载完成后加入当前数据集。', 'Added to this dataset when the download finishes.') : text('下载完成后作为新的数据集加入当前版本。', 'Added to this version as a new dataset when the download finishes.')}</span>
      <button type="submit" className="ui-btn ui-btn-primary" disabled={!ready}>{submitting || current ? <Loader2 size={14} className="animate-spin"/> : <Download size={14}/>}{text('开始下载', 'Start download')}</button>
    </div>
    {tasks.slice(0, 3).map(task => <article key={task.id} className="site-download-task" aria-label={text(`下载任务 ${task.query}`, `Download ${task.query}`)}>
      <div className="site-download-task-heading"><strong>{task.source === 'danbooru' ? 'Danbooru' : 'Gelbooru'} · {task.query}</strong>
        <span className={task.status === 'failed' ? 'site-download-failed' : ''}>{statusName(task.status)}</span>
        {task.can_cancel && <button type="button" className="ui-btn ui-btn-sm" onClick={() => void cancel(task.id)} aria-label={text(`取消下载 ${task.query}`, `Cancel download ${task.query}`)}><X size={14}/>{text('取消', 'Cancel')}</button>}</div>
      {active(task) && <><progress value={task.done} max={Math.max(task.total, 1)} aria-label={text('下载进度', 'Download progress')}/><p className="site-download-hint">{task.done} / {task.total}</p></>}
      {task.error && <p role="alert" className="site-download-error">{taskError(task.error)}</p>}
      {task.status === 'completed' && <p className="site-download-hint">{text(`已加入 ${task.images} 张图片`, `Added ${task.images} images`)}{task.duplicates ? text(`，跳过 ${task.duplicates} 张重复图片`, `, skipped ${task.duplicates} duplicates`) : ''}{task.images < task.total ? text(`（目标 ${task.total} 张，站点上没有更多符合条件的新图片）`, ` (target ${task.total}; the site had no more new matching images)`) : ''}</p>}
      {task.status === 'completed' && task.dataset_id && !targetDataset && <DatasetLink className="ui-link" to={`/datasets/${task.dataset_id}`}>{text('查看图片与标签', 'Review images and captions')}</DatasetLink>}
      {!!task.logs.length && <details><summary>{text('查看日志', 'View logs')}</summary><pre>{task.logs.map(logText).join('\n')}</pre></details>}
    </article>)}
  </form>;
}
