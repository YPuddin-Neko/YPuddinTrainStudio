import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Archive, ChevronLeft, ChevronRight, LayoutGrid, List, RefreshCw, Search, SearchX } from 'lucide-react';
import { Link, useSearchParams } from 'react-router-dom';
import { archivedVersions, type ArchivedVersion } from '../../api/archive';
import { apiClient } from '../../api/client';
import type { FamilyInfo } from '../../api/types';
import { ResourceActivityContext } from '../../api/resourcePolicy';
import PageLocation from '../../components/PageLocation';
import StudioSelect from '../../components/StudioSelect';
import { SlidingIndicator } from '../../components/motion';
import { useWorkspaceText } from '../../utils/workspaceText';
import { archiveUrl, projectUrl, type ProjectVersion } from '../../utils/projectVersions';
import { familyDisplayName } from '../../utils/trainingFamilies';
import { ttsEngineLabel } from '../../utils/ttsEngines';
import { formatApiError } from '../../utils/errors';
import { ProjectArtwork, ProjectDeletionLine } from './ProjectCardParts';
import ProjectCardMenu from './ProjectCardMenu';
import ProjectDeleteDialog from './ProjectDeleteDialog';
import type { GalleryProject } from './projectGallery';
import './projects.css';

const PAGE_SIZE = 24;
export default function ArchivedVersions({ archiveTabs }: { archiveTabs: React.ReactNode }) {
  const text = useWorkspaceText();
  const { i18n } = useTranslation();
  const client = useQueryClient();
  const active = React.useContext(ResourceActivityContext);
  const [params, setParams] = useSearchParams();
  const search = params.get('q') || '', projectFilter = params.get('project') || '';
  const view = params.get('view') === 'list' ? 'list' : 'grid';
  const sort = params.get('sort') === 'name' ? 'name' : params.get('sort') === 'created' ? 'created' : 'updated';
  const requestedPage = Math.max(1, Math.floor(Number(params.get('page')) || 1));
  const query = useQuery({ queryKey: ['archived-versions'], queryFn: ({ signal }) => archivedVersions(signal), enabled: active, refetchOnWindowFocus: false });
  const projects = useQuery({ queryKey: ['archive-projects'], queryFn: ({ signal }) => apiClient.get<GalleryProject[]>('/projects', { params: { include_archived: true }, signal, silent: true }), enabled: active, refetchOnWindowFocus: false });
  const families = useQuery({ queryKey: ['families'], queryFn: () => apiClient.get<FamilyInfo[]>('/families', { silent: true }), staleTime: 5 * 60 * 1000 });
  const rows = query.data || [];
  const [pending, setPending] = React.useState<string | null>(null);
  const pendingRef = React.useRef(false);
  const [failure, setFailure] = React.useState('');
  const [deleting, setDeleting] = React.useState<ArchivedVersion | null>(null);
  const deletingAny = rows.some(row => row.deletion?.state === 'deleting');
  const refresh = query.refetch;
  React.useEffect(() => {
    if (!deletingAny || !active) return;
    const timer = window.setInterval(() => { if (!document.hidden) void refresh({ cancelRefetch: false }); }, 2000);
    return () => window.clearInterval(timer);
  }, [deletingAny, active, refresh]);
  const changeFilter = (patch: Record<string, string | null>, replace = false) => {
    const next = new URLSearchParams(params); next.delete('page');
    Object.entries(patch).forEach(([key, value]) => value ? next.set(key, value) : next.delete(key));
    setParams(next, { replace });
  };
  const filtered = !!(search.trim() || projectFilter);
  const needle = search.trim().toLocaleLowerCase();
  const visible = rows.filter(row => (!projectFilter || row.project_id === projectFilter) && (!needle || `${row.name} ${row.id} ${row.note} ${row.project_name} ${row.project_id}`.toLocaleLowerCase().includes(needle)))
    .sort((a, b) => sort === 'name' ? a.name.localeCompare(b.name, i18n.resolvedLanguage) : (sort === 'created' ? b.created_at - a.created_at : b.updated_at - a.updated_at));
  const pages = Math.max(1, Math.ceil(visible.length / PAGE_SIZE));
  const page = Math.min(requestedPage, pages);
  const shown = visible.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  React.useEffect(() => {
    if (query.isPending || query.error || requestedPage <= pages) return;
    const next = new URLSearchParams(params); if (pages === 1) next.delete('page'); else next.set('page', String(pages));
    setParams(next, { replace: true });
  }, [query.isPending, query.error, requestedPage, pages, params, setParams]);
  const restore = async (row: ArchivedVersion) => {
    if (pendingRef.current || row.project_archived || row.deletion) return;
    pendingRef.current = true; setPending(row.id); setFailure('');
    try {
      const updated = await apiClient.patch<ProjectVersion>(`/projects/${row.project_id}/versions/${row.id}`, { archived: false }, { silent: true });
      client.setQueryData<ArchivedVersion[]>(['archived-versions'], previous => previous?.filter(item => item.id !== updated.id));
      client.setQueryData<ProjectVersion[]>(['project-versions', row.project_id], previous => previous?.map(item => item.id === updated.id ? updated : item));
      await Promise.all([query.refetch(), client.invalidateQueries({ queryKey: ['project', row.project_id] }), client.invalidateQueries({ queryKey: ['project-versions', row.project_id] })]);
    } catch (error) { setFailure(formatApiError(error)); }
    finally { pendingRef.current = false; setPending(null); }
  };
  const name = (row: ArchivedVersion) => row.name === `v${row.number}` ? row.name : `v${row.number} · ${row.name}`;
  const model = (row: ArchivedVersion) => row.project_type === 'tts'
    ? [text('语音', 'Speech'), ttsEngineLabel(row.engine), row.variant].filter(Boolean).join(' · ')
    : familyDisplayName(families.data, row.display_family || row.family || '');
  const counts = (row: ArchivedVersion) => [
    { label: text('训练数据', 'Training data'), value: row.project_type === 'tts' ? row.audio_stats?.train.state === 'valid' ? row.audio_stats.train.clips_count : null : row.stats.images, unit: row.project_type === 'tts' ? text('段音频', 'clips') : text('张图片', 'images') },
    { label: text('任务', 'Jobs'), value: row.stats.jobs, unit: text('个任务', 'jobs') },
    { label: text('产物', 'Outputs'), value: row.stats.artifacts, unit: text('个产物', 'outputs') },
  ];
  const date = (value: number) => <time dateTime={value > 0 ? new Date(value * 1000).toISOString() : undefined}>{value > 0 ? new Date(value * 1000).toLocaleDateString(i18n.resolvedLanguage, { month: 'short', day: 'numeric' }) : '—'}</time>;
  const menu = (row: ArchivedVersion) => <ProjectCardMenu scope="version" name={`${row.project_name} / ${name(row)}`} archived busy={!!pending || row.deletion?.state === 'deleting'} onArchive={() => void restore(row)} onDelete={() => setDeleting(row)} archiveDisabledReason={row.project_archived ? text('请先恢复所属项目。', 'Restore the parent project first.') : row.deletion ? text('请先完成删除。', 'Finish deleting this version first.') : undefined} parentArchiveUrl={row.project_archived ? archiveUrl('projects', row.project_id) : undefined}/>;
  const activity = (row: ArchivedVersion) => row.deletion ? <ProjectDeletionLine deletion={row.deletion}/> : <div className="project-activity" title={row.note}><span className="project-activity-detail">{row.note}</span></div>;
  const projectOptions = (projects.data || []).map(project => ({ value: project.id, label: project.name }));
  if (projectFilter && !projectOptions.some(option => option.value === projectFilter)) projectOptions.push({ value: projectFilter, label: rows.find(row => row.project_id === projectFilter)?.project_name || projectFilter });
  const error = failure || query.error && formatApiError(query.error);
  return <div className="projects-workspace archive-workspace" data-testid="archive-versions-page">
    <header className="projects-toolbar">
      <div className="projects-heading-row"><PageLocation trail={[{ label: text('项目', 'Projects'), to: '/projects' }, { label: text('归档', 'Archive') }]}/><div className="projects-page-heading"><h1>{text('归档', 'Archive')}</h1>{!query.isPending && <p>{filtered ? text(`显示 ${visible.length} / ${rows.length} 个版本`, `${visible.length} of ${rows.length} versions`) : text(`${rows.length} 个已归档版本`, `${rows.length} archived versions`)}</p>}</div><div className="projects-heading-actions"><button type="button" className="ui-btn ui-btn-icon" aria-label={text('刷新归档版本', 'Refresh archived versions')} title={text('刷新归档版本', 'Refresh archived versions')} disabled={query.isFetching || !!pending} onClick={() => { void query.refetch(); void projects.refetch(); }}><RefreshCw size={15}/></button></div></div>
      {archiveTabs}
      <div className="projects-filter-row">
        <label className="projects-search"><Search size={15}/><input type="text" aria-label={text('搜索归档版本', 'Search archived versions')} placeholder={text('版本名称、项目或备注', 'Version, project or note')} value={search} onChange={event => changeFilter({ q: event.target.value || null }, true)}/></label>
        <StudioSelect searchable aria-label={text('按项目筛选', 'Filter by project')} value={projectFilter} options={[{ value: '', label: text('所有项目', 'All projects') }, ...projectOptions]} onValueChange={value => changeFilter({ project: value || null })}/>
        <StudioSelect aria-label={text('版本排序', 'Version order')} value={sort} options={[{ value: 'updated', label: text('最近更新', 'Recently updated') }, { value: 'created', label: text('最近创建', 'Recently created') }, { value: 'name', label: text('名称顺序', 'Name') }]} onValueChange={value => changeFilter({ sort: value === 'updated' ? null : value })}/>
        <Link className="projects-archive-filter ui-btn" to="/projects"><Archive size={14}/>{text('返回项目', 'Back to projects')}</Link>
        <div className="projects-view-toggle ui-segmented ui-segmented-sm"><button type="button" aria-label={text('卡片视图', 'Grid view')} title={text('卡片视图', 'Grid view')} aria-pressed={view === 'grid'} onClick={() => changeFilter({ view: null })}><LayoutGrid size={15}/></button><button type="button" aria-label={text('列表视图', 'List view')} title={text('列表视图', 'List view')} aria-pressed={view === 'list'} onClick={() => changeFilter({ view: 'list' })}><List size={15}/></button><SlidingIndicator className="ui-segmented-thumb"/></div>
      </div>
      {error && <div role="alert" className="projects-list-error"><span>{error}</span><button type="button" className="ui-btn ui-btn-sm" onClick={() => { setFailure(''); void query.refetch(); }}>{text('重试', 'Retry')}</button></div>}
      {projects.error && <div role="alert" className="projects-list-error"><span>{text('项目筛选列表读取失败。', 'The project filter could not load.')}</span><button type="button" className="ui-btn ui-btn-sm" onClick={() => void projects.refetch()}>{text('重试', 'Retry')}</button></div>}
    </header>
    {query.isPending ? <div className="projects-gallery projects-skeleton" role="status" aria-label={text('读取归档版本…', 'Loading archived versions…')}>{[0, 1, 2, 3].map(index => <div className="project-card" key={index}><div className="project-card-media ui-skeleton"/><div className="project-card-body"><span className="ui-skeleton"/><span className="ui-skeleton"/></div></div>)}</div>
      : shown.length ? view === 'grid' ? <div className="projects-gallery" role="list" aria-label={text('归档版本列表', 'Archived version list')}>{shown.map(row => <article key={row.id} className="project-card" role="listitem" data-archived data-testid={`archive-version-${row.id}`}><Link className="project-card-link" to={projectUrl(row.project_id, row.id)} aria-label={text(`查看版本：${row.project_name} / ${name(row)}`, `View version: ${row.project_name} / ${name(row)}`)}><div className="project-card-media"><ProjectArtwork name={row.project_name} coverUrl={row.project_cover_url}/><span className="project-card-flag">{text('已归档', 'Archived')}</span></div><div className="project-card-body"><strong className="project-card-name" title={name(row)}>{name(row)}</strong><p className="archive-version-project" title={row.project_name}>{row.project_name}</p><div className="project-card-meta"><span>{model(row)}</span>{row.project_archived && <span>{text('项目已归档', 'Project archived')}</span>}</div>{(row.deletion || row.note) && activity(row)}<div className="project-card-footer"><div className="project-stats">{counts(row).map(item => <span key={item.label} aria-label={`${item.label}: ${item.value ?? '—'}`}>{item.value == null ? `${item.label} —` : `${item.value} ${item.unit}`}</span>)}</div>{date(row.updated_at)}</div></div></Link>{menu(row)}</article>)}</div>
        : <div className="projects-rows" role="list" aria-label={text('归档版本列表', 'Archived version list')}><div className="project-row-head" aria-hidden="true"><span/><span>{text('版本与项目', 'Version & project')}</span><span>{text('模型', 'Model')}</span><span>{text('备注', 'Note')}</span><span className="project-row-number">{text('训练数据', 'Training data')}</span><span className="project-row-number">{text('任务', 'Jobs')}</span><span className="project-row-number">{text('产物', 'Outputs')}</span><span>{text('更新', 'Updated')}</span><span/></div>{shown.map(row => <article key={row.id} className="project-row" role="listitem" data-archived data-testid={`archive-version-${row.id}`}><Link className="project-row-link" to={projectUrl(row.project_id, row.id)} aria-label={text(`查看版本：${row.project_name} / ${name(row)}`, `View version: ${row.project_name} / ${name(row)}`)}><div className="project-row-thumb"><ProjectArtwork name={row.project_name} coverUrl={row.project_cover_url}/></div><div className="project-row-name"><strong title={name(row)}>{name(row)}</strong><small title={row.project_name}>{row.project_name}</small></div><div className="project-row-meta"><span>{model(row)}</span>{row.project_archived && <small>{text('项目已归档', 'Project archived')}</small>}</div>{activity(row)}{counts(row).map(item => <span className="project-row-number" key={item.label} title={item.label} aria-label={`${item.label}: ${item.value ?? '—'}`}>{item.value ?? '—'}</span>)}{date(row.updated_at)}</Link>{menu(row)}</article>)}</div>
      : !query.error && <div className="projects-empty-state"><span aria-hidden="true">{filtered ? <SearchX size={28}/> : <Archive size={28}/>}</span><h3>{filtered ? text('没有匹配的归档版本', 'No matching archived versions') : text('暂无归档版本', 'No archived versions')}</h3>{filtered && <button type="button" className="ui-btn" onClick={() => changeFilter({ q: null, project: null })}>{text('清除筛选', 'Clear filters')}</button>}</div>}
    {!query.isPending && pages > 1 && <nav className="projects-pagination" aria-label={text('版本分页', 'Version pagination')}><span>{text(`共 ${visible.length} 个版本 · 每页 ${PAGE_SIZE} 个`, `${visible.length} versions · ${PAGE_SIZE} per page`)}</span><div><button type="button" className="ui-btn ui-btn-sm" aria-label={text('上一页', 'Previous')} disabled={page <= 1} onClick={() => changeFilter({ page: String(page - 1) })}><ChevronLeft size={15}/>{text('上一页', 'Previous')}</button><span>{page} / {pages}</span><button type="button" className="ui-btn ui-btn-sm" aria-label={text('下一页', 'Next')} disabled={page >= pages} onClick={() => changeFilter({ page: String(page + 1) })}>{text('下一页', 'Next')}<ChevronRight size={15}/></button></div></nav>}
    {deleting && <ProjectDeleteDialog project={{ id: deleting.project_id, name: deleting.project_name }} version={deleting} onClose={() => setDeleting(null)} onStarted={() => { setDeleting(null); void query.refetch(); }}/>}
  </div>;
}
