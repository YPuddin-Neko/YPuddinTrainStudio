import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { FolderPlus, Archive, Search, FolderOpen, SearchX, ChevronLeft, ChevronRight, RefreshCw, LayoutGrid, List } from 'lucide-react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { apiClient } from '../../api/client';
import type { FamilyInfo } from '../../api/types';
import StudioSelect from '../../components/StudioSelect';
import { formatApiError } from '../../utils/errors';
import { ttsEngineLabel } from '../../utils/ttsEngines';
import { useWorkspaceText } from '../../utils/workspaceText';
import ProjectEditor from './ProjectEditor';
import { categoryLabel, type GalleryProject } from './projectGallery';
import ProjectCardMenu from './ProjectCardMenu';
import { ProjectActivityLine, ProjectArtwork, ProjectDeletionLine } from './ProjectCardParts';
import ProjectDeleteDialog from './ProjectDeleteDialog';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import '../../styles/project-workspace.css';
import './projects.css';
import Switch from '../../components/Switch';
import { SlidingIndicator } from '../../components/motion';
import PageLocation from '../../components/PageLocation';

const PAGE_SIZE = 24;
const ACTIVE = ['running', 'pausing', 'cancelling'];
// Shown until the registered family labels arrive.
const FAMILY_NAMES: Record<string, string> = { anima: 'Anima', krea2: 'Krea 2', sdxl: 'SDXL', flux: 'FLUX.1', flux2: 'FLUX.2 Klein', toy: 'Toy' };

export default function Projects() {
  const { t, i18n } = useTranslation();
  const queryClient = useQueryClient();
  const english = i18n.resolvedLanguage?.startsWith('en') || false;
  const text = useWorkspaceText();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [error, setError] = React.useState('');
  const [projects, setProjects] = React.useState<GalleryProject[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [editor, setEditor] = React.useState<GalleryProject | 'new' | null>(null);
  const [pending, setPending] = React.useState<string | null>(null);
  const [deleting, setDeleting] = React.useState<GalleryProject | null>(null);
  const pendingRef = React.useRef<string | null>(null);
  const families = useQuery({ queryKey: ['families'], queryFn: () => apiClient.get<FamilyInfo[]>('/families', { silent: true }), staleTime: 5 * 60 * 1000 });
  const search = params.get('q') || '';
  const category = params.get('category') || '';
  const uncategorized = params.get('uncategorized') === 'true';
  const showArchived = params.get('archived') === '1';
  const view = params.get('view') === 'list' ? 'list' : 'grid';
  const sort = params.get('sort') === 'name' ? 'name' : params.get('sort') === 'created' ? 'created' : 'updated';
  const requestedPage = Math.max(1, Math.floor(Number(params.get('page')) || 1));
  const changeFilter = (patch: Record<string, string | null>, replace = false) => {
    const next = new URLSearchParams(params); next.delete('page');
    Object.entries(patch).forEach(([key, value]) => value ? next.set(key, value) : next.delete(key));
    setParams(next, { replace });
  };
  // `quiet` refreshes keep the cards on screen, e.g. while a deletion runs.
  const fetchProjects = React.useCallback((quiet = false) => {
    if (!quiet) setLoading(true);
    return apiClient.get<{ items: GalleryProject[] } | GalleryProject[]>('/projects', { params: { include_archived: true }, silent: true })
      .then(data => { setProjects(Array.isArray(data) ? data : data.items || []); setError(''); })
      .catch(failure => setError(formatApiError(failure))).finally(() => { if (!quiet) setLoading(false); });
  }, []);
  React.useEffect(() => { void fetchProjects(); }, [fetchProjects]);
  const deletionRunning = projects.some(project => project.deletion?.state === 'deleting');
  React.useEffect(() => {
    if (!deletionRunning) return;
    const timer = window.setInterval(() => { void fetchProjects(true); }, 2000);
    return () => window.clearInterval(timer);
  }, [deletionRunning, fetchProjects]);
  useEventStream<{ kind?: string; state?: string }>(EVENT_TYPES.BACKGROUND_CHANGED, task => {
    if (task.kind === 'project_delete' && task.state !== 'running') void fetchProjects(true);
  });
  const updateProject = (project: GalleryProject) => {
    queryClient.setQueryData(['project',project.id],project);
    setProjects(rows => rows.some(row => row.id === project.id)
      ? rows.map(row => row.id === project.id ? project : row) : [project, ...rows]);
  };
  const mutate = async (id: string, operation: () => Promise<unknown>) => {
    if (pendingRef.current) return;
    pendingRef.current = id; setPending(id); setError('');
    try { await operation(); await Promise.all([fetchProjects(),queryClient.invalidateQueries({queryKey:['project',id]})]); }
    catch (failure) { setError(formatApiError(failure)); }
    finally { pendingRef.current = null; setPending(null); }
  };
  const remove = (project: GalleryProject) => {
    if (!project.archived || pendingRef.current || project.deletion?.state === 'deleting') return;
    setDeleting(project);
  };
  const categories = [...new Set(projects.map(project => project.category?.trim()).filter((value): value is string => Boolean(value)))].sort((a, b) => a.localeCompare(b));
  const categoryCount = (value: string | null) => projects.filter(project => (project.category || null) === value).length;
  const categoryOptions = [...new Set([...categories, ...(category ? [category] : [])])];
  const query = search.trim().toLocaleLowerCase();
  const filtered = !!(query || category || uncategorized);
  const visibleProjects = projects.filter(project => {
    if (!showArchived && project.archived) return false;
    if (uncategorized ? Boolean(project.category) : category && project.category !== category) return false;
    return !query || `${project.name} ${project.id} ${project.note || ''}`.toLocaleLowerCase().includes(query);
  }).sort((left, right) => sort === 'name' ? left.name.localeCompare(right.name, i18n.resolvedLanguage) : (Number(sort === 'created' ? right.created_at : right.updated_at) || 0) - (Number(sort === 'created' ? left.created_at : left.updated_at) || 0));
  const pageCount = Math.max(1, Math.ceil(visibleProjects.length / PAGE_SIZE));
  const page = Math.min(requestedPage, pageCount);
  const pageProjects = visibleProjects.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  React.useEffect(() => {
    if (loading || error || requestedPage <= pageCount) return;
    const next = new URLSearchParams(params);
    if (pageCount === 1) next.delete('page'); else next.set('page', String(pageCount));
    setParams(next, { replace: true });
  }, [loading, error, requestedPage, pageCount, params, setParams]);

  const familyLabel = (name?: string | null) => name ? (Array.isArray(families.data) && families.data.find(family => family.name === name)?.label) || FAMILY_NAMES[name] || name : '';
  const versionLabel = (project: GalleryProject) => {
    const number = project.active_version_number ? `v${project.active_version_number}` : '';
    const name = project.active_version_name?.trim() || '';
    return number && name && name !== number ? `${number} · ${name}` : name || number;
  };
  const updated = (value: number) => {
    const date = value > 0 ? new Date(value * 1000) : null;
    const valid = date !== null && Number.isFinite(date.getTime());
    return <time dateTime={valid ? date.toISOString() : undefined}>{valid ? date.toLocaleDateString(i18n.resolvedLanguage || 'zh-CN', { month: 'short', day: 'numeric' }) : '—'}</time>;
  };
  const counts = (project: GalleryProject) => [
    project.project_type === 'tts'
      ? { key: 'audio', label: text('训练音频', 'Training audio'), value: project.audio_stats?.train.clips_count, unit: text('段音频', 'clips') }
      : { key: 'images', label: text('训练图片', 'Training images'), value: project.image_count, unit: text('张图片', 'images') },
    { key: 'versions', label: text('版本', 'Versions'), value: project.version_count, unit: text('个版本', 'versions') },
    { key: 'outputs', label: t('projects.artifacts'), value: project.stats?.artifacts, unit: text('个产物', 'outputs') },
  ];
  const training = projects.filter(project => !project.archived && ACTIVE.includes(project.latest_job?.status || '')).length;
  const archivedCount = projects.filter(project => project.archived).length;
  const summary = loading ? '' : filtered || (showArchived && archivedCount)
    ? text(`显示 ${visibleProjects.length} / ${projects.length} 个项目`, `${visibleProjects.length} of ${projects.length} projects`)
    : [text(`${projects.length - archivedCount} 个项目`, `${projects.length - archivedCount} projects`), training > 0 && text(`${training} 个训练中`, `${training} training`), archivedCount > 0 && text(`${archivedCount} 个已归档`, `${archivedCount} archived`)].filter(Boolean).join(' · ');
  const open = (project: GalleryProject) => `/projects/${encodeURIComponent(project.id)}?step=overview`;
  const menu = (project: GalleryProject) => <ProjectCardMenu name={project.name} archived={project.archived} busy={!!pending || project.deletion?.state === 'deleting'} onEdit={() => setEditor(project)} onArchive={() => void mutate(project.id, () => apiClient.patch(`/projects/${project.id}`, { archived: !project.archived }))} onDelete={() => remove(project)}/>;
  const activity = (project: GalleryProject) => project.deletion ? <ProjectDeletionLine deletion={project.deletion}/> : <ProjectActivityLine job={project.latest_job}/>;
  const modelLabel = (project: GalleryProject) => project.project_type === 'tts'
    ? [text('语音', 'Speech'), project.active_engine ? ttsEngineLabel(project.active_engine) : null].filter(Boolean).join(' · ')
    : familyLabel(project.active_display_family ?? project.active_family);
  const meta = (project: GalleryProject) => [project.category ? categoryLabel(project.category, english) : text('未分类', 'Uncategorized'), modelLabel(project), versionLabel(project)].filter(Boolean);
  const mixedDataTypes = pageProjects.some(project => project.project_type === 'tts') && pageProjects.some(project => project.project_type !== 'tts');

  return <div className="projects-workspace" data-testid="projects-page">
    <header className="projects-toolbar">
      <div className="projects-heading-row">
        <PageLocation trail={[{ label: text('项目', 'Projects') }]}/><div className="projects-page-heading"><h1>{t('projects.title')}</h1>{summary && <p>{summary}</p>}</div>
        <div className="projects-heading-actions"><button type="button" aria-label={text('刷新项目', 'Refresh projects')} disabled={loading || !!pending} onClick={() => void fetchProjects()} className="ui-btn ui-btn-icon" title={text('刷新项目', 'Refresh projects')}><RefreshCw size={15}/></button><button type="button" onClick={() => setEditor('new')} className="ui-btn ui-btn-primary"><FolderPlus size={15}/><span>{t('projects.newProject')}</span></button></div>
      </div>
      <div className="projects-filter-row">
        <label className="projects-search"><Search size={15}/><input type="text" value={search} onChange={event => changeFilter({ q: event.target.value || null }, true)} aria-label={text('搜索项目', 'Search projects')} placeholder={text('项目名称或备注', 'Project name or note')} data-testid="project-search-input"/></label>
        <StudioSelect className="projects-category-filter" aria-label={text('按分类筛选', 'Filter by category')} value={uncategorized ? 'uncategorized' : category ? `category:${category}` : ''}
          options={[{ value: '', label: text('所有分类', 'All categories') }, { value: 'uncategorized', label: `${text('未分类', 'Uncategorized')} · ${categoryCount(null)}` }, ...categoryOptions.map(value => ({ value: `category:${value}`, label: `${categoryLabel(value, english)} · ${categoryCount(value)}` }))]}
          onValueChange={value => changeFilter({ category: value.startsWith('category:') ? value.slice(9) : null, uncategorized: value === 'uncategorized' ? 'true' : null })}/>
        <StudioSelect className="projects-sort-filter" aria-label={text('项目排序', 'Sort projects')} value={sort} options={[{ value: 'updated', label: text('最近更新', 'Recently updated') }, { value: 'created', label: text('最近创建', 'Recently created') }, { value: 'name', label: text('名称排序', 'Name') }]} onValueChange={value => changeFilter({ sort: value === 'updated' ? null : value })}/>
        <Switch className="projects-archive-filter" checked={showArchived} onCheckedChange={checked => changeFilter({ archived: checked ? '1' : null })} data-testid="show-archived-toggle">{t('projects.showArchived', '显示已归档')}</Switch>
        <div className="projects-view-toggle ui-segmented" role="group" aria-label={text('项目显示方式', 'Project view')}><button type="button" aria-label={text('卡片视图', 'Grid view')} title={text('卡片视图', 'Grid view')} aria-pressed={view === 'grid'} onClick={() => changeFilter({ view: null })}><LayoutGrid size={15}/></button><button type="button" aria-label={text('列表视图', 'List view')} title={text('列表视图', 'List view')} aria-pressed={view === 'list'} onClick={() => changeFilter({ view: 'list' })}><List size={15}/></button><SlidingIndicator className="ui-segmented-thumb"/></div>
      </div>
      {error && <div role="alert" className="projects-list-error"><span>{error}</span><button type="button" className="ui-btn ui-btn-sm" onClick={() => void fetchProjects()} disabled={loading}>{t('common.retry')}</button></div>}
    </header>
    {loading ? <div className="projects-gallery projects-skeleton" data-testid="projects-skeleton" role="status" aria-label={text('读取项目…', 'Loading projects…')}>{[0, 1, 2, 3].map(index => <div className="project-card" key={index}><div className="project-card-media ui-skeleton"/><div className="project-card-body"><span className="ui-skeleton"/><span className="ui-skeleton"/></div></div>)}</div>
      : visibleProjects.length ? view === 'grid'
        ? <div className="projects-gallery" role="list" aria-label={text('项目列表', 'Project list')}>{pageProjects.map(project => <article key={project.id} role="listitem" className="project-card" data-archived={project.archived || undefined} data-testid={`project-card-${project.id}`}>
          <Link to={open(project)} className="project-card-link" aria-label={text(`打开项目：${project.name}`, `Open project: ${project.name}`)}>
            <div className="project-card-media"><ProjectArtwork name={project.name} coverUrl={project.cover_url}/>{project.archived && <span className="project-card-flag">{t('projects.archived')}</span>}</div>
            <div className="project-card-body">
              <strong className="project-card-name" title={project.name}>{project.name}</strong>
              <div className="project-card-meta">{meta(project).map((item, index) => <span key={index} className={index === 0 ? 'project-category' : undefined} title={item}>{item}</span>)}</div>
              {project.note?.trim() && <p className="project-card-note" title={project.note}>{project.note}</p>}
              {activity(project)}
              <div className="project-card-footer"><div className="project-stats">{counts(project).map(item => <span key={item.key} aria-label={`${item.label}: ${item.value ?? '—'}`}>{item.value == null ? `${item.label} —` : `${item.value} ${item.unit}`}</span>)}</div>{updated(project.updated_at)}</div>
            </div>
          </Link>
          {menu(project)}
        </article>)}</div>
        : <div className="projects-rows" role="list" aria-label={text('项目列表', 'Project list')}>
          <div className="project-row-head" aria-hidden="true"><span/><span>{text('项目', 'Project')}</span><span>{text('分类与模型', 'Category & model')}</span><span>{text('最近训练', 'Latest training')}</span>{counts(pageProjects[0]).map((item, index) => <span key={item.key} className="project-row-number">{index === 0 && mixedDataTypes ? text('训练数据', 'Training data') : item.label}</span>)}<span>{text('更新', 'Updated')}</span><span/></div>
          {pageProjects.map(project => <article key={project.id} role="listitem" className="project-row" data-archived={project.archived || undefined} data-testid={`project-card-${project.id}`}>
            <Link to={open(project)} className="project-row-link" aria-label={text(`打开项目：${project.name}`, `Open project: ${project.name}`)}>
              <div className="project-row-thumb"><ProjectArtwork name={project.name} coverUrl={project.cover_url}/></div>
              <div className="project-row-name"><strong title={project.name}>{project.name}</strong>{project.archived ? <small>{t('projects.archived')}</small> : project.note?.trim() && <small title={project.note}>{project.note}</small>}</div>
              <div className="project-row-meta"><span className="project-category">{meta(project)[0]}</span><small title={meta(project).slice(1).join(' · ')}>{meta(project).slice(1).join(' · ')}</small></div>
              {activity(project)}
              {counts(project).map(item => <span key={item.key} className="project-row-number" aria-label={`${item.label}: ${item.value ?? '—'}`}>{item.value ?? '—'}</span>)}
              {updated(project.updated_at)}
            </Link>
            {menu(project)}
          </article>)}
        </div>
      : !error && <div className="projects-empty-state" data-testid={filtered ? 'projects-no-results' : projects.length ? 'projects-all-archived' : 'projects-empty'}>
        {filtered ? <SearchX size={28}/> : projects.length ? <Archive size={28}/> : <FolderOpen size={28}/>}
        <h3>{filtered ? text('没有匹配的项目', 'No matching projects') : projects.length ? t('projects.allArchivedTitle', '所有项目都已归档') : t('projects.emptyTitle', '还没有项目')}</h3>
        {filtered && <button type="button" className="ui-btn" onClick={() => changeFilter({ q: null, category: null, uncategorized: null })}>{text('清除筛选', 'Clear filters')}</button>}
        {!filtered && projects.length > 0 && !showArchived && <button type="button" className="ui-btn" onClick={() => changeFilter({ archived: '1' })}>{text('查看已归档项目', 'View archived projects')}</button>}
        {!filtered && projects.length === 0 && <button type="button" className="ui-btn ui-btn-primary" onClick={() => setEditor('new')}><FolderPlus size={15}/>{text('创建第一个项目', 'Create your first project')}</button>}
      </div>}
    {!loading && pageCount > 1 && <nav className="projects-pagination" aria-label={text('项目分页', 'Project pagination')}><span>{text(`共 ${visibleProjects.length} 个项目 · 每页 ${PAGE_SIZE} 个`, `${visibleProjects.length} projects · ${PAGE_SIZE} per page`)}</span><div><button type="button" className="ui-btn ui-btn-sm" aria-label={text('上一页', 'Previous')} disabled={page <= 1} onClick={() => changeFilter({ page: String(page - 1) })}><ChevronLeft size={15}/>{text('上一页', 'Previous')}</button><span aria-label={text('当前页', 'Current page')}>{page} / {pageCount}</span><button type="button" className="ui-btn ui-btn-sm" aria-label={text('下一页', 'Next')} disabled={page >= pageCount} onClick={() => changeFilter({ page: String(page + 1) })}>{text('下一页', 'Next')}<ChevronRight size={15}/></button></div></nav>}
    {deleting && <ProjectDeleteDialog project={deleting} onClose={() => setDeleting(null)} onStarted={() => { setDeleting(null); void fetchProjects(true); }}/>}
    {editor && <ProjectEditor project={editor === 'new' ? undefined : editor} categories={categories} onClose={() => setEditor(null)} onPartial={updateProject} onSaved={project => { updateProject(project); setEditor(null); if (editor === 'new') navigate(`/projects/${encodeURIComponent(project.id)}?step=overview`); }}/>}
  </div>;
}
