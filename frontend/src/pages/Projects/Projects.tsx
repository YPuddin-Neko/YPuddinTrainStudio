import React from 'react';
import { useTranslation } from 'react-i18next';
import { FolderPlus, GitBranch, Archive, Search, FolderOpen, SearchX, Database, Activity, Box, ChevronLeft, ChevronRight, RefreshCw, LayoutGrid, List, ArrowUpRight } from 'lucide-react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { apiClient } from '../../api/client';
import StudioSelect from '../../components/StudioSelect';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import ProjectEditor, { ProjectCover } from './ProjectEditor';
import { categoryLabel, coverSource, type GalleryProject } from './projectGallery';
import ProjectCardMenu from './ProjectCardMenu';
import '../../styles/project-workspace.css';
import './projects.css';

export default function Projects() {
  const { t, i18n } = useTranslation();
  const english = i18n.resolvedLanguage?.startsWith('en') || false;
  const text = useWorkspaceText();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [error, setError] = React.useState('');
  const [projects, setProjects] = React.useState<GalleryProject[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [editor, setEditor] = React.useState<GalleryProject | 'new' | null>(null);
  const [pending, setPending] = React.useState<string | null>(null);
  const pendingRef = React.useRef<string | null>(null);
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
  const fetchProjects = React.useCallback(() => {
    setLoading(true);
    return apiClient.get<{ items: GalleryProject[] } | GalleryProject[]>('/projects', { params: { include_archived: true }, silent: true })
      .then(data => { setProjects(Array.isArray(data) ? data : data.items || []); setError(''); })
      .catch(failure => setError(formatApiError(failure))).finally(() => setLoading(false));
  }, []);
  React.useEffect(() => { void fetchProjects(); }, [fetchProjects]);
  const updateProject = (project: GalleryProject) => setProjects(rows => rows.some(row => row.id === project.id)
    ? rows.map(row => row.id === project.id ? project : row) : [project, ...rows]);
  const mutate = async (id: string, operation: () => Promise<unknown>) => {
    if (pendingRef.current) return;
    pendingRef.current = id; setPending(id); setError('');
    try { await operation(); await fetchProjects(); }
    catch (failure) { setError(formatApiError(failure)); }
    finally { pendingRef.current = null; setPending(null); }
  };
  const remove = (project: GalleryProject) => {
    if (!project.archived || pendingRef.current) return;
    if (window.confirm(t('projects.deleteConfirm').replace('{name}', project.name))) void mutate(project.id, () => apiClient.delete(`/projects/${project.id}`, { params: { delete_files: true } }));
  };
  const categories = [...new Set(projects.map(project => project.category?.trim()).filter((value): value is string => Boolean(value)))].sort((a, b) => a.localeCompare(b));
  const categoryCount = (value: string | null) => projects.filter(project => (project.category || null) === value).length;
  const categoryOptions = [...new Set([...categories, ...(category ? [category] : [])])];
  const query = search.trim().toLocaleLowerCase();
  const visibleProjects = projects.filter(project => {
    if (!showArchived && project.archived) return false;
    if (uncategorized ? Boolean(project.category) : category && project.category !== category) return false;
    return !query || `${project.name} ${project.id} ${project.note || ''}`.toLocaleLowerCase().includes(query);
  }).sort((left, right) => sort === 'name' ? left.name.localeCompare(right.name, i18n.resolvedLanguage) : (Number(sort === 'created' ? right.created_at : right.updated_at) || 0) - (Number(sort === 'created' ? left.created_at : left.updated_at) || 0));
  const pageCount = Math.max(1, Math.ceil(visibleProjects.length / 24));
  const page = Math.min(requestedPage, pageCount);
  const pageProjects = visibleProjects.slice((page - 1) * 24, page * 24);
  React.useEffect(() => {
    if (loading || error || requestedPage <= pageCount) return;
    const next = new URLSearchParams(params);
    if (pageCount === 1) next.delete('page'); else next.set('page', String(pageCount));
    setParams(next, { replace: true });
  }, [loading, error, requestedPage, pageCount, params, setParams]);
  const stat = (icon: React.ReactNode, label: string, value: number | undefined) => <span className="project-card-stat" aria-label={`${label}: ${value ?? '—'}`} title={`${label}: ${value ?? '—'}`}>{icon}<span>{label}</span><b>{value ?? '—'}</b></span>;
  const updated = (value: number) => value > 0 ? new Date(value * 1000).toLocaleDateString(i18n.resolvedLanguage || 'zh-CN', { month: 'short', day: 'numeric' }) : '—';

  return <div className="projects-workspace" data-testid="projects-page">
    <header className="projects-toolbar">
      <div className="projects-heading-row"><div className="projects-page-heading"><h2>{t('projects.title')}{!loading && <span>{visibleProjects.length === projects.length ? projects.length : `${visibleProjects.length}/${projects.length}`}</span>}</h2><p>{text('按项目组织数据、版本与训练结果。', 'Organize datasets, versions and training results by project.')}</p></div>
        <div className="projects-heading-actions"><button type="button" aria-label={text('刷新项目', 'Refresh projects')} disabled={loading || !!pending} onClick={() => void fetchProjects()} className="projects-page-button"><RefreshCw size={14}/></button><button type="button" onClick={() => setEditor('new')} className="projects-create-button"><FolderPlus size={14}/><span>{t('projects.newProject')}</span></button></div></div>
      <div className="projects-filter-row">
        <label className="projects-search"><Search size={14}/><input type="text" value={search} onChange={event => changeFilter({ q: event.target.value || null }, true)} aria-label={text('搜索项目', 'Search projects')} placeholder={t('projects.searchPlaceholder', '按名称 / 备注搜索…')} data-testid="project-search-input"/></label>
        <StudioSelect className="projects-category-filter" aria-label={text('按分类筛选', 'Filter by category')} value={uncategorized ? 'uncategorized' : category ? `category:${category}` : ''}
          options={[{ value: '', label: text('所有分类', 'All categories') }, { value: 'uncategorized', label: `${text('未分类', 'Uncategorized')} · ${categoryCount(null)}` }, ...categoryOptions.map(value => ({ value: `category:${value}`, label: `${categoryLabel(value, english)} · ${categoryCount(value)}` }))]}
          onValueChange={value => changeFilter({ category: value.startsWith('category:') ? value.slice(9) : null, uncategorized: value === 'uncategorized' ? 'true' : null })}/>
        <label className="projects-archive-filter"><input type="checkbox" checked={showArchived} onChange={event => changeFilter({ archived: event.target.checked ? '1' : null })} data-testid="show-archived-toggle"/><span>{t('projects.showArchived', '显示已归档')}</span></label>
        <StudioSelect className="projects-sort-filter" aria-label={text('项目排序', 'Sort projects')} value={sort} options={[{ value: 'updated', label: text('最近更新', 'Recently updated') }, { value: 'created', label: text('最近创建', 'Recently created') }, { value: 'name', label: text('名称排序', 'Name') }]} onValueChange={value => changeFilter({ sort: value === 'updated' ? null : value })}/>
        <div className="projects-view-toggle" role="group" aria-label={text('项目显示方式', 'Project view')}><button type="button" aria-label={text('卡片视图', 'Grid view')} aria-pressed={view === 'grid'} onClick={() => changeFilter({ view: null })}><LayoutGrid size={15}/></button><button type="button" aria-label={text('列表视图', 'List view')} aria-pressed={view === 'list'} onClick={() => changeFilter({ view: 'list' })}><List size={15}/></button></div>
        {pageCount > 1 && <nav className="projects-pagination" aria-label={text('项目分页', 'Project pagination')} title={text(`共 ${visibleProjects.length} 个项目 · 每页 24 个`, `${visibleProjects.length} projects · 24 per page`)}><button className="projects-page-button" aria-label={text('上一页', 'Previous')} disabled={loading || page <= 1} onClick={() => changeFilter({ page: String(page - 1) })}><ChevronLeft size={15}/></button><span aria-label={text('当前页', 'Current page')}>{page} / {pageCount}</span><button className="projects-page-button" aria-label={text('下一页', 'Next')} disabled={loading || page >= pageCount} onClick={() => changeFilter({ page: String(page + 1) })}><ChevronRight size={15}/></button></nav>}
      </div>
      {error && <div role="alert" className="projects-list-error"><span>{error}</span><button onClick={() => void fetchProjects()} disabled={loading}>{t('common.retry')}</button></div>}
    </header>
    {loading ? <div className="projects-gallery projects-skeleton" data-testid="projects-skeleton" role="status" aria-label={text('读取项目…', 'Loading projects…')}>{[0, 1, 2].map(index => <div className="project-preview-card" key={index}><div className="project-card-cover"/><div className="project-card-content"><span/><span/></div></div>)}</div>
      : visibleProjects.length ? <div className={`projects-gallery${view === 'list' ? ' projects-list-view' : ''}`} role="list" aria-label={text('项目列表', 'Project list')}>{pageProjects.map(project => <article key={project.id} role="listitem" className={`project-preview-card${project.archived ? ' project-preview-archived' : ''}`} data-testid={`project-card-${project.id}`}>
        <Link to={`/projects/${encodeURIComponent(project.id)}?step=overview`} className="project-card-open" aria-label={text(`打开项目：${project.name}`, `Open project: ${project.name}`)}>
          <div className="project-card-cover"><ProjectCover source={project.cover_url ? coverSource(project.cover_url) : null} name={project.name}/>{project.archived && <span className="project-card-archived">{t('projects.archived')}</span>}</div>
          <div className="project-card-content"><div className="project-card-title"><strong title={project.name}>{project.name}</strong></div>
            <div className="project-card-classification"><span className="project-category" title={project.category || undefined}>{project.category ? categoryLabel(project.category, english) : text('未分类', 'Uncategorized')}</span>{project.active_family && <span className="project-family">{project.active_family === 'krea2' ? 'Krea 2' : project.active_family === 'anima' ? 'Anima' : project.active_family}</span>}</div>
            {project.note?.trim() && <p className="project-card-note" title={project.note}>{project.note}</p>}
            <div className="project-card-stats">{stat(<GitBranch size={13}/>, text('版本', 'Versions'), project.version_count)}{stat(<Database size={13}/>, t('projects.datasets'), project.dataset_ids?.length)}{stat(<Activity size={13}/>, t('projects.jobs'), project.stats?.jobs)}{stat(<Box size={13}/>, t('projects.artifacts'), project.stats?.artifacts)}</div>
            <div className="project-card-footer"><span>{text('更新于', 'Updated')} {updated(project.updated_at)}</span><span>{text('打开概览', 'Open overview')}<ArrowUpRight size={13}/></span></div>
          </div>
        </Link>
        <ProjectCardMenu name={project.name} archived={project.archived} busy={!!pending} onEdit={() => setEditor(project)} onArchive={() => void mutate(project.id, () => apiClient.patch(`/projects/${project.id}`, { archived: !project.archived }))} onDelete={() => remove(project)}/>
      </article>)}</div>
      : !error && <div className="projects-empty-state" data-testid={query || category || uncategorized ? 'projects-no-results' : projects.length ? 'projects-all-archived' : 'projects-empty'}>
        {query || category || uncategorized ? <SearchX size={28}/> : projects.length ? <Archive size={28}/> : <FolderOpen size={28}/>}
        <h3>{query || category || uncategorized ? text('没有匹配的项目', 'No matching projects') : projects.length ? t('projects.allArchivedTitle', '所有项目都已归档') : t('projects.emptyTitle', '还没有项目')}</h3>
        <p>{query || category || uncategorized ? text('试试其他分类或关键词。', 'Try another category or search.') : projects.length ? t('projects.allArchivedHint', '勾选上方「显示已归档」查看。') : text('新建项目后可手动上传封面，导入训练数据。', 'Create a project, choose its cover and import training data.')}</p>
        {(query || category || uncategorized) && <button className="projects-page-button" onClick={() => changeFilter({ q: null, category: null, uncategorized: null })}>{text('清除筛选', 'Clear filters')}</button>}
        {!query && !category && !uncategorized && projects.length > 0 && !showArchived && <button className="projects-page-button" onClick={() => changeFilter({ archived: '1' })}>{text('查看已归档项目', 'View archived projects')}</button>}
        {!query && !category && !uncategorized && projects.length === 0 && <button className="projects-create-button" onClick={() => setEditor('new')}><FolderPlus size={15}/>{text('创建第一个项目', 'Create your first project')}</button>}
      </div>}
    {editor && <ProjectEditor project={editor === 'new' ? undefined : editor} categories={categories} onClose={() => setEditor(null)} onPartial={updateProject} onSaved={project => { updateProject(project); setEditor(null); if (editor === 'new') navigate(`/projects/${encodeURIComponent(project.id)}?step=overview`); }}/>}
  </div>;
}
