import React from 'react';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { Project } from '../../api/types';
import {
  FolderPlus,
  GitBranch,
  Trash2,
  Archive,
  Pencil,
  Search,
  FolderOpen,
  SearchX,
  Database,
  Activity,
  Box,
  Clock,
  RefreshCw,
} from 'lucide-react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import Dialog from '../../components/Dialog';
import { formatTime } from '../../utils/format';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import '../../styles/project-workspace.css';
import './projects.css';

interface ProjectListResponse {
  items: Project[];
  total: number;
}

export default function Projects() {
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [error, setError] = React.useState('');
  const [projects, setProjects] = React.useState<Project[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [modalOpen, setModalOpen] = React.useState(false);
  const [newName, setNewName] = React.useState('');
  const [newId, setNewId] = React.useState('');
  const [idTouched, setIdTouched] = React.useState(false);
  const [createError, setCreateError] = React.useState('');
  const [newNote, setNewNote] = React.useState('');
  const [creating, setCreating] = React.useState(false);
  const [editing, setEditing] = React.useState<{ id: string; name: string; note: string } | null>(null);
  const [renaming, setRenaming] = React.useState(false);
  const [renameError, setRenameError] = React.useState('');
  const search = params.get('q') || '';
  const showArchived = params.get('archived') !== '0';
  const requestedPage = Math.max(1, Math.floor(Number(params.get('page')) || 1));
  const changeFilter = (patch: Record<string, string | null>, replace = false) => {
    const next = new URLSearchParams(params); next.delete('page');
    Object.entries(patch).forEach(([key, value]) => value ? next.set(key, value) : next.delete(key));
    setParams(next, { replace });
  };

  const fetchProjects = () => {
    setLoading(true);
    return apiClient.get<ProjectListResponse | Project[]>('/projects', { silent: true })
      .then((data) => {
        if (Array.isArray(data)) {
          setProjects(data);
        } else if (data && Array.isArray(data.items)) {
          setProjects(data.items);
        }
        setError('');
      })
      .catch((error) => setError(formatApiError(error)))
      .finally(() => setLoading(false));
  };

  React.useEffect(() => {
    fetchProjects();
  }, []);

  const idError = !newId ? text('请填写项目 ID。', 'Enter a project ID.') : newId.length>64 ? text('项目 ID 最多 64 个字符。', 'Project ID must be at most 64 characters.') : !/^[A-Za-z0-9_]+$/.test(newId) ? text('项目 ID 只能包含英文字母、数字和下划线。', 'Use only ASCII letters, digits and underscores in the project ID.') : '';
  const handleCreate = () => {
    setIdTouched(true);
    if (creating || !newName.trim() || idError) return;
    setCreating(true);
    setError('');
    setCreateError('');
    apiClient.post<Project>('/projects', { id: newId, name: newName.trim(), note: newNote.trim() }, {silent:true})
      .then((project) => {
        setModalOpen(false);
        setNewName('');
        setNewNote('');
        setNewId(''); setIdTouched(false);
        navigate(`/projects/${project.id}`);
      })
      .catch((error) => setCreateError(formatApiError(error)))
      .finally(() => setCreating(false));
  };

  const handleRename = () => {
    if (renaming || !editing || !editing.name.trim()) return;
    setRenaming(true); setRenameError('');
    apiClient.patch<Project>(`/projects/${editing.id}`, { name: editing.name.trim(), note: editing.note }, { silent: true })
      .then(() => {
        setEditing(null);
        void fetchProjects();
      })
      .catch(error => setRenameError(formatApiError(error)))
      .finally(() => setRenaming(false));
  };

  const handleArchive = (id: string, archived: boolean) => {
    apiClient.patch<Project>(`/projects/${id}`, { archived: !archived })
      .then(fetchProjects)
      .catch(error => setError(formatApiError(error)));
  };

  const handleDelete = (id: string, name: string) => {
    // locale 使用单花括号占位符（非 i18next 默认 {{}}），手动替换
    if (window.confirm(t('projects.deleteConfirm').replace('{name}', name))) {
      apiClient.delete(`/projects/${id}`)
        .then(fetchProjects)
        .catch(error => setError(formatApiError(error)));
    }
  };

  const query = search.trim().toLowerCase();
  const visibleProjects = projects.filter((p) => {
    if (!showArchived && p.archived) return false;
    if (!query) return true;
    return p.name.toLowerCase().includes(query) || p.id.toLowerCase().includes(query) || (p.note || '').toLowerCase().includes(query);
  });
  const pageCount = Math.max(1, Math.ceil(visibleProjects.length / 24));
  const page = Math.min(requestedPage, pageCount);
  const pageProjects = visibleProjects.slice((page - 1) * 24, page * 24);
  React.useEffect(() => {
    if (loading || error || requestedPage <= pageCount) return;
    const next = new URLSearchParams(params);
    if (pageCount === 1) next.delete('page'); else next.set('page', String(pageCount));
    setParams(next, { replace: true });
  }, [loading, error, requestedPage, pageCount, params, setParams]);

  const statChip = (icon: React.ReactNode, label: string, value: number) => (
    <span className="inline-flex items-center space-x-1 px-2 py-1 rounded-full bg-slate-100 dark:bg-slate-700/60 text-slate-600 dark:text-slate-300">
      {icon}
      <span>{label}</span>
      <span className="font-mono font-medium">{value}</span>
    </span>
  );

  const emptyState = (testid: string, icon: React.ReactNode, title: string, hint: string) => (
    <div
      className="flex flex-col items-center justify-center py-16 text-center bg-white dark:bg-slate-800 rounded-xl border border-dashed border-slate-200 dark:border-slate-700"
      data-testid={testid}
    >
      {icon}
      <h3 className="mt-3 font-semibold">{title}</h3>
      <p className="mt-1 text-sm text-slate-500">{hint}</p>
    </div>
  );

  return (
    <div className="projects-workspace" data-testid="projects-page">
      <header className="projects-toolbar">
      <div className="flex flex-wrap gap-3 justify-between items-center">
        <h2 className="text-2xl font-bold">
          {t('projects.title')}
          {!loading && projects.length > 0 && (
            <span className="ml-2 text-sm font-mono font-normal text-slate-400">
              {visibleProjects.length === projects.length ? projects.length : `${visibleProjects.length}/${projects.length}`}
            </span>
          )}
        </h2>
        <div className="flex items-center gap-2"><button type="button" aria-label={text('刷新项目', 'Refresh projects')} disabled={loading} onClick={() => void fetchProjects()} className="projects-page-button"><RefreshCw size={15}/></button><button
          onClick={() => {setCreateError('');setModalOpen(true);}}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          <FolderPlus className="w-4 h-4" />
          <span>{t('projects.newProject')}</span>
        </button></div>
      </div>

      <p className="text-xs text-slate-500">{text('每个项目从 v1 开始；复制版本可对比不同数据、参数与训练结果。', 'Projects start with v1. Copy versions to compare data, parameters and training results.')}</p>

      <div className="projects-filter-row">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
          <input
            type="text"
            value={search}
            onChange={(e) => changeFilter({ q: e.target.value || null }, true)}
            aria-label={text('搜索项目', 'Search projects')}
            placeholder={t('projects.searchPlaceholder', '按名称 / 备注搜索…')}
            className="w-full pl-9 pr-3 py-2 border rounded-lg text-sm dark:bg-slate-900 dark:border-slate-600"
            data-testid="project-search-input"
          />
        </div>
        <label className="flex items-center space-x-2 text-sm text-slate-600 dark:text-slate-300 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showArchived}
            onChange={(e) => changeFilter({ archived: e.target.checked ? null : '0' })}
            className="rounded border-slate-300"
            data-testid="show-archived-toggle"
          />
          <span>{t('projects.showArchived', '显示已归档')}</span>
        </label>
      </div>
      <nav className="projects-pagination" aria-label={text('项目分页', 'Project pagination')}>
        <span>{loading ? text('读取中…', 'Loading…') : text(`共 ${visibleProjects.length} 个项目 · 每页 24 个`, `${visibleProjects.length} projects · 24 per page`)}</span>
        <div><button className="projects-page-button" disabled={loading || page <= 1} onClick={() => changeFilter({ page: String(page - 1) })}>{text('上一页', 'Previous')}</button><span aria-label={text('当前页', 'Current page')}>{page} / {pageCount}</span><button className="projects-page-button" disabled={loading || page >= pageCount} onClick={() => changeFilter({ page: String(page + 1) })}>{text('下一页', 'Next')}</button></div>
      </nav>
      {error && <div role="alert" className="projects-list-error"><span>{error}</span><button onClick={() => void fetchProjects()} disabled={loading}>{t('common.retry')}</button></div>}
      </header>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6" data-testid="projects-skeleton">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="p-5 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 animate-pulse"
            >
              <div className="h-5 w-1/2 rounded bg-slate-200 dark:bg-slate-700" />
              <div className="mt-3 h-3 w-full rounded bg-slate-200 dark:bg-slate-700" />
              <div className="mt-2 h-3 w-2/3 rounded bg-slate-200 dark:bg-slate-700" />
              <div className="mt-4 flex space-x-2">
                <div className="h-5 w-16 rounded-full bg-slate-200 dark:bg-slate-700" />
                <div className="h-5 w-16 rounded-full bg-slate-200 dark:bg-slate-700" />
                <div className="h-5 w-16 rounded-full bg-slate-200 dark:bg-slate-700" />
              </div>
            </div>
          ))}
        </div>
      ) : visibleProjects.length === 0 ? (
        query ? (
          emptyState(
            'projects-no-results',
            <SearchX className="w-10 h-10 text-slate-300 dark:text-slate-600" />,
            t('projects.noResultsTitle', '没有找到匹配的项目'),
            t('projects.noResultsHint', '换个关键词试试，或清空搜索框。')
          )
        ) : projects.length > 0 ? (
          emptyState(
            'projects-all-archived',
            <Archive className="w-10 h-10 text-slate-300 dark:text-slate-600" />,
            t('projects.allArchivedTitle', '所有项目都已归档'),
            t('projects.allArchivedHint', '勾选上方「显示已归档」查看。')
          )
        ) : (
          emptyState(
            'projects-empty',
            <FolderOpen className="w-10 h-10 text-slate-300 dark:text-slate-600" />,
            t('projects.emptyTitle', '还没有项目'),
            t('projects.emptyHint', '点击右上角「新建项目」创建第一个项目。')
          )
        )
      ) : (
        <div className="projects-grid" aria-label={text('项目列表', 'Project list')}>
          {pageProjects.map((proj) => (
            <div
              key={proj.id}
              className="block p-5 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 hover:border-blue-500 transition-colors"
              data-testid={`project-card-${proj.id}`}
            >
              <div className="flex justify-between items-start">
                <Link to={`/projects/${proj.id}`} className="font-semibold text-lg hover:text-blue-600 dark:hover:text-blue-400">
                  {proj.name}
                  {proj.archived && (
                    <span className="ml-2 text-xs px-2 py-0.5 bg-slate-200 dark:bg-slate-700 rounded text-slate-500">{t('projects.archived')}</span>
                  )}
                </Link>
                <div className="flex space-x-1">
                  <button
                    onClick={() => { setRenameError(''); setEditing({ id: proj.id, name: proj.name, note: proj.note || '' }); }}
                    className="p-1 text-slate-400 hover:text-blue-500"
                    title={t('projects.rename')}
                  >
                    <Pencil className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleArchive(proj.id, proj.archived)}
                    className="p-1 text-slate-400 hover:text-amber-500"
                    title={proj.archived ? t('projects.unarchive') : t('projects.archive')}
                  >
                    <Archive className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleDelete(proj.id, proj.name)}
                    className="p-1 text-slate-400 hover:text-red-500"
                    title={t('common.delete')}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
              <p className="text-sm text-slate-500 mt-1 line-clamp-2">{proj.note || t('projects.noNote')}</p>
              <div className="flex items-center space-x-1 mt-3 text-xs text-slate-400">
                <Clock className="w-3.5 h-3.5" />
                <span>{t('projects.createdAt')}</span>
                <span className="font-mono">{formatTime(proj.created_at)}</span>
              </div>
              <div className="flex flex-wrap gap-2 mt-3 text-xs">
                {statChip(<GitBranch className="w-3.5 h-3.5" />, text('版本', 'Versions'), proj.version_count || 1)}
                {statChip(<Database className="w-3.5 h-3.5" />, t('projects.datasets'), proj.dataset_ids?.length || 0)}
                {statChip(<Activity className="w-3.5 h-3.5" />, t('projects.jobs'), proj.stats?.jobs || 0)}
                {statChip(<Box className="w-3.5 h-3.5" />, t('projects.artifacts'), proj.stats?.artifacts || 0)}
              </div>
              <Link to={`/projects/${proj.id}`} className="mt-4 inline-flex rounded-lg bg-blue-50 px-3 py-2 text-sm font-semibold text-blue-600 hover:bg-blue-100 dark:bg-blue-950/40 dark:text-blue-300">{text('打开训练工作区 →', 'Open training workspace →')}</Link>
            </div>
          ))}
        </div>
      )}

      {/* Create Modal */}
      {modalOpen && (
        <Dialog title={t('projects.newProject')} onClose={() => setModalOpen(false)} closeDisabled={creating}>
          <form onSubmit={event=>{event.preventDefault();handleCreate();}} className="space-y-4" data-testid="create-project-modal" aria-busy={creating}>
            <p className="text-sm text-slate-500">{text('创建后进入工作区：上传数据 → 选择模型 → 设置参数 → 启动训练。', 'Next: upload data → choose a model → configure → start training.')}</p>
            <div className="space-y-3">
              <label className="block space-y-1 text-sm"><span>{text('项目名称', 'Project name')}</span><input
                type="text" required placeholder={t('projects.namePlaceholder')} value={newName} onChange={event=>setNewName(event.target.value)} disabled={creating}
                className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600" data-testid="project-name-input"
              /><span className="block text-xs text-slate-500">{text('用于界面显示，支持中文及其他语言，可随时重命名。', 'Display name supports any language and can be renamed later.')}</span></label>
              <label className="block space-y-1 text-sm"><span>{text('项目 ID', 'Project ID')}</span><input
                type="text" required maxLength={64} value={newId} onChange={event=>{setNewId(event.target.value);setIdTouched(true);setCreateError('');}} onBlur={()=>setIdTouched(true)} disabled={creating}
                aria-invalid={idTouched&&!!idError} aria-describedby="project-id-help project-id-error" autoComplete="off" spellCheck={false}
                placeholder="my_project_01" className="w-full px-3 py-2 border rounded-md text-sm font-mono dark:bg-slate-900 dark:border-slate-600" data-testid="project-id-input"
              /></label>
              <p id="project-id-help" className="text-xs text-slate-500">{text('手工填写，只允许 A–Z、a–z、0–9 和下划线。它决定目录名称，创建后不可修改。', 'Enter manually using A–Z, a–z, 0–9 and underscores. This permanent ID is used as the folder name.')}</p>
              {idTouched&&idError&&<p id="project-id-error" role="alert" className="text-xs text-red-600">{idError}</p>}
              {createError&&<p role="alert" className="whitespace-pre-line text-sm text-red-600">{createError}</p>}
              <p className="rounded bg-slate-100 p-2 font-mono text-xs break-all dark:bg-slate-900" aria-label={text('项目目录预览','Project folder preview')}>studio_data/project/{newId&&!idError?newId:'<project_id>'}/v1/</p>
              <label className="block space-y-1 text-sm"><span>{text('备注（可选）', 'Notes (optional)')}</span><textarea placeholder={t('projects.notePlaceholder')} value={newNote} onChange={event=>setNewNote(event.target.value)} rows={2} disabled={creating} className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"/></label>
            </div>
            <div className="flex justify-end space-x-2 pt-2">
              <button type="button" onClick={()=>setModalOpen(false)} disabled={creating} className="px-4 py-2 text-sm rounded bg-slate-200 dark:bg-slate-700 hover:bg-slate-300">{t('common.cancel')}</button>
              <button type="submit" disabled={creating||!newName.trim()||!!idError} className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50">{creating?t('projects.creating'):t('projects.create')}</button>
            </div>
          </form>
        </Dialog>
      )}

      {/* Edit Modal */}
      {editing && (
        <Dialog title={t('projects.renameTitle', '重命名项目')} onClose={() => setEditing(null)} closeDisabled={renaming}>
          <form className="space-y-4" onSubmit={event => { event.preventDefault(); handleRename(); }} aria-busy={renaming}>
            {renameError && <p role="alert" className="whitespace-pre-line text-sm text-red-600">{renameError}</p>}
            <div className="space-y-3">
              <label className="block space-y-1 text-sm"><span>{text('项目名称', 'Project name')}</span><input
                type="text"
                required disabled={renaming}
                value={editing.name}
                onChange={(e) => setEditing({ ...editing, name: e.target.value })}
                className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
              /></label>
              <label className="block space-y-1 text-sm"><span>{text('备注（可选）', 'Notes (optional)')}</span><textarea
                disabled={renaming}
                value={editing.note}
                onChange={(e) => setEditing({ ...editing, note: e.target.value })}
                rows={3}
                className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
              /></label>
            </div>
            <div className="flex justify-end space-x-2 pt-2">
              <button
                type="button" disabled={renaming}
                onClick={() => setEditing(null)}
                className="px-4 py-2 text-sm rounded bg-slate-200 dark:bg-slate-700 hover:bg-slate-300"
              >
                {t('common.cancel')}
              </button>
              <button
                type="submit" disabled={renaming || !editing.name.trim()}
                className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700"
              >
                {renaming ? text('保存中…', 'Saving…') : t('common.save')}
              </button>
            </div>
          </form>
        </Dialog>
      )}
    </div>
  );
}
