import React from 'react';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { Project } from '../../api/types';
import {
  FolderPlus,
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
} from 'lucide-react';
import { Link } from 'react-router-dom';
import { formatTime } from '../../utils/format';

interface ProjectListResponse {
  items: Project[];
  total: number;
}

export default function Projects() {
  const { t } = useTranslation();
  const [projects, setProjects] = React.useState<Project[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [modalOpen, setModalOpen] = React.useState(false);
  const [newName, setNewName] = React.useState('');
  const [newNote, setNewNote] = React.useState('');
  const [creating, setCreating] = React.useState(false);
  const [editing, setEditing] = React.useState<{ id: string; name: string; note: string } | null>(null);
  const [search, setSearch] = React.useState('');
  const [showArchived, setShowArchived] = React.useState(true);

  const fetchProjects = () => {
    apiClient.get<ProjectListResponse | Project[]>('/projects')
      .then((data) => {
        if (Array.isArray(data)) {
          setProjects(data);
        } else if (data && Array.isArray(data.items)) {
          setProjects(data.items);
        }
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  React.useEffect(() => {
    fetchProjects();
  }, []);

  const handleCreate = () => {
    if (!newName.trim()) return;
    setCreating(true);
    apiClient.post<Project>('/projects', { name: newName.trim(), note: newNote.trim() })
      .then(() => {
        setModalOpen(false);
        setNewName('');
        setNewNote('');
        fetchProjects();
      })
      .catch(console.error)
      .finally(() => setCreating(false));
  };

  const handleRename = () => {
    if (!editing || !editing.name.trim()) return;
    apiClient.patch<Project>(`/projects/${editing.id}`, { name: editing.name.trim(), note: editing.note })
      .then(() => {
        setEditing(null);
        fetchProjects();
      })
      .catch(console.error);
  };

  const handleArchive = (id: string, archived: boolean) => {
    apiClient.patch<Project>(`/projects/${id}`, { archived: !archived })
      .then(fetchProjects)
      .catch(console.error);
  };

  const handleDelete = (id: string, name: string) => {
    // locale 使用单花括号占位符（非 i18next 默认 {{}}），手动替换
    if (window.confirm(t('projects.deleteConfirm').replace('{name}', name))) {
      apiClient.delete(`/projects/${id}`)
        .then(fetchProjects)
        .catch(console.error);
    }
  };

  const query = search.trim().toLowerCase();
  const visibleProjects = projects.filter((p) => {
    if (!showArchived && p.archived) return false;
    if (!query) return true;
    return p.name.toLowerCase().includes(query) || (p.note || '').toLowerCase().includes(query);
  });

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
    <div className="space-y-6" data-testid="projects-page">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">
          {t('projects.title')}
          {!loading && projects.length > 0 && (
            <span className="ml-2 text-sm font-mono font-normal text-slate-400">
              {visibleProjects.length === projects.length ? projects.length : `${visibleProjects.length}/${projects.length}`}
            </span>
          )}
        </h2>
        <button
          onClick={() => setModalOpen(true)}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          <FolderPlus className="w-4 h-4" />
          <span>{t('projects.newProject')}</span>
        </button>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={t('projects.searchPlaceholder', '按名称 / 备注搜索…')}
            className="w-full pl-9 pr-3 py-2 border rounded-lg text-sm dark:bg-slate-900 dark:border-slate-600"
            data-testid="project-search-input"
          />
        </div>
        <label className="flex items-center space-x-2 text-sm text-slate-600 dark:text-slate-300 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showArchived}
            onChange={(e) => setShowArchived(e.target.checked)}
            className="rounded border-slate-300"
            data-testid="show-archived-toggle"
          />
          <span>{t('projects.showArchived', '显示已归档')}</span>
        </label>
      </div>

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
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {visibleProjects.map((proj) => (
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
                    onClick={() => setEditing({ id: proj.id, name: proj.name, note: proj.note || '' })}
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
                {statChip(<Database className="w-3.5 h-3.5" />, t('projects.datasets'), proj.dataset_ids?.length || 0)}
                {statChip(<Activity className="w-3.5 h-3.5" />, t('projects.jobs'), proj.stats?.jobs || 0)}
                {statChip(<Box className="w-3.5 h-3.5" />, t('projects.artifacts'), proj.stats?.artifacts || 0)}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Modal */}
      {modalOpen && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-800 rounded-xl max-w-md w-full p-6 space-y-4 shadow-xl" data-testid="create-project-modal">
            <h3 className="font-semibold text-lg">{t('projects.newProject')}</h3>
            <div className="space-y-3">
              <input
                type="text"
                placeholder={t('projects.namePlaceholder')}
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
                data-testid="project-name-input"
                autoFocus
              />
              <textarea
                placeholder={t('projects.notePlaceholder')}
                value={newNote}
                onChange={(e) => setNewNote(e.target.value)}
                rows={3}
                className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
              />
            </div>
            <div className="flex justify-end space-x-2 pt-2">
              <button
                onClick={() => setModalOpen(false)}
                className="px-4 py-2 text-sm rounded bg-slate-200 dark:bg-slate-700 hover:bg-slate-300"
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={handleCreate}
                disabled={creating || !newName.trim()}
                className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
              >
                {creating ? t('projects.creating') : t('projects.create')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Edit Modal */}
      {editing && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-800 rounded-xl max-w-md w-full p-6 space-y-4 shadow-xl">
            <h3 className="font-semibold text-lg">{t('projects.renameTitle', '重命名项目')}</h3>
            <div className="space-y-3">
              <input
                type="text"
                value={editing.name}
                onChange={(e) => setEditing({ ...editing, name: e.target.value })}
                className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
              />
              <textarea
                value={editing.note}
                onChange={(e) => setEditing({ ...editing, note: e.target.value })}
                rows={3}
                className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
              />
            </div>
            <div className="flex justify-end space-x-2 pt-2">
              <button
                onClick={() => setEditing(null)}
                className="px-4 py-2 text-sm rounded bg-slate-200 dark:bg-slate-700 hover:bg-slate-300"
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={handleRename}
                className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700"
              >
                {t('common.save')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
