import React from 'react';
import { apiClient } from '../../api/client';
import { Project } from '../../api/types';
import { FolderPlus, Trash2, Archive, Pencil } from 'lucide-react';
import { Link } from 'react-router-dom';

interface ProjectListResponse {
  items: Project[];
  total: number;
}

export default function Projects() {
  const [projects, setProjects] = React.useState<Project[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [modalOpen, setModalOpen] = React.useState(false);
  const [newName, setNewName] = React.useState('');
  const [newNote, setNewNote] = React.useState('');
  const [creating, setCreating] = React.useState(false);
  const [editing, setEditing] = React.useState<{ id: string; name: string; note: string } | null>(null);

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
    if (window.confirm(`Delete project "${name}"? This will remove the project directory from disk. This action cannot be undone.`)) {
      apiClient.delete(`/projects/${id}`)
        .then(fetchProjects)
        .catch(console.error);
    }
  };

  return (
    <div className="space-y-6" data-testid="projects-page">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Projects</h2>
        <button
          onClick={() => setModalOpen(true)}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          <FolderPlus className="w-4 h-4" />
          <span>New Project</span>
        </button>
      </div>

      {loading ? (
        <div className="text-slate-500">Loading...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {projects.map((proj) => (
            <div
              key={proj.id}
              className="block p-5 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 hover:border-blue-500 transition-colors"
              data-testid={`project-card-${proj.id}`}
            >
              <div className="flex justify-between items-start">
                <Link to={`/projects/${proj.id}`} className="font-semibold text-lg hover:text-blue-600 dark:hover:text-blue-400">
                  {proj.name}
                  {proj.archived && (
                    <span className="ml-2 text-xs px-2 py-0.5 bg-slate-200 dark:bg-slate-700 rounded text-slate-500">Archived</span>
                  )}
                </Link>
                <div className="flex space-x-1">
                  <button
                    onClick={() => setEditing({ id: proj.id, name: proj.name, note: proj.note || '' })}
                    className="p-1 text-slate-400 hover:text-blue-500"
                    title="Rename"
                  >
                    <Pencil className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleArchive(proj.id, proj.archived)}
                    className="p-1 text-slate-400 hover:text-amber-500"
                    title={proj.archived ? 'Unarchive' : 'Archive'}
                  >
                    <Archive className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => handleDelete(proj.id, proj.name)}
                    className="p-1 text-slate-400 hover:text-red-500"
                    title="Delete"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
              <p className="text-sm text-slate-500 mt-1 line-clamp-2">{proj.note || 'No description'}</p>
              <div className="flex space-x-4 mt-4 text-xs text-slate-400">
                <span>Datasets: {proj.dataset_ids?.length || 0}</span>
                <span>Jobs: {proj.stats?.jobs || 0}</span>
                <span>Artifacts: {proj.stats?.artifacts || 0}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Modal */}
      {modalOpen && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-800 rounded-xl max-w-md w-full p-6 space-y-4 shadow-xl" data-testid="create-project-modal">
            <h3 className="font-semibold text-lg">New Project</h3>
            <div className="space-y-3">
              <input
                type="text"
                placeholder="Project name"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="w-full px-3 py-2 border rounded-md text-sm dark:bg-slate-900 dark:border-slate-600"
                data-testid="project-name-input"
                autoFocus
              />
              <textarea
                placeholder="Note (optional)"
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
                Cancel
              </button>
              <button
                onClick={handleCreate}
                disabled={creating || !newName.trim()}
                className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
              >
                {creating ? 'Creating...' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Edit Modal */}
      {editing && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-800 rounded-xl max-w-md w-full p-6 space-y-4 shadow-xl">
            <h3 className="font-semibold text-lg">Rename Project</h3>
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
                Cancel
              </button>
              <button
                onClick={handleRename}
                className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
