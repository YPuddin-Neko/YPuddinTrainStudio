import React from 'react';
import { apiClient } from '../../api/client';
import { Project } from '../../api/types';
import { FolderPlus } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Projects() {
  const [projects, setProjects] = React.useState<Project[]>([]);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    apiClient.get<Project[]>('/projects')
      .then((data) => {
        setProjects(Array.isArray(data) ? data : []);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Projects</h2>
        <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
          <FolderPlus className="w-4 h-4" />
          <span>New Project</span>
        </button>
      </div>

      {loading ? (
        <div>Loading...</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {projects.map((proj) => (
            <Link
              key={proj.id}
              to={`/projects/${proj.id}`}
              className="block p-5 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 hover:border-blue-500 transition-colors"
            >
              <h3 className="font-semibold text-lg">{proj.name}</h3>
              <p className="text-sm text-slate-500 mt-1 line-clamp-2">{proj.note || 'No description'}</p>
              <div className="flex space-x-4 mt-4 text-xs text-slate-400">
                <span>Datasets: {proj.dataset_ids?.length || 0}</span>
                <span>Jobs: {proj.stats?.jobs || 0}</span>
                <span>Artifacts: {proj.stats?.artifacts || 0}</span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
