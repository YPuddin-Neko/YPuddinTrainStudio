import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { apiClient } from '../../api/client';
import { Project } from '../../api/types';

export default function ProjectDetail() {
  const { id } = useParams<{ id: string }>();
  const [project, setProject] = React.useState<Project | null>(null);

  React.useEffect(() => {
    if (id) {
      apiClient.get<Project>(`/projects/${id}`).then(setProject).catch(console.error);
    }
  }, [id]);

  if (!project) return <div>Loading...</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold">{project.name}</h2>
          <p className="text-slate-500 mt-1">{project.note}</p>
        </div>
        <Link
          to={`/projects/${id}/train`}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          New Training Job
        </Link>
      </div>

      <div className="border-b border-slate-200 dark:border-slate-700">
        <nav className="flex space-x-8">
          <button className="border-b-2 border-blue-500 py-4 px-1 font-medium text-blue-600 dark:text-blue-400">Datasets</button>
          <button className="border-b-2 border-transparent py-4 px-1 font-medium text-slate-500 hover:text-slate-700">Configs</button>
          <button className="border-b-2 border-transparent py-4 px-1 font-medium text-slate-500 hover:text-slate-700">Jobs</button>
          <button className="border-b-2 border-transparent py-4 px-1 font-medium text-slate-500 hover:text-slate-700">Artifacts</button>
        </nav>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow-sm border border-slate-200 dark:border-slate-700 min-h-[400px]">
         <p className="text-slate-500 text-center py-10">Project datasets will show here</p>
      </div>
    </div>
  );
}
