import React from 'react';
import { apiClient } from '../../api/client';
import { Job } from '../../api/types';
import { Link } from 'react-router-dom';

export default function Queue() {
  const [jobs, setJobs] = React.useState<Job[]>([]);

  React.useEffect(() => {
    apiClient.get<Job[]>('/jobs').then((data) => {
      setJobs(Array.isArray(data) ? data : []);
    }).catch(console.error);
  }, []);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Job Queue</h2>
      
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700 text-xs font-semibold text-slate-400">
              <th className="p-4">ID</th>
              <th className="p-4">Name</th>
              <th className="p-4">Status</th>
              <th className="p-4">Progress</th>
              <th className="p-4">Priority</th>
              <th className="p-4">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200 dark:divide-slate-700 text-sm">
            {jobs.map((job) => (
              <tr key={job.id} className="hover:bg-slate-50 dark:hover:bg-slate-750">
                <td className="p-4 font-mono text-xs">{job.id}</td>
                <td className="p-4 font-medium">
                  <Link to={`/jobs/${job.id}`} className="hover:underline text-blue-500">
                    {job.name}
                  </Link>
                </td>
                <td className="p-4">
                  <span className={`px-2 py-1 rounded text-xs ${
                    job.status === 'running' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300'
                  }`}>
                    {job.status}
                  </span>
                </td>
                <td className="p-4">
                  {job.progress ? `${job.progress.step} / ${job.progress.total_steps}` : '--'}
                </td>
                <td className="p-4">{job.priority}</td>
                <td className="p-4">
                  <button className="text-red-500 hover:underline text-xs">Cancel</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
