import React from 'react';
import { apiClient } from '../../api/client';
import { Artifact } from '../../api/types';

export default function Artifacts() {
  const [artifacts, setArtifacts] = React.useState<Artifact[]>([]);

  React.useEffect(() => {
    apiClient.get<Artifact[]>('/artifacts').then((data) => {
      setArtifacts(Array.isArray(data) ? data : []);
    }).catch(console.error);
  }, []);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Artifacts</h2>
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
        {artifacts.length === 0 ? (
          <p className="text-slate-500">No artifacts generated yet.</p>
        ) : (
          <ul className="divide-y divide-slate-200 dark:divide-slate-700">
            {artifacts.map((a) => (
              <li key={a.id} className="py-3 flex justify-between">
                <span>{a.name}</span>
                <span className="text-slate-400">{a.algo} (rank {a.rank})</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
