import React from 'react';
import { apiClient } from '../../api/client';
import { Artifact } from '../../api/types';
import { Box, Download, Trash2, FileJson } from 'lucide-react';

const CONVERT_FORMATS = ['comfyui', 'peft', 'kohya'] as const;

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function formatTime(t: string | number): string {
  const d = typeof t === 'number' ? new Date(t * 1000) : new Date(t);
  return d.toLocaleString();
}

export default function Artifacts() {
  const [artifacts, setArtifacts] = React.useState<Artifact[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [converting, setConverting] = React.useState<string | null>(null);
  const [metadataFor, setMetadataFor] = React.useState<Artifact | null>(null);

  const fetchArtifacts = () => {
    apiClient.get<Artifact[]>('/artifacts')
      .then((data) => setArtifacts(Array.isArray(data) ? data : []))
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  React.useEffect(() => {
    fetchArtifacts();
  }, []);

  const handleConvert = (id: string, format: (typeof CONVERT_FORMATS)[number]) => {
    setConverting(id);
    apiClient.post<Artifact>(`/artifacts/${id}/convert`, { format })
      .then(fetchArtifacts)
      .catch(console.error)
      .finally(() => setConverting(null));
  };

  const handleDelete = (id: string, name: string) => {
    if (window.confirm(`Delete artifact "${name}"? This cannot be undone.`)) {
      apiClient.delete(`/artifacts/${id}`).then(fetchArtifacts).catch(console.error);
    }
  };

  return (
    <div className="space-y-6" data-testid="artifacts-page">
      <h2 className="text-2xl font-bold flex items-center space-x-2">
        <Box className="w-6 h-6 text-indigo-500" />
        <span>Artifacts</span>
      </h2>

      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        {loading ? (
          <p className="p-6 text-slate-500">Loading...</p>
        ) : artifacts.length === 0 ? (
          <p className="p-6 text-slate-500">No artifacts generated yet.</p>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 dark:bg-slate-800/50 text-xs text-slate-400 border-b dark:border-slate-700">
              <tr>
                <th className="p-4">Name</th>
                <th className="p-4">Job</th>
                <th className="p-4">Algo</th>
                <th className="p-4">Rank / Alpha / Factor</th>
                <th className="p-4">Size</th>
                <th className="p-4">Created</th>
                <th className="p-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
              {artifacts.map((a) => (
                <tr key={a.id} className="hover:bg-slate-50 dark:hover:bg-slate-750" data-testid={`artifact-row-${a.id}`}>
                  <td className="p-4 font-mono text-xs font-semibold">{a.name}</td>
                  <td className="p-4 font-mono text-xs text-slate-500">{a.job_id}</td>
                  <td className="p-4 capitalize">{a.algo}{a.kind ? <span className="ml-1 text-xs text-slate-400">({a.kind})</span> : null}</td>
                  <td className="p-4 text-xs">{a.rank} / {a.alpha} / {a.factor}</td>
                  <td className="p-4 text-xs">{formatSize(a.size)}</td>
                  <td className="p-4 text-xs text-slate-400">{formatTime(a.created_at)}</td>
                  <td className="p-4 text-right">
                    <div className="flex items-center justify-end space-x-2">
                      <a
                        href={`/api/artifacts/${a.id}/download`}
                        className="p-1.5 text-slate-500 hover:text-blue-600 rounded hover:bg-slate-100 dark:hover:bg-slate-700"
                        title="Download"
                      >
                        <Download className="w-4 h-4" />
                      </a>
                      <button
                        onClick={() => setMetadataFor(a)}
                        className="p-1.5 text-slate-500 hover:text-indigo-600 rounded hover:bg-slate-100 dark:hover:bg-slate-700"
                        title="View metadata"
                      >
                        <FileJson className="w-4 h-4" />
                      </button>
                      <select
                        disabled={converting === a.id}
                        onChange={(e) => {
                          const fmt = e.target.value as (typeof CONVERT_FORMATS)[number];
                          if (fmt) handleConvert(a.id, fmt);
                          e.target.value = '';
                        }}
                        defaultValue=""
                        className="text-xs px-1.5 py-1 border rounded dark:bg-slate-900 dark:border-slate-600"
                        title="Convert format"
                      >
                        <option value="" disabled>
                          {converting === a.id ? 'Converting…' : 'Convert'}
                        </option>
                        {CONVERT_FORMATS.map((f) => (
                          <option key={f} value={f}>{f}</option>
                        ))}
                      </select>
                      <button
                        onClick={() => handleDelete(a.id, a.name)}
                        className="p-1.5 text-slate-500 hover:text-red-600 rounded hover:bg-slate-100 dark:hover:bg-slate-700"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {metadataFor && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4" onClick={() => setMetadataFor(null)}>
          <div
            className="bg-white dark:bg-slate-800 rounded-xl max-w-2xl w-full p-6 space-y-4 shadow-xl max-h-[80vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center border-b pb-2 dark:border-slate-700">
              <h3 className="font-semibold">Metadata — {metadataFor.name}</h3>
              <button onClick={() => setMetadataFor(null)} className="text-slate-400 hover:text-slate-600">✕</button>
            </div>
            <pre className="text-xs font-mono bg-slate-50 dark:bg-slate-900 p-4 rounded overflow-x-auto">
              {JSON.stringify(metadataFor.metadata, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
