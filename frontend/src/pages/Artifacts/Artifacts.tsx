import React from 'react';
import { useTranslation } from 'react-i18next';
import { apiClient, apiUrl } from '../../api/client';
import { Artifact } from '../../api/types';
import { formatBytes, formatTime } from '../../utils/format';
import { Box, Download, Trash2, FileJson, PackageOpen } from 'lucide-react';

const CONVERT_FORMATS = ['comfyui', 'peft', 'kohya'] as const;

export default function Artifacts() {
  const { t } = useTranslation();
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
    if (window.confirm(t('artifacts.deleteConfirm', { name }))) {
      apiClient.delete(`/artifacts/${id}`).then(fetchArtifacts).catch(console.error);
    }
  };

  return (
    <div className="space-y-6" data-testid="artifacts-page">
      <h2 className="text-2xl font-bold flex items-center space-x-2">
        <Box className="w-6 h-6 text-indigo-500" />
        <span>{t('artifacts.title')}</span>
      </h2>

      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        {loading ? (
          <p className="p-6 text-slate-500">{t('common.loading')}</p>
        ) : artifacts.length === 0 ? (
          <div className="p-12 flex flex-col items-center justify-center text-center space-y-3">
            <PackageOpen className="w-12 h-12 text-slate-300 dark:text-slate-600" />
            <p className="font-medium text-slate-500 dark:text-slate-400">{t('artifacts.empty')}</p>
            <p className="text-sm text-slate-400 dark:text-slate-500">
              {t('artifacts.emptyHint', '训练任务完成后，生成的产物会显示在这里。')}
            </p>
          </div>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 dark:bg-slate-800/50 text-xs text-slate-400 border-b dark:border-slate-700">
              <tr>
                <th className="p-4">{t('artifacts.name')}</th>
                <th className="p-4">{t('artifacts.job')}</th>
                <th className="p-4">{t('artifacts.algo')}</th>
                <th className="p-4">{t('artifacts.rankAlphaFactor')}</th>
                <th className="p-4">{t('artifacts.size')}</th>
                <th className="p-4">{t('artifacts.created')}</th>
                <th className="p-4 text-right">{t('artifacts.actions')}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
              {artifacts.map((a) => (
                <tr key={a.id} className="hover:bg-slate-50 dark:hover:bg-slate-750" data-testid={`artifact-row-${a.id}`}>
                  <td className="p-4 font-mono text-xs font-semibold">{a.name}</td>
                  <td className="p-4 font-mono text-xs text-slate-500">{a.job_id ?? '--'}</td>
                  <td className="p-4 capitalize">{a.algo}{a.kind ? <span className="ml-1 text-xs text-slate-400">({a.kind})</span> : null}</td>
                  <td className="p-4 text-xs font-mono">{a.rank} / {a.alpha} / {a.factor}</td>
                  <td className="p-4 text-xs font-mono">{formatBytes(a.size)}</td>
                  <td className="p-4 text-xs font-mono text-slate-400">{formatTime(a.created_at)}</td>
                  <td className="p-4 text-right">
                    <div className="flex items-center justify-end space-x-2">
                      <a
                        href={apiUrl(`/artifacts/${a.id}/download`)}
                        className="p-1.5 text-slate-500 hover:text-blue-600 rounded hover:bg-slate-100 dark:hover:bg-slate-700"
                        title={t('common.download')}
                      >
                        <Download className="w-4 h-4" />
                      </a>
                      <button
                        onClick={() => setMetadataFor(a)}
                        className="p-1.5 text-slate-500 hover:text-indigo-600 rounded hover:bg-slate-100 dark:hover:bg-slate-700"
                        title={t('artifacts.viewMetadata', '查看元数据')}
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
                        title={t('artifacts.convertFormat', '转换格式')}
                      >
                        <option value="" disabled>
                          {converting === a.id ? t('artifacts.converting') : t('artifacts.convert')}
                        </option>
                        {CONVERT_FORMATS.map((f) => (
                          <option key={f} value={f}>{f}</option>
                        ))}
                      </select>
                      <button
                        onClick={() => handleDelete(a.id, a.name)}
                        className="p-1.5 text-slate-500 hover:text-red-600 rounded hover:bg-slate-100 dark:hover:bg-slate-700"
                        title={t('common.delete')}
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
              <h3 className="font-semibold">{t('artifacts.metadata')} — {metadataFor.name}</h3>
              <button
                onClick={() => setMetadataFor(null)}
                className="text-slate-400 hover:text-slate-600"
                title={t('common.close')}
              >
                ✕
              </button>
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
