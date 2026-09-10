import React from 'react';
import { apiClient } from '../../api/client';
import { ModelAsset } from '../../api/types';
import { PathInput } from '../../components/PathBrowser';
import { HardDrive, Plus, Trash2, FolderSearch, Star } from 'lucide-react';

const KINDS = ['dit', 'text_encoder', 'vae', 'tokenizer'] as const;
const FAMILIES = ['anima', 'toy'] as const;

function formatSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

export default function Models() {
  const [models, setModels] = React.useState<ModelAsset[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [addOpen, setAddOpen] = React.useState(false);
  const [scanning, setScanning] = React.useState(false);

  const [family, setFamily] = React.useState<string>('anima');
  const [kind, setKind] = React.useState<string>('dit');
  const [path, setPath] = React.useState('');
  const [dtype, setDtype] = React.useState('bf16');
  const [isDefault, setIsDefault] = React.useState(false);
  const [adding, setAdding] = React.useState(false);

  const fetchModels = () => {
    apiClient.get<ModelAsset[]>('/models')
      .then((d) => setModels(Array.isArray(d) ? d : []))
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  React.useEffect(() => {
    fetchModels();
  }, []);

  const handleAdd = () => {
    if (!path.trim()) return;
    setAdding(true);
    apiClient.post<ModelAsset>('/models', {
      family,
      kind,
      path: path.trim(),
      dtype: dtype || null,
      is_default: isDefault,
    })
      .then(() => {
        setAddOpen(false);
        setPath('');
        fetchModels();
      })
      .catch(console.error)
      .finally(() => setAdding(false));
  };

  const handleScan = () => {
    setScanning(true);
    apiClient.post('/models/scan', {})
      .then((res: any) => {
        fetchModels();
        if (res && typeof res.added === 'number') {
          alert(`Scan complete: ${res.added} new model(s) found.`);
        }
      })
      .catch(console.error)
      .finally(() => setScanning(false));
  };

  const handleDelete = (id: string, p: string) => {
    if (window.confirm(`Remove model registration "${p}"? Files on disk will NOT be deleted.`)) {
      apiClient.delete(`/models/${id}`).then(fetchModels).catch(console.error);
    }
  };

  return (
    <div className="space-y-6" data-testid="models-page">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold flex items-center space-x-2">
          <HardDrive className="w-6 h-6 text-purple-500" />
          <span>Model Weights</span>
        </h2>
        <div className="flex space-x-2">
          <button
            onClick={handleScan}
            disabled={scanning}
            className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-slate-100 dark:bg-slate-800 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-50"
          >
            <FolderSearch className="w-4 h-4" />
            <span>{scanning ? 'Scanning…' : 'Scan Directory'}</span>
          </button>
          <button
            onClick={() => setAddOpen(true)}
            className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            data-testid="add-model-btn"
          >
            <Plus className="w-4 h-4" />
            <span>Add Model</span>
          </button>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        {loading ? (
          <p className="p-6 text-slate-500">Loading…</p>
        ) : models.length === 0 ? (
          <p className="p-10 text-center text-slate-400">No model weights registered. Add one or scan a directory.</p>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 dark:bg-slate-800/50 text-xs text-slate-400 border-b dark:border-slate-700">
              <tr>
                <th className="p-4">Family</th>
                <th className="p-4">Kind</th>
                <th className="p-4">Path</th>
                <th className="p-4">Size</th>
                <th className="p-4">Dtype</th>
                <th className="p-4">Status</th>
                <th className="p-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
              {models.map((m) => (
                <tr key={m.id} className="hover:bg-slate-50 dark:hover:bg-slate-750" data-testid={`model-row-${m.id}`}>
                  <td className="p-4 capitalize font-medium">{m.family}</td>
                  <td className="p-4">{m.kind}</td>
                  <td className="p-4 font-mono text-xs text-slate-500 break-all">
                    {m.is_default && <Star className="w-3.5 h-3.5 inline text-amber-500 mr-1" />}
                    {m.path}
                  </td>
                  <td className="p-4 text-xs">{formatSize(m.size)}</td>
                  <td className="p-4 text-xs font-mono">{m.dtype || '--'}</td>
                  <td className="p-4">
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      m.exists
                        ? 'bg-green-100 text-green-700 dark:bg-green-950/40 dark:text-green-400'
                        : 'bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-400'
                    }`}>
                      {m.exists ? 'exists' : 'missing'}
                    </span>
                  </td>
                  <td className="p-4 text-right">
                    <button
                      onClick={() => handleDelete(m.id, m.path)}
                      className="p-1.5 text-slate-400 hover:text-red-500"
                      title="Remove"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {addOpen && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4" onClick={() => setAddOpen(false)}>
          <div
            className="bg-white dark:bg-slate-800 rounded-xl max-w-lg w-full p-6 space-y-4 shadow-xl"
            onClick={(e) => e.stopPropagation()}
            data-testid="add-model-modal"
          >
            <h3 className="font-semibold text-lg">Add Model Weight</h3>
            <div className="space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-slate-400">Family</label>
                  <select
                    value={family}
                    onChange={(e) => setFamily(e.target.value)}
                    className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600"
                  >
                    {FAMILIES.map((f) => <option key={f} value={f}>{f}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-slate-400">Kind</label>
                  <select
                    value={kind}
                    onChange={(e) => setKind(e.target.value)}
                    className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600"
                  >
                    {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
                  </select>
                </div>
              </div>
              <div>
                <label className="text-xs text-slate-400">Path (file or HF directory)</label>
                <PathInput value={path} onChange={setPath} placeholder="/models/xxx.safetensors" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-slate-400">Dtype</label>
                  <select
                    value={dtype}
                    onChange={(e) => setDtype(e.target.value)}
                    className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600"
                  >
                    <option value="bf16">bf16</option>
                    <option value="fp16">fp16</option>
                    <option value="fp32">fp32</option>
                    <option value="fp8">fp8</option>
                    <option value="">unknown</option>
                  </select>
                </div>
                <div className="flex items-center space-x-2 pt-5">
                  <input
                    type="checkbox"
                    checked={isDefault}
                    onChange={(e) => setIsDefault(e.target.checked)}
                    className="rounded text-blue-600"
                  />
                  <span className="text-xs">Set as default for this family+kind</span>
                </div>
              </div>
            </div>
            <div className="flex justify-end space-x-2 pt-2">
              <button onClick={() => setAddOpen(false)} className="px-4 py-2 text-sm rounded bg-slate-200 dark:bg-slate-700">Cancel</button>
              <button
                onClick={handleAdd}
                disabled={adding || !path.trim()}
                className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                data-testid="add-model-submit"
              >
                {adding ? 'Adding…' : 'Add'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
