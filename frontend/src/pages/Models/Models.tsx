import React from 'react';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { ModelAsset } from '../../api/types';
import { PathInput } from '../../components/PathBrowser';
import { useFamilies, familyByName } from '../../api/hooks/useFamilies';
import { formatBytes } from '../../utils/format';
import { HardDrive, Plus, Trash2, FolderSearch, Star } from 'lucide-react';

const KIND_BY_FIELD: Record<string, string> = {
  dit_path: 'dit',
  text_encoder_path: 'text_encoder',
  vae_path: 'vae',
  tokenizer_path: 'tokenizer',
};

// locales 统一使用 {var} 单花括号插值，此处按调用覆盖 i18next 默认的 {{var}} 格式
const SINGLE_BRACE = { prefix: '{', suffix: '}' } as const;

export default function Models() {
  const { t } = useTranslation();
  const { data: families } = useFamilies();
  const [models, setModels] = React.useState<ModelAsset[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [addOpen, setAddOpen] = React.useState(false);
  const [scanning, setScanning] = React.useState(false);

  const [family, setFamily] = React.useState<string>('');
  const [kind, setKind] = React.useState<string>('dit');
  const [path, setPath] = React.useState('');
  const [dtype, setDtype] = React.useState('bf16');
  const [isDefault, setIsDefault] = React.useState(false);
  const [adding, setAdding] = React.useState(false);

  // 族列表就绪后默认选中第一个
  React.useEffect(() => {
    if (!family && families && families.length > 0) {
      setFamily(families[0].name);
    }
  }, [families, family]);

  // kind 选项按当前族的 weights[].field 映射
  const currentFamily = familyByName(families, family);
  const kindOptions = React.useMemo(() => {
    const fields = (currentFamily?.weights || []).map((w) => KIND_BY_FIELD[w.field]).filter(Boolean);
    return fields.length > 0 ? fields : ['dit', 'text_encoder', 'vae', 'tokenizer'];
  }, [currentFamily]);

  React.useEffect(() => {
    if (kindOptions.length > 0 && !kindOptions.includes(kind)) {
      setKind(kindOptions[0]);
    }
  }, [kindOptions, kind]);

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
          alert(t('models.scanResult', { n: res.added, interpolation: SINGLE_BRACE }));
        }
      })
      .catch(console.error)
      .finally(() => setScanning(false));
  };

  const handleDelete = (id: string, p: string) => {
    if (window.confirm(t('models.deleteConfirm', { path: p, interpolation: SINGLE_BRACE }))) {
      apiClient.delete(`/models/${id}`).then(fetchModels).catch(console.error);
    }
  };

  return (
    <div className="space-y-6" data-testid="models-page">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold flex items-center space-x-2">
          <HardDrive className="w-6 h-6 text-purple-500" />
          <span>{t('models.title')}</span>
        </h2>
        <div className="flex space-x-2">
          <button
            onClick={handleScan}
            disabled={scanning}
            className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-slate-100 dark:bg-slate-800 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-700 disabled:opacity-50"
          >
            <FolderSearch className="w-4 h-4" />
            <span>{scanning ? t('models.scanning') : t('models.scanDirectory')}</span>
          </button>
          <button
            onClick={() => setAddOpen(true)}
            className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            data-testid="add-model-btn"
          >
            <Plus className="w-4 h-4" />
            <span>{t('models.addModel')}</span>
          </button>
        </div>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        {loading ? (
          <p className="p-6 text-slate-500">{t('common.loading')}</p>
        ) : models.length === 0 ? (
          <div className="p-10 flex flex-col items-center justify-center text-center space-y-2">
            <HardDrive className="w-10 h-10 text-slate-300 dark:text-slate-600" />
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">{t('common.empty')}</p>
            <p className="text-xs text-slate-400">{t('models.empty')}</p>
          </div>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="bg-slate-50 dark:bg-slate-800/50 text-xs text-slate-400 border-b dark:border-slate-700">
              <tr>
                <th className="p-4">{t('models.family')}</th>
                <th className="p-4">{t('models.kind')}</th>
                <th className="p-4">{t('models.path')}</th>
                <th className="p-4">{t('models.size')}</th>
                <th className="p-4">{t('models.dtype')}</th>
                <th className="p-4">{t('models.status')}</th>
                <th className="p-4 text-right">{t('common.actions')}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
              {models.map((m) => (
                <tr key={m.id} className="hover:bg-slate-50 dark:hover:bg-slate-750" data-testid={`model-row-${m.id}`}>
                  <td className="p-4 capitalize font-medium">
                    {familyByName(families, m.family)?.label || m.family}
                  </td>
                  <td className="p-4">{m.kind}</td>
                  <td className="p-4 font-mono text-xs text-slate-500 break-all">
                    {m.is_default && <Star className="w-3.5 h-3.5 inline text-amber-500 mr-1" />}
                    {m.path}
                  </td>
                  <td className="p-4 text-xs font-mono">{formatBytes(m.size)}</td>
                  <td className="p-4 text-xs font-mono">
                    {m.dtype === 'fp8' ? (
                      <span className="px-1.5 py-0.5 rounded bg-purple-100 text-purple-700 dark:bg-purple-950/40 dark:text-purple-300 font-semibold" data-testid="fp8-tag">
                        fp8
                      </span>
                    ) : (
                      m.dtype || '--'
                    )}
                  </td>
                  <td className="p-4">
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      m.exists
                        ? 'bg-green-100 text-green-700 dark:bg-green-950/40 dark:text-green-400'
                        : 'bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-400'
                    }`}>
                      {m.exists ? t('models.exists') : t('models.missing')}
                    </span>
                  </td>
                  <td className="p-4 text-right">
                    <button
                      onClick={() => handleDelete(m.id, m.path)}
                      className="p-1.5 text-slate-400 hover:text-red-500"
                      title={t('common.remove')}
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
            <h3 className="font-semibold text-lg">{t('models.addModel')}</h3>
            <div className="space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-slate-400">{t('models.family')}</label>
                  <select
                    value={family}
                    onChange={(e) => setFamily(e.target.value)}
                    className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600"
                    data-testid="model-family-select"
                  >
                    {(families || []).map((f) => (
                      <option key={f.name} value={f.name}>{f.label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-slate-400">{t('models.kind')}</label>
                  <select
                    value={kind}
                    onChange={(e) => setKind(e.target.value)}
                    className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600"
                    data-testid="model-kind-select"
                  >
                    {kindOptions.map((k) => <option key={k} value={k}>{k}</option>)}
                  </select>
                </div>
              </div>
              <div>
                <label className="text-xs text-slate-400">{t('models.pathLabel')}</label>
                <PathInput value={path} onChange={setPath} placeholder={t('models.pathPlaceholder')} />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-slate-400">{t('models.dtype')}</label>
                  <select
                    value={dtype}
                    onChange={(e) => setDtype(e.target.value)}
                    className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600"
                  >
                    <option value="bf16">bf16</option>
                    <option value="fp16">fp16</option>
                    <option value="fp32">fp32</option>
                    <option value="fp8">fp8</option>
                    <option value="">{t('models.dtypeUnknown', '未知')}</option>
                  </select>
                </div>
                <div className="flex items-center space-x-2 pt-5">
                  <input
                    type="checkbox"
                    checked={isDefault}
                    onChange={(e) => setIsDefault(e.target.checked)}
                    className="rounded text-blue-600"
                  />
                  <span className="text-xs">{t('models.setDefault')}</span>
                </div>
              </div>
            </div>
            <div className="flex justify-end space-x-2 pt-2">
              <button onClick={() => setAddOpen(false)} className="px-4 py-2 text-sm rounded bg-slate-200 dark:bg-slate-700">{t('common.cancel')}</button>
              <button
                onClick={handleAdd}
                disabled={adding || !path.trim()}
                className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                data-testid="add-model-submit"
              >
                {adding ? t('models.adding') : t('models.add')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
