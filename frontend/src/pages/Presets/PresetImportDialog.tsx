import React from 'react';
import { Upload } from 'lucide-react';
import { apiClient } from '../../api/client';
import Dialog from '../../components/Dialog';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { mergeConfig } from '../../utils/config';
import { PRESET_MODEL_FIELDS, reusableTrainingPreset } from '../../utils/trainingPresets';

export interface ImportedPreset { name: string; description: string; config: Record<string, any>; }

export default function PresetImportDialog({ onClose, onImport }: { onClose: () => void; onImport: (preset: ImportedPreset) => Promise<boolean> }) {
  const text = useWorkspaceText();
  const [content, setContent] = React.useState('');
  const [filename, setFilename] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [reading, setReading] = React.useState(false);
  const [error, setError] = React.useState('');
  const fileInput = React.useRef<HTMLInputElement>(null);
  const readVersion = React.useRef(0);
  React.useEffect(() => () => { readVersion.current += 1; }, []);
  const readFile = (file?: File) => {
    if (!file) return;
    const version = ++readVersion.current;
    setReading(true); setError('');
    const reader = new FileReader();
    reader.onload = () => {
      if (version !== readVersion.current) return;
      setContent(String(reader.result || '')); setFilename(file.name); setReading(false);
    };
    reader.onerror = () => {
      if (version !== readVersion.current) return;
      setReading(false); setError(text('无法读取文件，请重新选择。', 'Could not read the file. Please select it again.'));
    };
    reader.readAsText(file);
  };
  const apply = async () => {
    setBusy(true); setError('');
    try {
      const source = content.trim().replace(/^\uFEFF/, '');
      const isJson = source.startsWith('{') || filename.toLowerCase().endsWith('.json');
      let name = filename.replace(/\.(json|toml)$/i, '');
      let description = '';
      let configText = source;
      if (isJson) {
        let parsed: unknown;
        try { parsed = JSON.parse(source); }
        catch { throw new Error(text('JSON 格式有误，请检查后重试。', 'Invalid JSON. Check the content and try again.')); }
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error(text('预设内容须为 JSON 对象。', 'The preset must be a JSON object.'));
        const envelope = parsed as Record<string, unknown>;
        let config = envelope;
        if (Object.prototype.hasOwnProperty.call(envelope, 'config')) {
          if (!envelope.config || typeof envelope.config !== 'object' || Array.isArray(envelope.config)) throw new Error(text('预设中的 config 须为对象。', 'The preset config must be an object.'));
          config = envelope.config as Record<string, unknown>;
          if (typeof envelope.name === 'string') name = envelope.name;
          if (typeof envelope.description === 'string') description = envelope.description;
        }
        const model = config.model as Record<string, unknown> | undefined;
        const family = typeof model?.family === 'string' ? model.family : 'anima';
        const defaults = reusableTrainingPreset(await apiClient.get<Record<string, any>>('/config/defaults', { params: { family }, silent: true }));
        for (const field of PRESET_MODEL_FIELDS) if (defaults.model) delete defaults.model[field];
        configText = JSON.stringify(mergeConfig(defaults, config));
      }
      const config = await apiClient.post<Record<string, any>>('/config/import', { text: configText, format: isJson ? 'json' : 'toml' }, { silent: true });
      if (await onImport({ name, description, config })) onClose();
    } catch (failure) { setError(formatApiError(failure)); }
    finally { setBusy(false); }
  };
  return <Dialog title={text('导入预设', 'Import preset')} onClose={onClose} closeDisabled={busy}>
    <div className="presets-import">
      <div className="presets-import-file"><button type="button" className="ui-btn" disabled={busy || reading} onClick={() => fileInput.current?.click()}><Upload size={15}/>{text('选择文件', 'Choose file')}</button><span>{filename || 'JSON / TOML'}</span><input ref={fileInput} type="file" accept=".json,.toml,application/json,application/toml" hidden aria-label={text('预设文件', 'Preset file')} onChange={event => { readFile(event.target.files?.[0]); event.target.value = ''; }}/></div>
      <label>{text('预设内容', 'Preset content')}<textarea aria-label={text('预设内容', 'Preset content')} disabled={busy || reading} value={content} placeholder={text('也可以粘贴 JSON 或 TOML 参数', 'Or paste JSON or TOML parameters')} onChange={event => { setContent(event.target.value); setFilename(''); setError(''); }}/></label>
      {error && <p role="alert" className="studio-error">{error}</p>}
    </div>
    <div className="presets-confirm-actions"><button type="button" className="ui-btn" disabled={busy} onClick={onClose}>{text('取消', 'Cancel')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={busy || reading || !content.trim()} onClick={() => void apply()}>{busy ? text('正在导入…', 'Importing…') : text('导入为新预设', 'Import as new preset')}</button></div>
  </Dialog>;
}
