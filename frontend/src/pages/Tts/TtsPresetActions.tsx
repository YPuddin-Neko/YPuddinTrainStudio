import React from 'react';
import { Link } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Save, Settings2 } from 'lucide-react';
import { ttsPresetsApi, type TtsPresetConfig, type TtsPresetResolved, type TtsPresetResolveBody } from '../../api/ttsPresets';
import type { TtsEngine } from '../../api/tts';
import Dialog from '../../components/Dialog';
import StudioSelect from '../../components/StudioSelect';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import type { TtsEditorHandle } from './TtsVersionEditor';
import { fieldCopy, fields, type TtsField } from './ttsVersionFields';
import { gptSovitsFieldCopy } from './gptSovitsVersionFields';
import { portableTtsParameters } from './ttsPresetParameters';
import '../Presets/tts-presets.css';

export default function TtsPresetActions({ engine, readOnly, editorRef }: { engine: TtsEngine; readOnly: boolean; editorRef: React.RefObject<TtsEditorHandle> }) {
  const text = useWorkspaceText(), english = text('zh', 'en') === 'en';
  const client = useQueryClient();
  const list = useQuery({ queryKey: ['tts-presets', engine], queryFn: ({ signal }) => ttsPresetsApi.list(engine, signal) });
  const [busy, setBusy] = React.useState(false), pending = React.useRef(false), mounted = React.useRef(true);
  const [error, setError] = React.useState('');
  const [saveAs, setSaveAs] = React.useState<TtsPresetConfig | null>(null);
  const [name, setName] = React.useState(''), [description, setDescription] = React.useState('');
  const [preview, setPreview] = React.useState<{ result: TtsPresetResolved; name: string; source: string } | null>(null);
  React.useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const resolve = async (id: string) => {
    const editor = editorRef.current;
    if (pending.current || readOnly || !editor) return;
    const config = editor.getDraftConfig(), source = JSON.stringify(config);
    pending.current = true; setBusy(true); setError('');
    try {
      const result = await ttsPresetsApi.resolve(id, { config: JSON.parse(source) as TtsPresetResolveBody['config'] });
      if (!mounted.current || !editorRef.current) return;
      if (source !== JSON.stringify(editorRef.current.getDraftConfig())) { setError(text('参数已变化，请重新载入预设。', 'Parameters changed. Load the preset again.')); return; }
      setPreview({ result, source, name: list.data?.find(item => item.id === id)?.name || '' });
    } catch (failure) { if (mounted.current) setError(formatApiError(failure)); }
    finally { pending.current = false; if (mounted.current) setBusy(false); }
  };
  const apply = () => {
    if (!preview || readOnly || pending.current) return;
    const editor = editorRef.current;
    if (!editor || preview.source !== JSON.stringify(editor.getDraftConfig())) { setPreview(null); setError(text('参数已变化，请重新载入预设。', 'Parameters changed. Load the preset again.')); return; }
    if (editor.loadConfig(preview.result.config)) { setPreview(null); setError(''); }
    else setError(text('当前参数暂时无法载入预设，请先处理保存或冲突。', 'Resolve the pending save or conflict before loading the preset.'));
  };
  const create = async () => {
    if (!saveAs || pending.current || readOnly) return;
    pending.current = true; setBusy(true); setError('');
    try {
      await ttsPresetsApi.create({ name: name.trim(), description, config: saveAs });
      if (!mounted.current) return;
      setSaveAs(null); void client.invalidateQueries({ queryKey: ['tts-presets'] });
    } catch (failure) { if (mounted.current) setError(formatApiError(failure)); }
    finally { pending.current = false; if (mounted.current) setBusy(false); }
  };
  const label = (field: string) => engine === 'gpt-sovits-v5' ? gptSovitsFieldCopy(field, english).label : fields.includes(field as TtsField) ? fieldCopy(field as TtsField, english).label : field;
  return <>
    <StudioSelect aria-label={text('载入语音预设', 'Load speech preset')} placeholder={text('载入预设', 'Load preset')} value="" disabled={readOnly || busy || list.isPending || !list.data?.length} onValueChange={id => void resolve(id)} options={(list.data || []).filter(item => item.config.engine === engine).map(item => ({ value: item.id, label: item.name }))}/>
    <button type="button" className="ui-btn" disabled={readOnly || busy} onClick={() => { const config = editorRef.current?.getConfig(); if (!config) return; setName(''); setDescription(''); setError(''); setSaveAs(portableTtsParameters(config)); }}><Save size={14}/>{text('另存为新预设', 'Save as new preset')}</button>
    <Link className="ui-btn" to={`/presets?type=tts&engine=${encodeURIComponent(engine)}`}><Settings2 size={14}/>{text('管理参数预设', 'Manage parameter presets')}</Link>
    {list.error && <span className="tts-preset-feedback" role="alert">{formatApiError(list.error)} <button type="button" className="ui-link" onClick={() => void list.refetch()}>{text('重试', 'Retry')}</button></span>}
    {error && !saveAs && !preview && <span className="tts-preset-feedback" role="alert">{error}</span>}
    {preview && <Dialog title={text('载入参数预设', 'Load parameter preset')} onClose={() => { setPreview(null); setError(''); }}>
      <p>{text(`将“${preview.name}”载入当前参数草稿。`, `Load “${preview.name}” into the current parameter draft.`)}</p>
      {preview.result.changed_fields.length ? <ul className="tts-preset-changes">{preview.result.changed_fields.map(field => <li key={field}>{label(field)}</li>)}</ul> : <p>{text('训练参数与此预设相同。', 'Training parameters already match this preset.')}</p>}
      {error && <p role="alert" className="studio-error">{error}</p>}
      <footer className="tts-dialog-actions"><button type="button" className="ui-btn" onClick={() => setPreview(null)}>{text('取消', 'Cancel')}</button><button type="button" className="ui-btn ui-btn-primary" onClick={apply}>{text('载入草稿', 'Load into draft')}</button></footer>
    </Dialog>}
    {saveAs && <Dialog title={text('另存为新预设', 'Save as new preset')} closeDisabled={busy} onClose={() => { setSaveAs(null); setError(''); }}><form className="preset-create-dialog tts-preset-create-dialog" onSubmit={event => { event.preventDefault(); event.stopPropagation(); void create(); }}>
      <label>{text('预设名称', 'Preset name')}<input autoFocus aria-label={text('预设名称', 'Preset name')} value={name} disabled={busy} required onChange={event => setName(event.target.value)}/></label>
      <label>{text('用途与说明', 'Description')}<input aria-label={text('用途与说明', 'Description')} value={description} disabled={busy} onChange={event => setDescription(event.target.value)}/></label>
      {error && <p role="alert" className="studio-error">{error}</p>}
      <footer><button type="button" className="ui-btn" disabled={busy} onClick={() => setSaveAs(null)}>{text('取消', 'Cancel')}</button><button type="submit" className="ui-btn ui-btn-primary" disabled={busy || !name.trim()}>{busy ? text('保存中…', 'Saving…') : text('保存新预设', 'Save new preset')}</button></footer>
    </form></Dialog>}
  </>;
}
