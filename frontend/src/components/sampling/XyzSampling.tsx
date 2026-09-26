import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowDown, ArrowRight, ChevronDown, Download, Grid2X2, Layers, Loader2, Play, RefreshCw, Square, X } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import { formatApiError } from '../../utils/errors';
import { formatTime } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import StudioSelect from '../StudioSelect';
import CheckboxSelect from '../CheckboxSelect';
import GpuDevicePicker from '../GpuDevicePicker';
import { gpuDeviceLabel } from '../../utils/gpuDevices';
import Dialog from '../Dialog';
import { axisCount, axisNames, parseAxis, type AxisKey, type XyzAxis, type XyzCell, type XyzOptions, type XyzRequest, type XyzTask, type SamplingValues } from './xyzTypes';
import './xyz-sampling.css';
import { SlidingIndicator } from '../motion';

const activeStatuses = new Set(['queued', 'scheduled', 'running', 'preparing', 'caching', 'cancelling']);
const imageUrl = (url: string) => url.startsWith('/api/') ? apiUrl(url.slice(4)) : url;
type AxisDraft = { key: AxisKey; raw: string };
const initialAxis = (options: XyzOptions): AxisDraft => options.checkpoints.length > 1
  ? { key: 'checkpoint', raw: options.checkpoints.slice(0, options.limits.max_axis_values).map(cp => cp.id).join(', ') }
  : { key: 'steps', raw: [Math.max(1, options.defaults.steps - 5), options.defaults.steps, options.defaults.steps + 5].join(', ') };

function compatibleValues(values: SamplingValues, options: XyzOptions): SamplingValues {
  if (options.training_mode !== 'full') return values;
  return { ...values, sampling_model_id: null, adapter_scale: 1,
    checkpoint_id: options.checkpoints.some(cp => cp.id === values.checkpoint_id)
      ? values.checkpoint_id : options.checkpoints[0]?.id || null };
}
function compatibleDrafts(drafts: (AxisDraft | null)[], options: XyzOptions) {
  const result = drafts.map(draft => {
    if (!draft || !options.axes.some(axis => axis.key === draft.key)) return null;
    if (draft.key !== 'checkpoint') return draft;
    const ids = parseAxis(draft.key, draft.raw).values.filter(id => options.checkpoints.some(cp => cp.id === id));
    return ids.length ? { ...draft, raw: ids.join(', ') } : null;
  });
  if (!result[0]) {
    result[0] = initialAxis(options);
    for (let index = 1; index < result.length; index++) if (result[index]?.key === result[0].key) result[index] = null;
  }
  return result;
}

function AxisEditor({ position, draft, onChange, options, used, disabled, weights }: { position: 'X' | 'Y' | 'Z'; draft: AxisDraft | null; onChange: (draft: AxisDraft | null) => void; options: XyzOptions; used: AxisKey[]; disabled: boolean; weights: string[] }) {
  const text = useWorkspaceText();
  const name = (key: AxisKey) => key === 'checkpoint' && options.training_mode === 'full' ? text('模型检查点', 'Model checkpoint') : text(...axisNames[key]);
  const title = position === 'X' ? text('X · 横向比较', 'X · Columns') : position === 'Y' ? text('Y · 纵向比较', 'Y · Rows') : text('Z · 分页比较', 'Z · Pages');
  const icon = position === 'X' ? <ArrowRight size={14}/> : position === 'Y' ? <ArrowDown size={14}/> : <Layers size={14}/>;
  const current = options.axes.find(axis => axis.key === draft?.key);
  const choices = draft?.key === 'checkpoint' ? options.checkpoints.map(cp => ({ value: cp.id, label: cp.name })) : current?.values?.map(value => ({ value: String(value), label: String(value) }));
  const values = draft ? parseAxis(draft.key, draft.raw).values.map(String) : [];
  return <fieldset className="xyz-axis" disabled={disabled}>
    <legend>{icon}{title}</legend>
    <StudioSelect aria-label={title} disabled={disabled} value={draft?.key || ''} onValueChange={value => {
      if (!value) { onChange(null); return; }
      const key = value as AxisKey;
      const axis = options.axes.find(item => item.key === key);
      const defaults = key === 'checkpoint' ? (weights.length > 1 ? weights : options.checkpoints.slice(0, 3).map(cp => cp.id)) : axis?.values?.slice(0, 3) || (key === 'steps' ? [8, 16, 24] : key === 'adapter_scale' ? [0.6, 0.8, 1] : key === 'cfg' ? [1, 3, 5] : key === 'seed' ? [42, 43, 44] : [1, 3]);
      onChange({ key, raw: defaults.join(', ') });
    }} options={[...(position !== 'X' ? [{ value: '', label: text('不使用', 'Off') }] : []), ...options.axes.map(axis => ({ value: axis.key, label: name(axis.key),
      // Weights move to this axis from another one; they are chosen once, in the setup above.
      disabled: axis.key === 'checkpoint' ? !options.checkpoints.length : used.includes(axis.key) }))]}/>
    {draft && (draft.key === 'checkpoint' ? <p className="xyz-axis-note">{text(`逐个对比上方“${name('checkpoint')}”中勾选的 ${values.length} 个权重。`, `Compares the ${values.length} weights ticked in “${name('checkpoint')}” above.`)}</p>
      : choices ? <div className="xyz-axis-values"><span>{text('勾选要对比的取值', 'Tick the values to compare')}</span><CheckboxSelect aria-label={`${title} · ${text('取值', 'Values')}`} disabled={disabled} max={options.limits.max_axis_values}
        values={values} options={choices} onValuesChange={next => onChange({ ...draft, raw: next.join(', ') })}/></div>
      : <label className="xyz-axis-values"><span>{text('按顺序填写，用逗号分隔', 'Enter values in order, separated by commas')}</span><input aria-label={`${title} · ${text('取值', 'Values')}`} value={draft.raw} onChange={event => onChange({ ...draft, raw: event.target.value })}/></label>)}
  </fieldset>;
}

export default function XyzSampling({ sourceJobId, readOnly = false, initialTaskId }: { sourceJobId: string; readOnly?: boolean; initialTaskId?: string }) {
  return <SamplingWorkspace key={sourceJobId} sourceJobId={sourceJobId} readOnly={readOnly} initialTaskId={initialTaskId}/>;
}

function SamplingWorkspace({ sourceJobId, readOnly, initialTaskId }: { sourceJobId: string; readOnly: boolean; initialTaskId?: string }) {
  const text = useWorkspaceText();
  const [options, setOptions] = React.useState<XyzOptions | null>(null);
  const [values, setValues] = React.useState<SamplingValues | null>(null);
  const [drafts, setDrafts] = React.useState<(AxisDraft | null)[]>([null, null, null]);
  const [history, setHistory] = React.useState<XyzTask[]>([]);
  const [selected, setSelected] = React.useState(initialTaskId || '');
  const [error, setError] = React.useState('');
  const [loading, setLoading] = React.useState(true);
  const [submitting, setSubmitting] = React.useState(false);
  const [gpuDevices, setGpuDevices] = React.useState<string[]>([]);
  const [gpuValid, setGpuValid] = React.useState(true);
  const [page, setPage] = React.useState(0);
  const [preview, setPreview] = React.useState<XyzCell | null>(null);
  const [revision, setRevision] = React.useState(0);
  const [collapsed, setCollapsed] = React.useState(false);
  const task = history.find(item => item.id === selected);
  const running = history.find(item => activeStatuses.has(item.status));
  const locked = readOnly || submitting;
  const fullModel = options?.training_mode === 'full';
  const name = (key: AxisKey) => key === 'checkpoint' && fullModel ? text('模型检查点', 'Model checkpoint') : text(...axisNames[key]);
  const source = `/jobs/${encodeURIComponent(sourceJobId)}/xyz`;
  const displayValue = (axis: XyzAxis | null | undefined, value: string | number | null) => axis?.key === 'checkpoint' ? options?.checkpoints.find(cp => cp.id === value)?.name || String(value ?? '') : String(value ?? '');

  React.useEffect(() => {
    const controller = new AbortController(); setLoading(true); setError('');
    void Promise.all([apiClient.get<XyzOptions>(`${source}/options`, { signal: controller.signal, silent: true }), apiClient.get<XyzTask[]>(source, { signal: controller.signal, silent: true })]).then(([next, rows]) => {
      if (controller.signal.aborted) return;
      if (next.training_mode === 'full') next = { ...next, axes: next.axes.filter(axis => axis.key !== 'adapter_scale'), sampling_models: [] };
      setOptions(next); setValues(previous => compatibleValues(previous || next.defaults, next));
      setDrafts(previous => compatibleDrafts(previous, next));
      setHistory(rows); setSelected(previous => rows.some(row => row.id === previous) ? previous : rows[0]?.id || '');
      if (!revision) setCollapsed(rows.length > 0);
    }).catch(err => { if (!controller.signal.aborted) setError(formatApiError(err)); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [source, revision]);

  const activeIds = history.filter(item => activeStatuses.has(item.status)).map(item => item.id).sort().join(',');
  React.useEffect(() => {
    if (!activeIds) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const updates = await Promise.all(activeIds.split(',').map(id => apiClient.get<XyzTask>(`/xyz/${encodeURIComponent(id)}`, { signal: controller.signal, silent: true })));
        if (!controller.signal.aborted) { setHistory(previous => previous.map(row => updates.find(update => update.id === row.id) || row)); setError(''); }
      } catch (err) { if (!controller.signal.aborted) setError(formatApiError(err)); }
      if (!controller.signal.aborted) timer = setTimeout(() => void poll(), 2000);
    };
    timer = setTimeout(() => void poll(), 700);
    return () => { controller.abort(); clearTimeout(timer); };
  }, [activeIds]);

  const axes = drafts.map(draft => draft ? parseAxis(draft.key, draft.raw) : null);
  const weightAxis = drafts.findIndex(draft => draft?.key === 'checkpoint');
  const weights = weightAxis >= 0 ? (axes[weightAxis]?.values || []).map(String) : values?.checkpoint_id ? [values.checkpoint_id] : [];
  const positionName = (index: number) => [text('X · 横向', 'X · columns'), text('Y · 纵向', 'Y · rows'), text('Z · 分页', 'Z · pages')][index];
  /** None is the base model alone, one is fixed, several are compared along an axis (Y, then Z, unless one already holds them). */
  const chooseWeights = (ids: string[]) => {
    if (!options) return;
    setValues(previous => previous && { ...previous, checkpoint_id: ids[0] || null });
    setDrafts(previous => {
      const current = previous.findIndex(draft => draft?.key === 'checkpoint');
      if (ids.length < 2) {
        if (current < 0) return previous;
        const next = previous.map(draft => draft?.key === 'checkpoint' ? null : draft);
        if (!next[0]) next[0] = initialAxis({ ...options, checkpoints: [] });
        return next;
      }
      const target = current >= 0 ? current : previous[1] == null ? 1 : previous[2] == null ? 2 : 2;
      return previous.map((draft, index) => index === target ? { key: 'checkpoint', raw: ids.join(', ') } : draft);
    });
  };
  const changeAxis = (index: number, draft: AxisDraft | null) => setDrafts(previous => {
    const next = previous.map((value, other) => other === index ? draft : draft?.key === 'checkpoint' && value?.key === 'checkpoint' ? null : value);
    // The weights may leave X; X always compares something.
    if (!next[0] && options) next[0] = initialAxis({ ...options, checkpoints: [] });
    return next;
  });
  const count = axisCount(axes);
  const axisInvalid = !axes[0] || axes.some(axis => axis && (!options?.axes.some(option => option.key === axis.key) || !axis.values.length || axis.values.length > (options?.limits.max_axis_values || 12) || axis.values.some(value => typeof value === 'number' && !Number.isFinite(value))));
  const checkpointMissing = fullModel && !options?.checkpoints.some(cp => cp.id === values?.checkpoint_id);
  const overLimit = count > (options?.limits.max_cells || 64);
  const tooManyPixels = !!values && count * values.width * values.height > (options?.limits.max_pixels || 64 * 1024 * 1024);
  const update = <K extends keyof SamplingValues>(key: K, value: SamplingValues[K]) => setValues(previous => previous && { ...previous, [key]: value });
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!values || !axes[0] || axisInvalid || checkpointMissing || overLimit || tooManyPixels || locked || !gpuValid) return;
    setSubmitting(true); setError('');
    try {
      const request: XyzRequest = { ...values, name: text('模型测试', 'Model testing'), gpu_devices: gpuDevices, x: axes[0], y: axes[1], z: axes[2] };
      const created = await apiClient.post<XyzTask>(source, request, { silent: true });
      setHistory(previous => [created, ...previous]); setSelected(created.id); setPage(0); setCollapsed(true);
    } catch (err) { setError(formatApiError(err)); }
    finally { setSubmitting(false); }
  };
  const cancel = async (item: XyzTask) => {
    setSubmitting(true); setError('');
    try { const updated = await apiClient.post<XyzTask>(`/xyz/${encodeURIComponent(item.id)}/cancel`, {}, { silent: true }); setHistory(previous => previous.map(row => row.id === item.id ? updated : row)); }
    catch (err) { setError(formatApiError(err)); } finally { setSubmitting(false); }
  };
  const stateLabel = (status: string) => ({ completed: text('已完成', 'Complete'), running: text('生成中', 'Generating'), queued: text('等待设备', 'Queued'), cancelled: text('已取消', 'Cancelled'), cancelling: text('正在取消', 'Cancelling'), failed: text('失败', 'Failed') }[status] || status);
  const request = task?.request;
  const xValues = request?.x.values || [];
  const yValues = request?.y?.values || [null];
  const zValues = request?.z?.values || [null];
  const cells = new Map((task?.manifest?.cells || []).filter(cell => cell.z === page).map(cell => [`${cell.x}:${cell.y}`, cell]));
  const grid = task?.manifest?.grids.find(item => item.z === page);
  const reuse = () => {
    if (!request || !options || locked) return;
    const { x, y, z, gpu_devices, ...fixed } = request;
    setGpuDevices(gpu_devices || []);
    delete fixed.name;
    setValues(compatibleValues(fixed, options));
    setDrafts(compatibleDrafts([x, y, z].map(axis => axis ? { key: axis.key, raw: axis.values.join(', ') } : null), options));
    setCollapsed(false);
    setError('');
  };

  if (!options || !values) return <div className="xyz-loading">{loading ? <><Loader2 size={20} className="animate-spin"/>{text('读取采样配置…', 'Loading sampling setup…')}</> : <><p role="alert">{error}</p><button type="button" className="ui-btn ui-btn-sm" onClick={() => setRevision(value => value + 1)}>{text('重试', 'Retry')}</button></>}</div>;
  return <section className="xyz-workspace" aria-label={text('模型测试', 'Model testing')}>
    {error && <div className="results-error" role="alert">{error}<button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" onClick={() => setError('')} aria-label={text('关闭错误提示', 'Dismiss error')}><X size={14}/></button></div>}
    <div className="xyz-layout">
      <form className="xyz-form" data-collapsed={collapsed} onSubmit={event => void submit(event)}>
        <header><Grid2X2 size={17}/><h3>{text('对比设置', 'Comparison setup')}</h3><span>{options.family.toUpperCase()}</span><button className="ui-link xyz-settings-toggle" type="button" aria-expanded={!collapsed} onClick={() => setCollapsed(value => !value)}>{collapsed ? text('展开设置', 'Show settings') : text('收起设置', 'Hide settings')}<ChevronDown size={14}/></button></header>
        <div className="xyz-form-body">
          <GpuDevicePicker value={gpuDevices} onChange={setGpuDevices} disabled={locked} onValidityChange={setGpuValid}/>
          <fieldset className="xyz-section" disabled={locked}>
            {!fullModel && <><label><span>{text('采样底模', 'Sampling base model')}</span><StudioSelect aria-label={text('采样底模', 'Sampling base model')} disabled={locked} value={values.sampling_model_id || ''} options={[{ value: '', label: text('沿用本次训练底模', 'Use training base model') }, ...options.sampling_models.map(model => ({ value: model.id, label: model.name }))]} onValueChange={value => {
              const turbo = options.sampling_models.find(model => model.id === value)?.variant === 'turbo';
              setValues(previous => previous && { ...previous, sampling_model_id: value || null, steps: turbo ? 8 : options.defaults.steps, cfg: turbo ? 0 : options.defaults.cfg, shift: null });
              if (drafts[0]?.key === 'steps') setDrafts(previous => [{ key: 'steps', raw: turbo ? '4, 8, 12' : initialAxis({ ...options, checkpoints: [] }).raw }, previous[1], previous[2]]);
            }}/></label>
            <Link className="ui-link xyz-model-link" to={`/settings/environment?tab=models&family=${encodeURIComponent(options.family)}`}>{text('管理与下载模型', 'Manage & download models')}<ArrowRight size={12}/></Link></>}
            <div className="xyz-weights"><span id="xyz-weights-label">{fullModel ? text('全量模型检查点', 'Full model checkpoint') : text('训练权重', 'Trained checkpoint')}</span>
              <CheckboxSelect aria-label={text('对比使用的训练权重', 'Checkpoints for comparison')} aria-describedby="xyz-weights-note" searchable disabled={locked || !options.checkpoints.length} max={options.limits.max_axis_values}
                placeholder={fullModel ? text('请选择检查点', 'Choose a checkpoint') : text('不加载训练权重（只看底模）', 'No trained weights (base model only)')}
                values={weights} options={options.checkpoints.map(cp => ({ value: cp.id, label: cp.name }))} onValuesChange={chooseWeights}/>
              <small id="xyz-weights-note">{weights.length > 1 ? text(`${weights.length} 个权重放在 ${positionName(weightAxis)} 逐个对比；可在下方把“${name('checkpoint')}”换到其他轴。`, `${weights.length} weights are compared along ${positionName(weightAxis)}; move “${name('checkpoint')}” to another axis below.`)
                : weights.length === 1 ? text('勾选多个权重即可逐个对比。', 'Tick several weights to compare them.')
                  : fullModel ? text('全量训练需要选择一个检查点。', 'Full training needs a checkpoint.') : text('未勾选时只用底模生成，便于和训练前对比。', 'With none ticked, only the base model is used — useful to compare with before training.')}</small>
            </div>
            <label><span>{text('提示词', 'Prompt')}</span><textarea aria-label={text('模型测试提示词', 'Model testing prompt')} required rows={3} value={values.prompt} onChange={event => update('prompt', event.target.value)}/></label>
          </fieldset>
          <div className="xyz-axes">{(['X', 'Y', 'Z'] as const).map((position, index) => <AxisEditor key={position} position={position} draft={drafts[index]} options={options} disabled={locked} weights={weights} used={drafts.filter((_, other) => other !== index).flatMap(draft => draft ? [draft.key] : [])} onChange={draft => changeAxis(index, draft)}/>)}</div>
          <details className="xyz-settings"><summary>{text('固定参数', 'Fixed parameters')}<span>{values.width} × {values.height} · Seed {values.seed}</span></summary><fieldset className="xyz-fields" disabled={locked}>
            {(['width', 'height', 'seed', 'steps', 'cfg', 'adapter_scale'] as const).filter(key => !fullModel || key !== 'adapter_scale').map(key => <label key={key}><span>{key === 'width' ? text('宽度', 'Width') : key === 'height' ? text('高度', 'Height') : name(key)}</span><input aria-label={`${text('固定', 'Fixed')} ${key}`} type="number" min={key === 'adapter_scale' ? -4 : key === 'cfg' || key === 'seed' ? 0 : 1} step={key === 'cfg' || key === 'adapter_scale' ? 0.1 : 1} required disabled={axes.some(axis => axis?.key === key)} value={values[key]} onChange={event => update(key, Number(event.target.value))}/></label>)}
            {(['sampler', 'scheduler'] as const).map(key => <label key={key}><span>{name(key)}</span><StudioSelect aria-label={`${text('固定', 'Fixed')} ${name(key)}`} disabled={locked || axes.some(axis => axis?.key === key)} value={values[key]} options={(options.axes.find(axis => axis.key === key)?.values || [values[key]]).map(value => ({ value: String(value), label: String(value) }))} onValueChange={value => update(key, value)}/></label>)}
            <label className="xyz-span"><span>{text('负面提示词', 'Negative prompt')}</span><textarea rows={2} value={values.negative} onChange={event => update('negative', event.target.value)}/></label>
          </fieldset></details>
        </div>
        <footer><p className={axisInvalid || checkpointMissing || overLimit || tooManyPixels ? 'xyz-invalid' : ''} aria-live="polite">{checkpointMissing ? text('当前训练尚未保存模型检查点，保存后才能生成对比图', 'Save a model checkpoint before generating a comparison') : axisInvalid ? text('请填写有效的轴取值', 'Enter valid axis values') : overLimit ? text(`一次最多 ${options.limits.max_cells} 张，请减少取值`, `Maximum ${options.limits.max_cells} cells per comparison`) : tooManyPixels ? text('网格总像素过多，请减少取值或降低尺寸', 'Too many pixels; reduce values or image dimensions') : text(`${axes[0]?.values.length || 0} 列 × ${axes[1]?.values.length || 1} 行 × ${axes[2]?.values.length || 1} 页，共 ${count} 张`, `${count} images · ${axes[0]?.values.length || 0} columns × ${axes[1]?.values.length || 1} rows × ${axes[2]?.values.length || 1} pages`)}</p><button className="ui-btn ui-btn-primary ui-btn-block" type="submit" disabled={locked || axisInvalid || checkpointMissing || overLimit || tooManyPixels || !gpuValid || !values.prompt.trim()}>{submitting ? <Loader2 size={15} className="animate-spin"/> : <Play size={15}/>} {running ? text('加入测试队列', 'Add to test queue') : text('生成对比图', 'Generate comparison')}</button></footer>
      </form>
      <div className="xyz-results">
        <header className="xyz-result-header"><label><span>{text('对比记录', 'Comparisons')}</span><StudioSelect searchable aria-label={text('对比记录', 'Comparisons')} value={selected} placeholder={text('尚未生成', 'No comparisons yet')} options={history.map(item => ({ value: item.id, label: `${formatTime(item.created_at)} · ${item.total} ${text('张', 'images')} · ${stateLabel(item.status)}` }))} onValueChange={value => { setSelected(value); setPage(0); }}/></label><button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" disabled={loading} onClick={() => setRevision(value => value + 1)} aria-label={text('刷新对比记录', 'Refresh comparisons')} title={text('刷新对比记录', 'Refresh comparisons')}><RefreshCw size={15}/></button></header>
        {task ? <>
          <div className="xyz-progress" role="status"><div><strong>{stateLabel(task.status)}</strong><span>{task.done} / {task.total}</span>{task.can_cancel && <button type="button" className="ui-btn ui-btn-sm" disabled={submitting || readOnly} onClick={() => void cancel(task)}><Square size={12}/>{text('取消生成', 'Cancel')}</button>}</div><progress max={Math.max(1, task.total)} value={task.done}/>{task.phase && activeStatuses.has(task.status) && <small>{{ loading: text('加载模型', 'Loading model'), encoding_text: text('处理提示词', 'Encoding prompts'), sampling: text('正在出图', 'Sampling'), decoding: text('解码图片', 'Decoding image') }[task.phase] || stateLabel(task.status)}{task.sample_steps ? text(` · 当前图片 ${task.sample_step || 0} / ${task.sample_steps} 步`, ` · Image ${task.sample_step || 0} / ${task.sample_steps} steps`) : ''}</small>}{task.error && <p role="alert" className="xyz-invalid">{task.error}</p>}</div>
          <div className="xyz-result-context"><span>{request?.gpu_devices?.length ? `${text('申请显卡', 'Requested GPU')}: ${request.gpu_devices.map(gpuDeviceLabel).join(', ')} · ` : ''}{request?.width} × {request?.height} · {text('提示词', 'Prompt')}: {request?.prompt}</span><button type="button" className="ui-btn ui-btn-sm" disabled={locked} onClick={reuse}>{text('复用参数', 'Reuse settings')}</button>{grid && <a className="ui-btn ui-btn-sm" href={imageUrl(grid.url)} download><Download size={14}/>{text('下载本页网格', 'Download grid')}</a>}</div>
          {request?.z && <nav className="xyz-pages ui-tabs" aria-label={text('Z 轴分页', 'Z axis pages')}>{zValues.map((value, index) => <button key={index} type="button" aria-current={page === index ? 'page' : undefined} onClick={() => setPage(index)}>{name(request.z!.key)} · {displayValue(request.z, value)}</button>)}<SlidingIndicator className="ui-tabs-indicator"/></nav>}
          <div className="xyz-grid-scroll" tabIndex={0} aria-label={text('对比网格，可横向滚动查看所有列', 'Comparison grid, scroll horizontally for all columns')}>
            <table className="xyz-grid" style={{ minWidth: 88 + xValues.length * 150 }}><thead><tr><th>{request?.y ? `${name(request.y.key)} ↓` : ''}<br/>{request ? `${name(request.x.key)} →` : ''}</th>{xValues.map((value, x) => <th key={x} title={displayValue(request?.x, value)}>{displayValue(request?.x, value)}</th>)}</tr></thead><tbody>{yValues.map((value, y) => <tr key={y}><th title={displayValue(request?.y, value)}>{displayValue(request?.y, value)}</th>{xValues.map((_, x) => { const cell = cells.get(`${x}:${y}`); return <td key={x}>{cell ? <button type="button" className="xyz-cell" onClick={() => setPreview(cell)} aria-label={text(`查看第 ${x + 1} 列第 ${y + 1} 行`, `View column ${x + 1}, row ${y + 1}`)}><img src={imageUrl(cell.url)} loading="lazy" alt={`${name(request!.x.key)} ${displayValue(request?.x, cell.x_value)}, ${request?.y ? `${name(request.y.key)} ${displayValue(request.y, cell.y_value)}` : ''}`} width={request?.width} height={request?.height}/></button> : <div className="xyz-cell-pending"><Grid2X2 size={19}/><span>{activeStatuses.has(task.status) ? text('等待生成', 'Waiting') : text('未生成', 'Not generated')}</span></div>}</td>; })}</tr>)}</tbody></table>
          </div>
        </> : <div className="xyz-empty"><Grid2X2 size={34}/><h3>{text('把差异放在一起看', 'Compare results side by side')}</h3><div className="xyz-empty-grid" aria-hidden="true">{Array.from({ length: 6 }, (_, index) => <span key={index}/>)}</div></div>}
      </div>
    </div>
    {preview && <Dialog title={`${text('模型测试图片', 'Model test image')} · ${displayValue(request?.x, preview.x_value)}`} wide onClose={() => setPreview(null)}><img className="xyz-full-image" src={imageUrl(preview.url)} alt={request?.prompt}/><div className="xyz-image-details"><span>Seed {preview.seed} · {preview.steps} {text('步', 'steps')} · CFG {preview.cfg} · {preview.sampler} / {preview.scheduler}{!fullModel && <> · LoRA {preview.adapter_scale}</>}</span><a className="ui-btn ui-btn-sm" href={imageUrl(preview.url)} download><Download size={14}/>{text('下载原图', 'Download image')}</a></div></Dialog>}
  </section>;
}
