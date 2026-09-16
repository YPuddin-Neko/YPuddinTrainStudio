import React from 'react';
import { Link } from 'react-router-dom';
import { Upload, FolderOpen, X, Loader2, CheckCircle2, Plus, ChevronUp } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { DatasetInfo } from '../../api/types';
import { PathInput } from '../../components/PathBrowser';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import CaptionFormatSelect from '../../components/CaptionFormatSelect';
import ConfigHelp from '../../components/ConfigHelp';
import { formatBytes, formatEta } from '../../utils/format';
import { filesFromDrop, filesFromSelection, type DatasetUploadFile } from '../../utils/datasetFiles';
import { formatDatasetImportError } from '../../utils/datasetImportErrors';
import { useDatasetImportProgress } from '../../utils/useDatasetImportProgress';
import DatasetImportProgress from '../../components/datasets/DatasetImportProgress';
import './project-data-import.css';

export default function ProjectDataImport({ projectId, versionId, onImported, defaultIsReg = false, captionFormats }: { projectId: string; versionId?: string; onImported: () => void; defaultIsReg?: boolean; captionFormats?: readonly string[] }) {
  const text = useWorkspaceText();
  const repeatsId = React.useId();
  const [mode, setMode] = React.useState<'upload' | 'path'>('upload');
  const [files, setFiles] = React.useState<DatasetUploadFile[]>([]);
  const [reading, setReading] = React.useState(false);
  const [path, setPath] = React.useState('');
  const [repeats, setRepeats] = React.useState(1);
  const isReg = defaultIsReg;
  const [captionExt, setCaptionExt] = React.useState('auto');
  const [priorWeight, setPriorWeight] = React.useState(1);
  const [classPrompt, setClassPrompt] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [dragging, setDragging] = React.useState(false);
  const [error, setError] = React.useState('');
  const [created, setCreated] = React.useState<DatasetInfo[]>([]);
  const [showForm, setShowForm] = React.useState(true);
  const selectionGeneration = React.useRef(0);
  const importGeneration = React.useRef(0);
  const healthRequest = React.useRef<AbortController | null>(null);
  const { operation, start: startProgress, finish: finishProgress, reset: resetProgress } = useDatasetImportProgress(projectId);
  React.useEffect(() => () => { selectionGeneration.current += 1; importGeneration.current += 1; healthRequest.current?.abort(); }, [projectId, versionId]);
  const fileInput = React.useRef<HTMLInputElement>(null);
  const folderInput = React.useRef<HTMLInputElement>(null);
  const inputClass = 'project-import-input';
  const selectFiles = (incoming: DatasetUploadFile[]) => {
    resetProgress();
    setError(''); setCreated([]);
    setFiles(incoming);
    if (fileInput.current) fileInput.current.value = '';
    if (folderInput.current) folderInput.current.value = '';
  };
  const chooseFiles = (incoming: File[]) => {
    try { selectFiles(filesFromSelection(incoming)); }
    catch (failure) { setError(formatApiError(failure)); }
  };
  const dropFiles = async (transfer: DataTransfer) => {
    const generation = ++selectionGeneration.current;
    resetProgress();
    setReading(true); setError(''); setCreated([]);
    try {
      const selected = await filesFromDrop(transfer);
      if (generation === selectionGeneration.current) selectFiles(selected);
    } catch (failure) {
      if (generation === selectionGeneration.current) setError(formatApiError(failure));
    } finally { if (generation === selectionGeneration.current) setReading(false); }
  };
  const locked = busy || reading;
  const folderName = mode === 'path' ? path.replace(/\\/g,'/').split('/').filter(Boolean).pop() : files[0]?.relativePath.split('/').slice(0,-1)[0];
  const autoName = files.length === 1 && files[0].relativePath.toLowerCase().endsWith('.zip') ? files[0].file.name.replace(/\.zip$/i, '') : folderName || '';
  const detectedRepeats = folderName?.match(/^([1-9]\d{0,5})[_-]/)?.[1];
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (locked || !Number.isInteger(repeats) || repeats < 1 || (mode === 'upload' ? files.length === 0 : !path.trim())) return;
    const generation = ++importGeneration.current;
    const current = () => generation === importGeneration.current;
    setBusy(true); setError(''); setCreated([]);
    try {
      const progressId = startProgress(mode);
      let result: DatasetInfo & { datasets?: DatasetInfo[] };
      if (mode === 'upload') {
        const form = new FormData();
        files.forEach(({file, relativePath}) => form.append('files', file, relativePath));
        form.append('name', autoName); form.append('repeats', String(repeats));
        form.append('is_reg', String(isReg)); form.append('prior_weight', String(priorWeight));
        form.append('class_prompt', classPrompt.trim()); form.append('caption_ext', captionExt);
        result = await apiClient.post<DatasetInfo>(`/projects/${projectId}/datasets/upload`, form, { params: { version_id: versionId, progress_id: progressId }, silent: true });
        if (!current()) return;
        setFiles([]);
        if (fileInput.current) fileInput.current.value = '';
        if (folderInput.current) folderInput.current.value = '';
      } else {
        result = await apiClient.post<DatasetInfo>(`/projects/${projectId}/datasets`, {
          path: path.trim(), repeats, caption_ext: captionExt.trim(), is_reg: isReg, prior_weight: priorWeight, class_prompt: classPrompt.trim() || null,
        }, { params: { version_id: versionId, progress_id: progressId }, silent: true });
        if (!current()) return;
        setPath('');
      }
      finishProgress('completed');
      setCreated(result.datasets || [result]); setShowForm(false); onImported();
    } catch (failure) {
      if (!current()) return;
      finishProgress('failed');
      const isNetworkFailure = failure instanceof Error && /^(Failed to fetch|Load failed|NetworkError when attempting to fetch resource\.?)$/i.test(failure.message);
      if (mode === 'upload' && isNetworkFailure) {
        const controller = new AbortController();
        healthRequest.current = controller;
        const timeout = window.setTimeout(() => controller.abort(), 3000);
        try {
          await apiClient.get('/health', { silent: true, signal: controller.signal });
          if (!current()) return;
          setError(text('训练服务仍可连接，但文件未能发送。请确认文件没有被移动或删除，重新选择文件夹后重试。', 'The training service is reachable, but the files could not be sent. Check that they have not moved or been deleted, then select the folder again.'));
        } catch { if (current()) setError(text('上传连接已中断，所选文件已保留。请检查训练服务与网络连接后重试。', 'The upload connection was interrupted. Your selection is retained. Check the training service and network, then retry.')); }
        finally { window.clearTimeout(timeout); if (healthRequest.current === controller) healthRequest.current = null; }
      } else setError(formatDatasetImportError(failure));
    }
    finally { if (current()) setBusy(false); }
  };

  return <form onSubmit={submit} className="project-data-import" data-testid="project-data-import" aria-busy={locked} data-completed={!showForm}>
    <header className="project-import-heading">
      <h3>{defaultIsReg ? text('添加已有正则图', 'Add existing regularization images') : text('添加训练图片', 'Add training images')}</h3>
      {showForm ? <div className="project-import-modes" role="group" aria-label={text('数据导入方式', 'Data import method')}>
        {[['upload', text('上传文件或文件夹', 'Upload files or folders')], ['path', text('从训练电脑导入', 'Import from training computer')]].map(([key, label]) => <button key={key} type="button" disabled={locked} onClick={() => { setMode(key as typeof mode); setError(''); setCreated([]); resetProgress(); }} aria-pressed={mode === key}>{label}</button>)}
      </div> : <button type="button" className="project-import-button" onClick={() => setShowForm(true)}><Plus size={16}/>{text('继续添加', 'Add more')}</button>}
    </header>
    {operation && operation.state !== 'completed' && <DatasetImportProgress operation={operation}/>}
    {error && <div role="alert" className="project-import-message project-import-error">{error}</div>}
    {created.length > 0 && <div role="status" className="project-import-message project-import-success"><CheckCircle2 size={20}/><div><strong>{text(`已导入当前版本，共 ${created.length} 组图片。`, `Imported ${created.length} image groups into this version.`)}</strong><div className="project-import-result-links">{created.map(dataset => <Link key={dataset.source.id} to={`/datasets/${dataset.source.id}`}>{created.length === 1 ? text('查看图片与标签', 'Review images and captions') : dataset.source.path.replace(/\\/g, '/').split('/').pop()}</Link>)}</div></div>{operation && <span className="project-import-elapsed">{text('用时', 'Elapsed')} {operation.elapsed < 1 ? text('不足 1 秒', '<1s') : formatEta(operation.elapsed)}</span>}</div>}
    {showForm && <>
    {mode === 'upload' ? <>
      <div className="project-import-dropzone" data-testid="dataset-dropzone" data-dragging={dragging} onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = locked ? 'none' : 'copy'; if (!locked) setDragging(true); }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false); }} onDrop={event => { event.preventDefault(); event.stopPropagation(); setDragging(false); if (!locked) void dropFiles(event.dataTransfer); }}>
        <div className="project-import-drop-copy">{reading ? <Loader2 size={20} className="animate-spin"/> : <Upload size={20}/>}<div><strong>{reading ? text('正在读取文件夹…', 'Reading folders…') : text('拖入文件夹、图片或 ZIP 压缩包', 'Drop folders, images or a ZIP archive')}</strong><p>{text('已有文件夹按原结构同步；只有散图片时自动创建目录。', 'Existing folders keep their structure. Loose images get a new folder automatically.')}</p></div></div>
        <div className="project-import-file-actions">
          <button type="button" disabled={locked} onClick={() => folderInput.current?.click()} className="project-import-button project-import-primary"><FolderOpen size={14}/>{text('选择文件夹', 'Choose folder')}</button>
          <button type="button" disabled={locked} onClick={() => fileInput.current?.click()} className="project-import-button">{text('选择文件 / ZIP', 'Choose files / ZIP')}</button>
        </div>
        <input ref={fileInput} hidden type="file" multiple accept="image/*,.txt,.json,.mask,.zip" aria-label={text('选择训练文件', 'Choose training files')} disabled={locked} onChange={event => chooseFiles(Array.from(event.target.files || []))}/>
        <input ref={folderInput} hidden type="file" multiple {...{ webkitdirectory: '' }} aria-label={text('选择训练文件夹', 'Choose training folder')} disabled={locked} onChange={event => chooseFiles(Array.from(event.target.files || []))}/>
      </div>
      <p className="project-import-limit">{text('同名 TXT / JSON 标签和遮罩一起导入 · 单次最多 2 GiB、5,000 个文件', 'Matching TXT / JSON captions and masks are included · Up to 2 GiB and 5,000 files per upload')}</p>
      {files.length > 0 && <div className="project-import-selected"><div className="project-import-selection-summary"><span>{text(`已选 ${files.length} 个文件`, `${files.length} files selected`)} · {formatBytes(files.reduce((total, item) => total + item.file.size, 0))}</span><button type="button" disabled={locked} onClick={() => selectFiles([])}>{text('清空选择', 'Clear selection')}</button></div>
        <ul className="project-import-files">{files.slice(0, 50).map(({file, relativePath}, index) => <li key={relativePath}><span title={relativePath}>{relativePath}</span><small>{formatBytes(file.size)}</small><button type="button" disabled={locked} aria-label={text(`移除 ${relativePath}`, `Remove ${relativePath}`)} onClick={() => setFiles(current => current.filter((_, i) => i !== index))}><X size={14}/></button></li>)}</ul>
        {files.length > 50 && <p className="project-import-limit">{text(`仅预览前 50 项，全部 ${files.length} 个文件都会导入。`, `Showing the first 50 entries; all ${files.length} files will be imported.`)}</p>}
      </div>}
    </> : <div className="project-import-location">
      <p>{text('选择运行训练服务的电脑上的目录。浏览器所在电脑的文件，请使用“上传文件或文件夹”。', 'Choose a folder on the computer running the training service. For files on this browser’s computer, use Upload files or folders.')}</p>
      <fieldset disabled={locked} role="group" aria-label={text('训练图片文件夹路径', 'Training image folder path')}><PathInput value={path} onChange={value => { setPath(value); resetProgress(); setCreated([]); }} placeholder={text('例如 D:\\训练素材\\角色图', 'For example D:\\training-data\\character')}/></fieldset>
    </div>}
    <div className="project-import-configuration">
      <details className="project-import-options"><summary>{text('导入选项', 'Import options')}</summary>
        <div className="project-import-fields">
          <label className="project-import-field project-import-caption">{text('标签格式', 'Caption format')}<CaptionFormatSelect value={captionExt} onChange={setCaptionExt} disabled={busy} formats={captionFormats}/></label>
          <div className="project-import-field project-import-repeats"><div className="project-import-field-label"><label htmlFor={repeatsId}>{text('每张图片重复次数', 'Repeats per image')}</label><ConfigHelp label={text('重复次数说明','Repeats help')}>{text('默认每轮使用每张图一次。次数越高，这组图片的训练占比越大，不复制文件。可从形如 5_character 的目录名读取 5；没有命名约定时不猜测。','Each image is used once per epoch by default. More repeats increase this dataset’s share without copying files. A folder named 5_character can suggest 5 repeats; otherwise no value is inferred.')}</ConfigHelp></div><input id={repeatsId} className={inputClass} type="number" min="1" step="1" required value={repeats} onChange={event => setRepeats(Number(event.target.value))} disabled={busy}/></div>
        </div>
        {detectedRepeats && Number(detectedRepeats)!==repeats && <button type="button" disabled={busy} onClick={()=>setRepeats(Number(detectedRepeats))}>{text(`目录名检测到重复 ${detectedRepeats} 次，应用`, `Folder name suggests ${detectedRepeats} repeats — apply`)}</button>}
        {isReg && <div className="project-import-reg-options"><label className="project-import-field project-import-prompt">{text('类别提示词', 'Class prompt')}<input className={inputClass} disabled={busy} value={classPrompt} onChange={event => setClassPrompt(event.target.value)}/></label><label className="project-import-field project-import-prior">{text('正则损失权重', 'Regularization loss weight')}<input className={inputClass} type="number" min="0" step="0.1" disabled={busy} value={priorWeight} onChange={event => setPriorWeight(Number(event.target.value))}/></label></div>}
      </details>
      <button type="submit" disabled={locked || !Number.isInteger(repeats) || repeats < 1 || (mode === 'upload' ? files.length === 0 : !path.trim())} className="project-import-button project-import-primary project-import-submit">{busy && <Loader2 size={14} className="animate-spin"/>}{busy ? text('正在导入…', 'Importing…') : text('导入当前版本', 'Import into this version')}</button>
      {created.length > 0 && <button type="button" className="project-import-button" disabled={locked} onClick={() => setShowForm(false)}><ChevronUp size={14}/>{text('收起', 'Collapse')}</button>}
    </div>
    </>}
  </form>;
}
