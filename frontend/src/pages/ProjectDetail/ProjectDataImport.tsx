import React from 'react';
import { Link } from 'react-router-dom';
import { Upload, FolderOpen, X, Loader2, CheckCircle2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { DatasetInfo } from '../../api/types';
import { PathInput } from '../../components/PathBrowser';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { formatBytes } from '../../utils/format';
import './project-data-import.css';

export default function ProjectDataImport({ projectId, versionId, onImported, defaultIsReg = false }: { projectId: string; versionId?: string; onImported: () => void; defaultIsReg?: boolean }) {
  const text = useWorkspaceText();
  const [mode, setMode] = React.useState<'upload' | 'path'>('upload');
  const [files, setFiles] = React.useState<File[]>([]);
  const [name, setName] = React.useState('');
  const [path, setPath] = React.useState('');
  const [repeats, setRepeats] = React.useState(1);
  const [isReg, setIsReg] = React.useState(defaultIsReg);
  const [captionExt, setCaptionExt] = React.useState('.txt');
  const [priorWeight, setPriorWeight] = React.useState(1);
  const [classPrompt, setClassPrompt] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [dragging, setDragging] = React.useState(false);
  const [error, setError] = React.useState('');
  const [created, setCreated] = React.useState<DatasetInfo | null>(null);
  const fileInput = React.useRef<HTMLInputElement>(null);
  const folderInput = React.useRef<HTMLInputElement>(null);
  const inputClass = 'project-import-input';
  const selectFiles = (incoming: File[]) => {
    setError(''); setCreated(null);
    setFiles(incoming);
    if (fileInput.current) fileInput.current.value = '';
    if (folderInput.current) folderInput.current.value = '';
  };
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (busy || !Number.isInteger(repeats) || repeats < 1 || (mode === 'upload' ? files.length === 0 : !path.trim())) return;
    setBusy(true); setError(''); setCreated(null);
    try {
      let result: DatasetInfo;
      if (mode === 'upload') {
        const form = new FormData();
        files.forEach((file) => form.append('files', file, file.webkitRelativePath || file.name));
        form.append('name', name.trim()); form.append('repeats', String(repeats));
        form.append('is_reg', String(isReg)); form.append('prior_weight', String(priorWeight));
        form.append('class_prompt', classPrompt.trim()); form.append('caption_ext', '.txt');
        result = await apiClient.post<DatasetInfo>(`/projects/${projectId}/datasets/upload`, form, { params: { version_id: versionId }, silent: true });
        setFiles([]);
        if (fileInput.current) fileInput.current.value = '';
        if (folderInput.current) folderInput.current.value = '';
      } else {
        result = await apiClient.post<DatasetInfo>(`/projects/${projectId}/datasets`, {
          path: path.trim(), repeats, caption_ext: captionExt.trim(), is_reg: isReg, prior_weight: priorWeight, class_prompt: classPrompt.trim() || null,
        }, { params: { version_id: versionId }, silent: true });
        setPath('');
      }
      setCreated(result); onImported();
    } catch (error) { setError(formatApiError(error)); }
    finally { setBusy(false); }
  };

  return <form onSubmit={submit} className="project-data-import" data-testid="project-data-import" aria-busy={busy}>
    <header className="project-import-heading">
      <h3>{defaultIsReg ? text('添加已有正则图', 'Add existing regularization images') : text('添加训练图片', 'Add training images')}</h3>
      <div className="project-import-modes" role="group" aria-label={text('数据导入方式', 'Data import method')}>
        {[['upload', text('从电脑上传', 'Upload from computer')], ['path', text('导入本机目录', 'Import a server folder')]].map(([key, label]) => <button key={key} type="button" disabled={busy} onClick={() => { setMode(key as typeof mode); setError(''); }} aria-pressed={mode === key}>{label}</button>)}
      </div>
    </header>
    {error && <div role="alert" className="project-import-message project-import-error">{error}</div>}
    {created && <div role="status" className="project-import-message project-import-success"><CheckCircle2 size={15}/>{text('已添加到当前版本。', 'Added to this version.')}<Link to={`/datasets/${created.source.id}`}>{text('查看图片与标签', 'Review images and captions')}</Link></div>}
    {mode === 'upload' ? <>
      <div className="project-import-dropzone" data-dragging={dragging} onDragOver={event => { event.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); if (!busy) selectFiles(Array.from(event.dataTransfer.files)); }}>
        <div className="project-import-drop-copy"><Upload size={20}/><div><strong>{text('拖入图片、标签文件，或一个 ZIP 压缩包', 'Drop images, captions, or one ZIP archive')}</strong><p>{text('图片、同名 .txt / 遮罩按目录导入当前版本；最多 2 GiB、5,000 个文件。', 'Import images and matching .txt / masks with their folders into this version. Up to 2 GiB / 5,000 files.')}</p></div></div>
        <div className="project-import-file-actions">
          <button type="button" disabled={busy} onClick={() => fileInput.current?.click()} className="project-import-button project-import-primary">{text('选择图片 / ZIP', 'Choose images / ZIP')}</button>
          <button type="button" disabled={busy} onClick={() => folderInput.current?.click()} className="project-import-button"><FolderOpen size={14}/>{text('选择整个文件夹', 'Choose a folder')}</button>
        </div>
        <input ref={fileInput} className="sr-only" type="file" multiple accept="image/*,.txt,.zip" aria-label={text('选择训练文件', 'Choose training files')} disabled={busy} onChange={event => selectFiles(Array.from(event.target.files || []))}/>
        <input ref={folderInput} className="sr-only" type="file" multiple {...{ webkitdirectory: '' }} aria-label={text('选择训练文件夹', 'Choose training folder')} disabled={busy} onChange={event => selectFiles(Array.from(event.target.files || []))}/>
      </div>
      {files.length > 0 && <div className="project-import-selected"><div className="project-import-selection-summary"><span>{text(`已选 ${files.length} 个文件`, `${files.length} files selected`)} · {formatBytes(files.reduce((total, file) => total + file.size, 0))}</span><button type="button" disabled={busy} onClick={() => selectFiles([])}>{text('清空选择', 'Clear selection')}</button></div>
        <ul className="project-import-files">{files.map((file, index) => <li key={`${file.name}-${index}`}><span title={file.webkitRelativePath || file.name}>{file.webkitRelativePath || file.name}</span><small>{formatBytes(file.size)}</small><button type="button" disabled={busy} aria-label={text(`移除 ${file.name}`, `Remove ${file.name}`)} onClick={() => setFiles(current => current.filter((_, i) => i !== index))}><X size={14}/></button></li>)}</ul>
      </div>}
    </> : <div className="project-import-location">
      <p>{text('选择训练器电脑上的文件夹，将复制到当前版本。其他电脑的文件请用“从电脑上传”。', 'This trainer folder will be copied into the current version. Use Upload from computer for files on another machine.')}</p>
      <fieldset disabled={busy} role="group" aria-label={text('训练图片文件夹路径', 'Training image folder path')}><PathInput value={path} onChange={setPath} placeholder={text('例如 D:\\训练素材\\角色图', 'For example D:\\training-data\\character')}/></fieldset>
    </div>}
    <div className="project-import-configuration">
      <div className="project-import-options">
        <div className="project-import-fields">
          {mode === 'upload' ? <label className="project-import-field project-import-name">{text('数据集名称（可选）', 'Dataset name (optional)')}<input className={inputClass} value={name} onChange={event => setName(event.target.value)} disabled={busy} placeholder={text('例如：角色正面照', 'For example: character portraits')}/></label> : <label className="project-import-field project-import-caption">{text('标签文件扩展名', 'Caption file extension')}<input className={inputClass} value={captionExt} onChange={event => setCaptionExt(event.target.value)} required disabled={busy}/></label>}
          <label className="project-import-field project-import-repeats">{text('每张图片重复次数', 'Repeats per image')}<input className={inputClass} type="number" min="1" step="1" required value={repeats} onChange={event => setRepeats(Number(event.target.value))} disabled={busy}/></label>
          <label className="project-import-reg"><input type="checkbox" disabled={busy || defaultIsReg} checked={isReg} onChange={event => setIsReg(event.target.checked)}/>{text('这是正则化数据集（先验保持）', 'Regularization dataset (prior preservation)')}</label>
        </div>
        {isReg && <div className="project-import-reg-options"><label className="project-import-field project-import-prompt">{text('类别提示词', 'Class prompt')}<input className={inputClass} disabled={busy} value={classPrompt} onChange={event => setClassPrompt(event.target.value)}/></label><label className="project-import-field project-import-prior">{text('先验保持权重', 'Prior preservation weight')}<input className={inputClass} type="number" min="0" step="0.1" disabled={busy} value={priorWeight} onChange={event => setPriorWeight(Number(event.target.value))}/></label></div>}
      </div>
      <button type="submit" disabled={busy || repeats < 1 || (mode === 'upload' ? files.length === 0 : !path.trim())} className="project-import-button project-import-primary project-import-submit">{busy && <Loader2 size={14} className="animate-spin"/>}{busy ? text('正在传输并登记，请稍候…', 'Transferring and registering…') : mode === 'upload' ? text('上传并添加到项目', 'Upload and add to project') : text('导入文件夹', 'Import folder')}</button>
    </div>
  </form>;
}
