import React from 'react';
import { Link } from 'react-router-dom';
import { Upload, FolderOpen, X, Loader2, CheckCircle2 } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { DatasetInfo } from '../../api/types';
import { PathInput } from '../../components/PathBrowser';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { formatBytes } from '../../utils/format';

export default function ProjectDataImport({ projectId, versionId, onImported }: { projectId: string; versionId?: string; onImported: () => void }) {
  const text = useWorkspaceText();
  const [mode, setMode] = React.useState<'upload' | 'path'>('upload');
  const [files, setFiles] = React.useState<File[]>([]);
  const [name, setName] = React.useState('');
  const [path, setPath] = React.useState('');
  const [repeats, setRepeats] = React.useState(1);
  const [isReg, setIsReg] = React.useState(false);
  const [captionExt, setCaptionExt] = React.useState('.txt');
  const [priorWeight, setPriorWeight] = React.useState(1);
  const [classPrompt, setClassPrompt] = React.useState('');
  const [busy, setBusy] = React.useState(false);
  const [dragging, setDragging] = React.useState(false);
  const [error, setError] = React.useState('');
  const [created, setCreated] = React.useState<DatasetInfo | null>(null);
  const fileInput = React.useRef<HTMLInputElement>(null);
  const folderInput = React.useRef<HTMLInputElement>(null);
  const inputClass = 'w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600';
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

  return <form onSubmit={submit} className="version-data-import" data-testid="project-data-import">
    <div><h3 className="text-base font-semibold">{text('添加训练图片', 'Add training images')}</h3><p className="mt-1 text-sm text-slate-500">{text('图片与同名标签、遮罩一起导入到当前版本。', 'Import images with matching captions and masks into this version.')}</p></div>
    <div className="flex flex-wrap gap-2" role="group" aria-label={text('数据导入方式', 'Data import method')}>
      {[['upload', text('从电脑上传', 'Upload from computer')], ['path', text('导入本机目录', 'Import a server folder')]].map(([key, label]) => <button key={key} type="button" disabled={busy} onClick={() => { setMode(key as typeof mode); setError(''); }} aria-pressed={mode === key} className={`rounded-lg px-3 py-2 text-sm border ${mode === key ? 'bg-blue-50 border-blue-400 text-blue-700 dark:bg-blue-950 dark:text-blue-300' : 'border-slate-200 dark:border-slate-600'}`}>{label}</button>)}
    </div>
    {error && <div role="alert" className="whitespace-pre-line rounded bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">{error}</div>}
    {created && <div role="status" className="flex flex-wrap items-center gap-2 rounded bg-green-50 p-3 text-sm text-green-700 dark:bg-green-950 dark:text-green-300"><CheckCircle2 className="h-4 w-4" />{text('已添加到项目，可在下方查看索引状态。', 'Added to the project. Check the indexing status below.')}<Link className="underline" to={`/datasets/${created.source.id}`}>{text('查看图片与标签', 'Review images and captions')}</Link></div>}
    {mode === 'upload' ? <>
      <div onDragOver={(e) => { e.preventDefault(); if (!busy) setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); setDragging(false); if (!busy) selectFiles(Array.from(e.dataTransfer.files)); }}
        className={`rounded-lg border border-dashed p-4 text-center ${dragging ? 'border-blue-500 bg-blue-50 dark:bg-blue-950' : 'border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-slate-900/50'}`}>
        <Upload className="mx-auto h-5 w-5 text-blue-500" /><p className="mt-2 font-medium text-sm">{text('拖入图片、标签文件，或一个 ZIP 压缩包', 'Drop images, captions, or one ZIP archive')}</p>
        <p className="mt-1 text-xs text-slate-500">{text('JPG / PNG / WebP / BMP / TIFF 与 .txt；目录结构会保留。单次最多 2 GiB、5,000 个文件。', 'JPG / PNG / WebP / BMP / TIFF and .txt; folder structure is preserved. Up to 2 GiB and 5,000 files per upload.')}</p>
        <div className="mt-4 flex flex-wrap justify-center gap-2">
          <button type="button" disabled={busy} onClick={() => fileInput.current?.click()} className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50">{text('选择图片 / ZIP', 'Choose images / ZIP')}</button>
          <button type="button" disabled={busy} onClick={() => folderInput.current?.click()} className="inline-flex items-center gap-2 rounded border border-slate-300 px-3 py-2 text-sm dark:border-slate-600"><FolderOpen className="h-4 w-4" />{text('选择整个文件夹', 'Choose a folder')}</button>
        </div>
        <input ref={fileInput} className="sr-only" type="file" multiple accept="image/*,.txt,.zip" aria-label={text('选择训练文件', 'Choose training files')} disabled={busy} onChange={(e) => selectFiles(Array.from(e.target.files || []))} />
        <input ref={folderInput} className="sr-only" type="file" multiple {...{ webkitdirectory: '' }} aria-label={text('选择训练文件夹', 'Choose training folder')} disabled={busy} onChange={(e) => selectFiles(Array.from(e.target.files || []))} />
      </div>
      {files.length > 0 && <div className="space-y-2"><div className="flex justify-between text-sm"><span>{text(`已选 ${files.length} 个文件`, `${files.length} files selected`)} · {formatBytes(files.reduce((total, file) => total + file.size, 0))}</span><button type="button" disabled={busy} onClick={() => selectFiles([])} className="text-slate-500 underline">{text('清空选择', 'Clear selection')}</button></div>
        <ul className="max-h-40 overflow-auto divide-y divide-slate-100 dark:divide-slate-700">{files.map((file, index) => <li key={`${file.name}-${index}`} className="flex items-center gap-2 py-1.5 text-xs"><span className="min-w-0 flex-1 truncate">{file.webkitRelativePath || file.name}</span><span className="text-slate-400">{formatBytes(file.size)}</span><button type="button" disabled={busy} aria-label={text(`移除 ${file.name}`, `Remove ${file.name}`)} onClick={() => setFiles((current) => current.filter((_, i) => i !== index))}><X className="h-4 w-4" /></button></li>)}</ul>
      </div>}
      <label className="block text-sm">{text('数据集名称（可选）', 'Dataset name (optional)')}<input className={`${inputClass} mt-1`} value={name} onChange={(e) => setName(e.target.value)} disabled={busy} placeholder={text('例如：角色正面照', 'For example: character portraits')} /></label>
    </> : <>
      <p className="text-sm text-slate-500">{text('选择运行训练器电脑上的文件夹，导入时会复制到当前版本。浏览器在另一台电脑时，请使用“从电脑上传”。', 'The trainer copies this folder into the current version. Use Upload from computer for files on another machine.')}</p>
      <div role="group" aria-label={text('训练图片文件夹路径', 'Training image folder path')}><PathInput value={path} onChange={setPath} placeholder={text('例如 D:\\训练素材\\角色图', 'For example D:\\training-data\\character')} /></div>
      <label className="block text-sm">{text('标签文件扩展名', 'Caption file extension')}<input className={`${inputClass} mt-1 max-w-40`} value={captionExt} onChange={(e) => setCaptionExt(e.target.value)} required /></label>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={isReg} onChange={(e) => setIsReg(e.target.checked)} />{text('这是正则化数据集（先验保持）', 'Regularization dataset (prior preservation)')}</label>
      {isReg && <label className="block text-sm">{text('类别提示词', 'Class prompt')}<input className={`${inputClass} mt-1`} value={classPrompt} onChange={(e) => setClassPrompt(e.target.value)} /></label>}
      {isReg && <label className="block text-sm">{text('先验保持权重', 'Prior preservation weight')}<input className={`${inputClass} mt-1 max-w-40`} type="number" min="0" step="0.1" value={priorWeight} onChange={(e) => setPriorWeight(Number(e.target.value))} /></label>}
    </>}
    <div className="flex flex-wrap items-end justify-between gap-3"><label className="text-sm">{text('每张图片重复次数', 'Repeats per image')}<input className={`${inputClass} mt-1 max-w-28 block`} type="number" min="1" step="1" required value={repeats} onChange={(e) => setRepeats(Number(e.target.value))} disabled={busy} /></label>
      <button type="submit" disabled={busy || repeats < 1 || (mode === 'upload' ? files.length === 0 : !path.trim())} className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50">{busy && <Loader2 className="h-4 w-4 animate-spin" />}{busy ? text('正在传输并登记，请稍候…', 'Transferring and registering…') : mode === 'upload' ? text('上传并添加到项目', 'Upload and add to project') : text('导入文件夹', 'Import folder')}</button>
    </div>
  </form>;
}
