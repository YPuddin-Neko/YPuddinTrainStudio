import React from 'react';
import { FolderOpen, ImagePlus, Loader2, X } from 'lucide-react';
import { apiClient } from '../../api/client';
import { useFamilies } from '../../api/hooks/useFamilies';
import Dialog from '../../components/Dialog';
import StudioSelect from '../../components/StudioSelect';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { trainingFamilyOptions } from '../../utils/trainingFamilies';
import { useTranslation } from 'react-i18next';
import { categoryLabel, coverSource, PROJECT_CATEGORIES, type GalleryProject } from './projectGallery';

export function ProjectCover({ source, name }: { source?: string | null; name: string }) {
  const text = useWorkspaceText();
  const [failed, setFailed] = React.useState(false);
  React.useEffect(() => setFailed(false), [source]);
  return source && !failed ? <img src={source} alt={text(`${name} 的封面`, `Cover for ${name}`)} loading="lazy" onError={() => setFailed(true)}/>
    : <div className="project-cover-placeholder"><FolderOpen size={27} aria-hidden="true"/><span>{source ? text('封面暂不可用', 'Cover unavailable') : text('未设置封面', 'No cover')}</span></div>;
}

export default function ProjectEditor({ project, categories, onClose, onSaved, onPartial }: {
  project?: GalleryProject; categories: string[]; onClose: () => void; onSaved: (project: GalleryProject) => void; onPartial: (project: GalleryProject) => void;
}) {
  const text = useWorkspaceText();
  const { i18n } = useTranslation();
  const english = i18n.resolvedLanguage?.startsWith('en') || false;
  const { data: families = [], isError: familiesError, refetch: reloadFamilies } = useFamilies();
  const [name, setName] = React.useState(project?.name || '');
  const [id, setId] = React.useState(project?.id || '');
  const [note, setNote] = React.useState(project?.note || '');
  const [category, setCategory] = React.useState(project?.category || '');
  const [customCategory, setCustomCategory] = React.useState(false);
  const [family, setFamily] = React.useState('anima');
  const familyOptions = trainingFamilyOptions(families, english);
  const familyAvailable = familyOptions.some(option => option.value === family && !option.disabled);
  const [file, setFile] = React.useState<File | null>(null);
  const [preview, setPreview] = React.useState<string | null>(null);
  const [removeCover, setRemoveCover] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const busyRef = React.useRef(false);
  const [idTouched, setIdTouched] = React.useState(false);
  const [error, setError] = React.useState('');
  const [savedProject, setSavedProject] = React.useState<GalleryProject | null>(project || null);
  const [partial, setPartial] = React.useState(false);
  const savedMetadata = React.useRef<string | null>(null);
  const uploadInput = React.useRef<HTMLInputElement>(null);
  React.useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const idError = !id ? text('请填写项目 ID。', 'Enter a project ID.') : id.length > 64 ? text('项目 ID 最多 64 个字符。', 'Project ID must be at most 64 characters.') : !/^[A-Za-z0-9_]+$/.test(id) ? text('项目 ID 只能包含英文字母、数字和下划线。', 'Use only ASCII letters, digits and underscores in the project ID.') : '';
  const categoryOptions = [...new Set([...PROJECT_CATEGORIES, ...categories, ...(project?.category ? [project.category] : [])])];
  const activeCover = preview || (!removeCover && savedProject?.cover_url ? coverSource(savedProject.cover_url) : null);
  const chooseFile = (selected?: File) => {
    if (!selected) return;
    if (selected.size > 8 * 1024 * 1024) { setError(text('封面不能超过 8 MiB。', 'Cover must be no larger than 8 MiB.')); return; }
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(selected.type)) { setError(text('请选择 JPEG、PNG 或 WebP 图片。', 'Choose a JPEG, PNG or WebP image.')); return; }
    setFile(selected); setRemoveCover(false); setError('');
  };
  const save = async () => {
    setIdTouched(true);
    if (busyRef.current || !name.trim() || (!project && (idError || !familyAvailable)) || category.trim().length > 64 || (customCategory && !category.trim())) return;
    busyRef.current = true; setBusy(true); setError('');
    let currentProject = savedProject;
    const metadata = { name: name.trim(), note: note.trim(), category: category.trim() || null };
    const signature = JSON.stringify(metadata);
    try {
      if (!currentProject) {
        currentProject = await apiClient.post<GalleryProject>('/projects', { id, ...metadata, family }, { silent: true });
        setSavedProject(currentProject); savedMetadata.current = signature;
      } else if (savedMetadata.current !== signature) {
        currentProject = await apiClient.patch<GalleryProject>(`/projects/${currentProject.id}`, metadata, { silent: true });
        setSavedProject(currentProject); savedMetadata.current = signature;
      }
      if (file) {
        const body = new FormData(); body.append('file', file);
        currentProject = await apiClient.post<GalleryProject>(`/projects/${currentProject.id}/cover`, body, { silent: true });
      } else if (removeCover && currentProject.cover_url) {
        currentProject = await apiClient.delete<GalleryProject>(`/projects/${currentProject.id}/cover`, { silent: true });
      }
      onSaved(currentProject);
    } catch (failure) {
      const details = formatApiError(failure);
      if (currentProject && savedMetadata.current === signature) {
        setPartial(true); onPartial(currentProject);
        setError(`${text('项目信息已保存，封面未保存。可重试，或移除本次封面后保存。', 'Project details were saved, but the cover was not. Retry, or remove the pending cover and save.')}\n${details}`);
      } else setError(details);
    } finally { busyRef.current = false; setBusy(false); }
  };
  return <Dialog title={project ? text('编辑项目', 'Edit project') : text('新建项目', 'New project')} onClose={onClose} closeDisabled={busy}>
    <form className="project-editor" onSubmit={event => { event.preventDefault(); void save(); }} data-testid={project ? 'edit-project-modal' : 'create-project-modal'} aria-busy={busy}>
      {error && <p role="alert" className="project-editor-error">{error}</p>}
      {partial && !project && <p role="status" className="project-editor-note">{text(`项目 ${savedProject?.id} 已创建；关闭窗口会保留此项目，重试不会重复创建。`, `Project ${savedProject?.id} exists. Closing keeps it; retrying will not create a duplicate.`)}</p>}
      <fieldset disabled={busy} className="project-editor-fields">
        <div className="project-cover-editor"><div className="project-cover-preview"><ProjectCover source={activeCover} name={name || text('项目', 'Project')}/></div><div className="project-cover-controls">
          <strong>{text('项目封面', 'Project cover')}</strong><p>{text('手动选择图片，仅用于项目预览。', 'Choose an image for this project preview.')}</p>
          <input ref={uploadInput} className="project-cover-file" type="file" tabIndex={-1} accept="image/jpeg,image/png,image/webp" aria-label={text('上传项目封面', 'Upload project cover')}
            onChange={event => { chooseFile(event.target.files?.[0]); event.target.value = ''; }}/>
          <div className="project-cover-buttons"><button type="button" className="projects-page-button" onClick={() => uploadInput.current?.click()}><ImagePlus size={14}/>{activeCover ? text('更换封面', 'Replace cover') : text('上传封面', 'Upload cover')}</button>
            {(file || (!removeCover && savedProject?.cover_url)) && <button type="button" className="projects-page-button" onClick={() => { setFile(null); setRemoveCover(true); setError(''); }}><X size={14}/>{text('移除封面', 'Remove cover')}</button>}
            {removeCover && savedProject?.cover_url && <button type="button" className="projects-page-button" onClick={() => setRemoveCover(false)}>{text('保留原封面', 'Keep current cover')}</button>}</div>
          <small>JPEG / PNG / WebP · ≤ 8 MiB</small>{file && <span className="project-cover-filename" title={file.name}>{file.name}</span>}
        </div></div>
        <label className="project-editor-field"><span>{text('项目名称', 'Project name')}</span><input required type="text" value={name} onChange={event => setName(event.target.value)} data-testid="project-name-input" placeholder={text('支持中文及其他语言', 'Any language supported')}/></label>
        {!project && <div className="project-editor-field"><label htmlFor="project-id-input">{text('项目 ID', 'Project ID')}</label><input id="project-id-input" required maxLength={64} value={id} onChange={event => { setId(event.target.value); setIdTouched(true); setError(''); }} onBlur={() => setIdTouched(true)} disabled={busy || !!savedProject}
          aria-invalid={idTouched && !!idError} aria-describedby="project-id-help" autoComplete="off" spellCheck={false} placeholder="my_project_01" data-testid="project-id-input"/>
          <small id="project-id-help">{text('仅 A–Z、a–z、0–9 和下划线，创建后不可修改。', 'Only A–Z, a–z, 0–9 and underscores; permanent after creation.')}</small>{idTouched && idError && <span role="alert" className="project-editor-error">{idError}</span>}
          <code className="project-folder-preview" aria-label={text('项目目录预览', 'Project folder preview')}>studio_data/project/{id && !idError ? id : '<project_id>'}/v1/</code></div>}
        <div className="project-editor-pair"><div className="project-editor-field"><label htmlFor="project-category">{text('项目分类', 'Project category')}</label><StudioSelect id="project-category" aria-label={text('项目分类', 'Project category')} disabled={busy} value={customCategory ? 'custom' : category ? `category:${category}` : ''}
          options={[{ value: '', label: text('未分类', 'Uncategorized') }, ...categoryOptions.map(value => ({ value: `category:${value}`, label: categoryLabel(value, english) })), { value: 'custom', label: text('自定义分类…', 'Custom category…') }]}
          onValueChange={value => { setCustomCategory(value === 'custom'); setCategory(value.startsWith('category:') ? value.slice(9) : ''); }}/>
          {customCategory && <input aria-label={text('自定义分类名称', 'Custom category name')} maxLength={64} required value={category} onChange={event => setCategory(event.target.value)} placeholder={text('例如：产品 LoRA', 'For example: Product LoRA')}/>}</div>
          {!project && <div className="project-editor-field"><label htmlFor="project-family">{text('初始模型类型', 'Initial model family')}</label><StudioSelect id="project-family" disabled={busy || !!savedProject} aria-label={text('初始模型类型', 'Initial model family')} value={family}
            options={familyOptions} onValueChange={setFamily}/>{familiesError && <p role="alert" className="project-editor-error">{text('无法读取可用模型类型。', 'Could not load model families.')}<button type="button" onClick={() => void reloadFamilies()}>{text('重试', 'Retry')}</button></p>}</div>}</div>
        <label className="project-editor-field"><span>{text('备注（可选）', 'Notes (optional)')}</span><textarea rows={2} value={note} onChange={event => setNote(event.target.value)}/></label>
      </fieldset>
      <div className="project-editor-footer"><button type="button" className="projects-page-button" disabled={busy} onClick={onClose}>{partial ? text('关闭', 'Close') : text('取消', 'Cancel')}</button><button type="submit" className="projects-create-button" disabled={busy || !name.trim() || (!project && (!!idError || !familyAvailable)) || (customCategory && !category.trim()) || category.trim().length > 64}>{busy && <Loader2 size={14} className="animate-spin"/>}{busy ? text('保存中…', 'Saving…') : project || partial ? text('保存', 'Save') : text('创建', 'Create')}</button></div>
    </form>
  </Dialog>;
}
