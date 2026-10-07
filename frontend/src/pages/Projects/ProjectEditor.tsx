import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, AudioLines, Crop, FolderOpen, Image as ImageIcon, ImagePlus, Loader2, X } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { FamilyInfo } from '../../api/types';
import Dialog from '../../components/Dialog';
import { LazyImage } from '../../components/Loading';
import StudioSelect from '../../components/StudioSelect';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import { trainingFamilyOptions } from '../../utils/trainingFamilies';
import { useTranslation } from 'react-i18next';
import { categoryLabel, coverSource, PROJECT_CATEGORIES, type GalleryProject } from './projectGallery';
import ProjectCoverCropper from './ProjectCoverCropper';
import { COVER_MAX_BYTES, cropImageStyle, type CoverCrop } from './coverCrop';
import './projects.css';

export function ProjectCover({ source, name, crop }: { source?: string | null; name: string; crop?: CoverCrop }) {
  const text = useWorkspaceText();
  const placeholder = (label: string) => <span className="project-cover-placeholder"><FolderOpen size={27} aria-hidden="true"/><span>{label}</span></span>;
  return source ? <LazyImage src={source} alt={text(`${name} 的封面`, `Cover for ${name}`)} style={crop ? cropImageStyle(crop) : undefined} loading="lazy"
    fallback={placeholder(text('封面暂不可用', 'Cover unavailable'))}/>
    : placeholder(text('未设置封面', 'No cover'));
}

export default function ProjectEditor({ project, categories, onClose, onSaved, onPartial }: {
  project?: GalleryProject; categories: string[]; onClose: () => void; onSaved: (project: GalleryProject) => void; onPartial: (project: GalleryProject) => void;
}) {
  const text = useWorkspaceText();
  const { i18n } = useTranslation();
  const english = i18n.resolvedLanguage?.startsWith('en') || false;
  const [projectType, setProjectType] = React.useState<'image' | 'tts'>(project?.project_type === 'tts' ? 'tts' : 'image');
  const [choosingType, setChoosingType] = React.useState(!project);
  const stepFocus = React.useRef(false);
  const nameInput = React.useRef<HTMLInputElement>(null);
  const typeButtons = React.useRef<Partial<Record<'image' | 'tts', HTMLButtonElement | null>>>({});
  const typeId = React.useId();
  const { data: families = [], isError: familiesError, refetch: reloadFamilies } = useQuery({
    queryKey: ['families'],
    queryFn: () => apiClient.get<FamilyInfo[]>('/families', { silent: true }),
    staleTime: 5 * 60 * 1000,
    enabled: !project && !choosingType && projectType === 'image',
  });
  const [name, setName] = React.useState(project?.name || '');
  const [id, setId] = React.useState(project?.id || '');
  const [note, setNote] = React.useState(project?.note || '');
  const [category, setCategory] = React.useState(project?.category || '');
  const [customCategory, setCustomCategory] = React.useState(false);
  const [family, setFamily] = React.useState('anima');
  const familyOptions = trainingFamilyOptions(families, english);
  const familyAvailable = familyOptions.some(option => option.value === family && !option.disabled);
  const [file, setFile] = React.useState<File | null>(null);
  const [crop, setCrop] = React.useState<CoverCrop>();
  const [candidate, setCandidate] = React.useState<{ file: File; crop?: CoverCrop } | null>(null);
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
  React.useLayoutEffect(() => {
    if (!stepFocus.current) return;
    stepFocus.current = false;
    (choosingType ? typeButtons.current[projectType] : nameInput.current)?.focus();
  }, [choosingType, projectType]);
  const chooseType = (type: 'image' | 'tts') => {
    if (busyRef.current || savedProject) return;
    stepFocus.current = true;
    setProjectType(type); setChoosingType(false); setError('');
  };
  React.useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const idError = !id ? text('请填写项目 ID。', 'Enter a project ID.') : id.length > 64 ? text('项目 ID 最多 64 个字符。', 'Project ID must be at most 64 characters.') : !/^[A-Za-z0-9_]+$/.test(id) ? text('项目 ID 只能包含英文字母、数字和下划线。', 'Use only ASCII letters, digits and underscores in the project ID.') : '';
  const creationBlocked = !savedProject && (!!idError || (projectType === 'image' && !familyAvailable));
  const categoryOptions = [...new Set([...PROJECT_CATEGORIES, ...categories, ...(project?.category ? [project.category] : [])])];
  const categoryChoices = [{ value: '', label: text('未分类', 'Uncategorized') }, ...categoryOptions.map(value => ({ value: `category:${value}`, label: categoryLabel(value, english) })), { value: 'custom', label: text('自定义分类…', 'Custom category…') }];
  const categoryBox = React.useRef<HTMLDivElement>(null);
  const focusCategory = React.useRef(false);
  // A custom category is typed in the same box, so focus follows the box when it changes form.
  React.useLayoutEffect(() => {
    if (!focusCategory.current) return;
    focusCategory.current = false;
    document.getElementById('project-category')?.focus();
  }, [customCategory]);
  const chooseCategory = (value: string) => {
    const custom = value === 'custom';
    if (custom !== customCategory) focusCategory.current = true;
    if (custom && customCategory) return;
    setCustomCategory(custom);
    setCategory(value.startsWith('category:') ? value.slice(9) : '');
  };
  const activeCover = preview || (!removeCover && savedProject?.cover_url ? coverSource(savedProject.cover_url) : null);
  const chooseFile = (selected?: File) => {
    if (!selected) return;
    if (selected.size > COVER_MAX_BYTES) { setError(text('封面不能超过 8 MB（8,388,608 字节）。', 'Cover must be no larger than 8 MB (8,388,608 bytes).')); return; }
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(selected.type)) { setError(text('请选择 JPEG、PNG 或 WebP 图片。', 'Choose a JPEG, PNG or WebP image.')); return; }
    setCandidate({ file: selected }); setError('');
  };
  const adjustCover = async () => {
    if (file) { setCandidate({ file, crop }); return; }
    if (!activeCover || busyRef.current) return;
    busyRef.current = true; setBusy(true); setError('');
    try {
      const response = await fetch(activeCover);
      if (!response.ok) throw new Error(text('无法读取封面，请重试。', 'Could not load the cover. Please retry.'));
      const blob = await response.blob();
      chooseFile(new File([blob], 'project-cover.webp', { type: blob.type }));
    } catch (failure) { setError(formatApiError(failure)); }
    finally { busyRef.current = false; setBusy(false); }
  };
  const save = async () => {
    setIdTouched(true);
    if (choosingType || busyRef.current || !name.trim() || creationBlocked || category.trim().length > 64 || (customCategory && !category.trim())) return;
    busyRef.current = true; setBusy(true); setError('');
    let currentProject = savedProject;
    const metadata = { name: name.trim(), note: note.trim(), category: category.trim() || null };
    const signature = JSON.stringify(metadata);
    try {
      if (!currentProject) {
        const model = projectType === 'tts' ? { project_type: 'tts' as const, engine: 'voxcpm1.5' as const } : { family };
        currentProject = await apiClient.post<GalleryProject>('/projects', { id, ...metadata, ...model }, { silent: true });
        setSavedProject(currentProject); savedMetadata.current = signature;
      } else if (savedMetadata.current !== signature) {
        currentProject = await apiClient.patch<GalleryProject>(`/projects/${currentProject.id}`, metadata, { silent: true });
        setSavedProject(currentProject); savedMetadata.current = signature;
      }
      if (file) {
        const body = new FormData(); body.append('file', file);
        if (crop) body.append('crop', JSON.stringify(crop));
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
  if (candidate) return <Dialog key="crop" title={text('裁切项目封面', 'Crop project cover')} onClose={() => setCandidate(null)}>
    <ProjectCoverCropper file={candidate.file} initialCrop={candidate.crop} onCancel={() => setCandidate(null)} onApply={selectedCrop => {
      setFile(candidate.file); setCrop(selectedCrop); setRemoveCover(false); setCandidate(null); setError('');
    }}/>
  </Dialog>;
  return <Dialog key="editor" title={project ? text('编辑项目', 'Edit project') : choosingType ? text('新建项目', 'New project') : projectType === 'tts' ? text('新建语音项目', 'New speech project') : text('新建图像项目', 'New image project')} onClose={onClose} closeDisabled={busy}>
    {choosingType ? <div className="project-type-choices" role="group" aria-label={text('项目类型', 'Project type')}>
      {([
        { type: 'image', Icon: ImageIcon, label: text('图像', 'Image'), description: text('使用图片训练图像生成模型', 'Train image generation models with images') },
        { type: 'tts', Icon: AudioLines, label: text('语音', 'Speech'), description: text('使用录音与转写训练语音模型', 'Train speech models with recordings and transcripts') },
      ] as const).map(({ type, Icon, label, description }) => <button key={type} ref={element => { typeButtons.current[type] = element; }} type="button" className="project-type-choice"
        aria-labelledby={`${typeId}-${type}`} aria-describedby={`${typeId}-${type}-description`} onClick={() => chooseType(type)}>
        <Icon size={34} strokeWidth={1.6} aria-hidden="true"/>
        <strong id={`${typeId}-${type}`}>{label}</strong>
        <span id={`${typeId}-${type}-description`}>{description}</span>
      </button>)}
    </div> : <form className="project-editor" onSubmit={event => { event.preventDefault(); void save(); }} data-testid={project ? 'edit-project-modal' : 'create-project-modal'} aria-busy={busy}>
      {error && <p role="alert" className="project-editor-error">{error}</p>}
      {partial && !project && <p role="status" className="project-editor-note">{text(`项目 ${savedProject?.id} 已创建；关闭窗口会保留此项目，重试不会重复创建。`, `Project ${savedProject?.id} exists. Closing keeps it; retrying will not create a duplicate.`)}</p>}
      <fieldset disabled={busy} className="project-editor-fields">
        <div className="project-cover-editor"><div className="project-cover-preview"><ProjectCover source={activeCover} crop={file ? crop : undefined} name={name || text('项目', 'Project')}/></div><div className="project-cover-controls">
          <strong>{text('项目封面', 'Project cover')}</strong>
          <input ref={uploadInput} className="project-cover-file" type="file" tabIndex={-1} accept="image/jpeg,image/png,image/webp" aria-label={text('上传项目封面', 'Upload project cover')}
            onChange={event => { chooseFile(event.target.files?.[0]); event.target.value = ''; }}/>
          <div className="project-cover-buttons"><button type="button" className="ui-btn" onClick={() => uploadInput.current?.click()}><ImagePlus size={14}/>{activeCover ? text('更换封面', 'Replace cover') : text('上传封面', 'Upload cover')}</button>
            {activeCover && <button type="button" className="ui-btn" onClick={() => void adjustCover()}><Crop size={14}/>{text('调整裁切', 'Adjust crop')}</button>}
            {(file || (!removeCover && savedProject?.cover_url)) && <button type="button" className="ui-btn" onClick={() => { setFile(null); setCrop(undefined); setRemoveCover(true); setError(''); }}><X size={14}/>{text('移除封面', 'Remove cover')}</button>}
            {removeCover && savedProject?.cover_url && <button type="button" className="ui-btn" onClick={() => setRemoveCover(false)}>{text('保留原封面', 'Keep current cover')}</button>}</div>
          <small title={text('文件上限 8,388,608 字节（8 MiB）', 'File limit: 8,388,608 bytes (8 MiB)')}>JPEG / PNG / WebP · ≤ 8 MB</small>{file && <span className="project-cover-filename" title={file.name}>{file.name}</span>}
        </div></div>
        <label className="project-editor-field"><span>{text('项目名称', 'Project name')}</span><input ref={nameInput} required type="text" value={name} onChange={event => setName(event.target.value)} data-testid="project-name-input" placeholder={text('支持中文及其他语言', 'Any language supported')}/></label>
        {!project && <div className="project-editor-field"><label htmlFor="project-id-input">{text('项目 ID', 'Project ID')}</label><input id="project-id-input" required maxLength={64} value={id} onChange={event => { setId(event.target.value); setIdTouched(true); setError(''); }} onBlur={() => setIdTouched(true)} disabled={busy || !!savedProject}
          aria-invalid={idTouched && !!idError} aria-describedby="project-id-help" autoComplete="off" spellCheck={false} placeholder="my_project_01" data-testid="project-id-input"/>
          <small id="project-id-help">{text('仅 A–Z、a–z、0–9 和下划线，创建后不可修改。', 'Only A–Z, a–z, 0–9 and underscores; permanent after creation.')}</small>{idTouched && idError && <span role="alert" className="project-editor-error">{idError}</span>}
          <code className="project-folder-preview" aria-label={text('项目目录预览', 'Project folder preview')}>studio_data/project/{id && !idError ? id : '<project_id>'}/v1/</code></div>}
        <div className="project-editor-pair"><div className="project-editor-field"><label htmlFor="project-category">{text('项目分类', 'Project category')}</label>
          {customCategory ? <div ref={categoryBox} className="project-category-custom">
            <input id="project-category" maxLength={64} required autoComplete="off" value={category} onChange={event => setCategory(event.target.value)} placeholder={text('自定义，例如：产品 LoRA', 'Custom, e.g. Product LoRA')}/>
            <StudioSelect className="project-category-toggle" anchorRef={categoryBox} aria-label={text('选择项目分类', 'Choose a project category')} disabled={busy} value="custom" options={categoryChoices} onValueChange={chooseCategory}/>
          </div> : <StudioSelect id="project-category" aria-label={text('项目分类', 'Project category')} disabled={busy} value={category ? `category:${category}` : ''} options={categoryChoices} onValueChange={chooseCategory}/>}</div>
          {!project && (projectType === 'tts'
            ? <div className="project-editor-field"><label htmlFor="project-engine">{text('训练模型', 'Training model')}</label><StudioSelect id="project-engine" disabled aria-label={text('训练模型', 'Training model')} value="voxcpm1.5" options={[{ value: 'voxcpm1.5', label: 'VoxCPM 1.5' }]} onValueChange={() => {}}/></div>
            : <div className="project-editor-field"><label htmlFor="project-family">{text('初始模型类型', 'Initial model family')}</label><StudioSelect id="project-family" disabled={busy || !!savedProject} aria-label={text('初始模型类型', 'Initial model family')} value={family}
              options={familyOptions} onValueChange={setFamily}/>{familiesError && <p role="alert" className="project-editor-error">{text('无法读取可用模型类型。', 'Could not load model families.')}<button type="button" className="ui-link" onClick={() => void reloadFamilies()}>{text('重试', 'Retry')}</button></p>}</div>)}</div>
        <label className="project-editor-field"><span>{text('备注（可选）', 'Notes (optional)')}</span><textarea rows={2} value={note} onChange={event => setNote(event.target.value)}/></label>
      </fieldset>
      <div className="project-editor-footer">{!project && <button type="button" className="ui-btn ui-btn-quiet project-type-back" disabled={busy || !!savedProject} onClick={() => {
        if (busyRef.current || savedProject) return;
        stepFocus.current = true; setChoosingType(true); setError('');
      }}><ArrowLeft size={14} aria-hidden="true"/>{text('选择类型', 'Choose type')}</button>}<button type="button" className="ui-btn" disabled={busy} onClick={onClose}>{partial ? text('关闭', 'Close') : text('取消', 'Cancel')}</button><button type="submit" className="ui-btn ui-btn-primary" disabled={busy || !name.trim() || creationBlocked || (customCategory && !category.trim()) || category.trim().length > 64}>{busy && <Loader2 size={14} className="animate-spin"/>}{busy ? text('保存中…', 'Saving…') : project || partial ? text('保存', 'Save') : text('创建', 'Create')}</button></div>
    </form>}
  </Dialog>;
}
