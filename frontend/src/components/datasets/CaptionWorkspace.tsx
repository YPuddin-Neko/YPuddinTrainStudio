import React from 'react';
import { UNSAFE_DataRouterContext, useLocation, useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ChevronLeft, ChevronRight, Folder, Loader2, RefreshCw, Save, Search, X } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import type { DatasetImage, DatasetImagesPage, DatasetInfo } from '../../api/types';
import type { components } from '../../api/generated';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { parseTags, serializeTags } from '../../utils/tags';
import StudioSelect from '../StudioSelect';
import Dialog from '../Dialog';
import DatasetNavigationGuard from './DatasetNavigationGuard';
import StructuredCaptionEditor from './StructuredCaptionEditor';
import { getCaptionStructure, captionFieldChanges, type CaptionFieldDraft } from '../../utils/captionStructure';
import './caption-workspace.css';
import { SlidingIndicator } from '../motion';

type CaptionStats = components['schemas']['DatasetCaptionStats'];
interface Draft {
  key: string; caption: string; description: string; baseCaption: string; baseDescription: string;
  fields: CaptionFieldDraft; adding: string; editing: { index: number; value: string } | null;
}
const imageKey = (image: DatasetImage) => `${image.hash}/${image.rel_path}`;
const nameOf = (path: string) => path.replace(/\\/g, '/').split('/').filter(Boolean).pop() || path;
const PAGE_SIZE = 30;

export default function CaptionWorkspace({ projectId, versionId, initialDatasetId = '', readOnly = false, onChanged }: { projectId: string; versionId: string; initialDatasetId?: string; readOnly?: boolean; onChanged?: () => void }) {
  const text = useWorkspaceText();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const location = useLocation();
  const dataRouter = !!React.useContext(UNSAFE_DataRouterContext);
  const [datasetId, setDatasetId] = React.useState(initialDatasetId);
  const [navigation, setNavigation] = React.useState({ scope: '', page: 1, searchDraft: '', search: '', tag: '', status: '', selected: '' });
  const [draft, setDraft] = React.useState<Draft | null>(null);
  const [editorMode, setEditorMode] = React.useState<'tags' | 'text'>('tags');
  const [statsSearch, setStatsSearch] = React.useState('');
  const [statsLimit, setStatsLimit] = React.useState(80);
  const [saving, setSaving] = React.useState(false);
  const [message, setMessage] = React.useState('');
  const [error, setError] = React.useState('');
  const [leavePrompt, setLeavePrompt] = React.useState(false);
  const leavePending = React.useRef<{ promise: Promise<boolean>; resolve: (leave: boolean) => void } | null>(null);
  const mounted = React.useRef(true);
  const workspace = React.useRef<HTMLElement>(null);
  const editor = React.useRef<HTMLDivElement>(null);
  const grid = React.useRef<HTMLDivElement>(null);
  const datasets = useQuery({ queryKey: ['caption-datasets', projectId, versionId], refetchOnWindowFocus: false, refetchOnReconnect: false,
    queryFn: ({ signal }) => apiClient.get<DatasetInfo[]>(`/projects/${projectId}/datasets`, { params: { version_id: versionId, include_cache: false }, signal, silent: true }) });
  const source = datasets.data?.find(item => item.source.id === datasetId) || datasets.data?.[0];
  const scope = JSON.stringify([projectId, versionId, source?.source.id]);
  const currentScope = React.useRef(scope);
  React.useLayoutEffect(() => { currentScope.current = scope; }, [scope]);
  const emptyNavigation = { scope, page: 1, searchDraft: '', search: '', tag: '', status: '', selected: '' };
  const currentNavigation = navigation.scope === scope ? navigation : emptyNavigation;
  const { page, searchDraft, search, tag, status, selected } = currentNavigation;
  const updateNavigation = (patch: Partial<typeof navigation>) => setNavigation(previous => ({ ...(previous.scope === scope ? previous : emptyNavigation), ...patch }));
  const images = useQuery<DatasetImagesPage>({ queryKey: ['caption-workspace-images', scope, page, search, tag, status], enabled: !!source, refetchOnWindowFocus: false, refetchOnReconnect: false,
    placeholderData: (previous, query) => query?.queryKey[1] === scope ? previous : undefined,
    queryFn: ({ signal }) => apiClient.get<DatasetImagesPage>(`/datasets/${source!.source.id}/images`, {
      params: { page, page_size: PAGE_SIZE, q: search || undefined, tag: tag || undefined, caption_status: status || undefined }, signal, silent: true,
    }) });
  const stats = useQuery({ queryKey: ['caption-stats', scope], enabled: !!source, refetchOnWindowFocus: false, refetchOnReconnect: false,
    queryFn: ({ signal }) => apiClient.get<CaptionStats>(`/datasets/${source!.source.id}/caption-stats`, { signal, silent: true }) });
  const items = images.data?.items || [];
  const selectedIndex = Math.max(0, items.findIndex(item => imageKey(item) === selected));
  const image = items[selectedIndex];
  const pages = Math.max(1, Math.ceil((images.data?.total || 0) / PAGE_SIZE));
  const key = image ? `${scope}/${imageKey(image)}` : '';
  const base: Draft = { key, caption: image?.caption_tags ?? image?.caption ?? '', description: image?.caption_description || '', baseCaption: image?.caption_tags ?? image?.caption ?? '', baseDescription: image?.caption_description || '', fields: {}, adding: '', editing: null };
  const activeDraft = draft?.key === key ? draft : base;
  const updateDraft = (patch: Partial<Draft>) => { setDraft({ ...activeDraft, ...patch }); setMessage(''); };
  const isJson = image?.caption_format?.toLowerCase().replace(/^\./, '') === 'json';
  const structure = getCaptionStructure(image);
  const fieldChanges = captionFieldChanges(structure, activeDraft.fields);
  const invalid = !!image?.caption_error || image?.caption_status === 'invalid';
  const dirty = !readOnly && !!image && (fieldChanges.length > 0 || activeDraft.caption !== activeDraft.baseCaption || activeDraft.description !== activeDraft.baseDescription || !!activeDraft.adding.trim() || !!activeDraft.editing && activeDraft.editing.value !== parseTags(activeDraft.caption)[activeDraft.editing.index]);
  const tags = parseTags(activeDraft.caption);
  const locked = readOnly || saving || invalid || images.isFetching || isJson && !structure?.editable;
  const statsTags = (stats.data?.tags || []).filter(item => item.tag.toLocaleLowerCase().includes(statsSearch.trim().toLocaleLowerCase()));

  React.useLayoutEffect(() => {
    const root = workspace.current, panel = editor.current;
    if (!root || !panel) return;
    let frame = 0;
    const update = () => {
      frame = 0;
      let top = panel.getBoundingClientRect().top + window.scrollY;
      // Use the unscrolled position so scrolling cannot resize the editor in a loop.
      for (let ancestor = panel.parentElement; ancestor && ancestor !== document.body; ancestor = ancestor.parentElement) top += ancestor.scrollTop;
      const height = `${Math.max(280, Math.floor(window.innerHeight - top - 16))}px`;
      if (root.style.getPropertyValue('--caption-editor-height') !== height) root.style.setProperty('--caption-editor-height', height);
    };
    const schedule = () => { if (!frame) frame = window.requestAnimationFrame(update); };
    update();
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(schedule);
    for (let ancestor: HTMLElement | null = root; ancestor && ancestor !== document.body; ancestor = ancestor.parentElement) observer?.observe(ancestor);
    window.addEventListener('resize', schedule);
    return () => {
      observer?.disconnect(); window.removeEventListener('resize', schedule);
      if (frame) window.cancelAnimationFrame(frame);
      root.style.removeProperty('--caption-editor-height');
    };
  }, [key]);

  React.useEffect(() => { mounted.current = true; return () => { mounted.current = false; leavePending.current?.resolve(false); leavePending.current = null; }; }, []);
  React.useEffect(() => {
    if (!images.data || images.isFetching || images.isError || page <= pages) return;
    setNavigation(previous => previous.scope === scope ? { ...previous, page: pages, selected: '' } : previous);
  }, [images.data, images.isFetching, images.isError, page, pages, scope]);
  const settledNavigation = React.useRef('');
  React.useLayoutEffect(() => {
    if (images.isFetching) return;
    const next = JSON.stringify([scope, page, search, tag, status]);
    if (settledNavigation.current !== next && grid.current) grid.current.scrollTop = 0;
    settledNavigation.current = next;
  }, [scope, page, search, tag, status, images.isFetching]);
  React.useLayoutEffect(() => {
    const container = grid.current;
    if (images.isFetching) return;
    const card = container?.querySelector<HTMLElement>('[aria-pressed="true"]');
    if (!container || !card) return;
    const viewport = container.getBoundingClientRect(), bounds = card.getBoundingClientRect();
    if (bounds.top < viewport.top) container.scrollTop += bounds.top - viewport.top;
    else if (bounds.bottom > viewport.bottom) container.scrollTop += bounds.bottom - viewport.bottom;
  }, [key, images.isFetching]);

  const confirmLeave = () => {
    if (saving) return Promise.resolve(false);
    if (!dirty) return Promise.resolve(true);
    if (leavePending.current) return leavePending.current.promise;
    let resolve: (leave: boolean) => void = () => {};
    const promise = new Promise<boolean>(done => { resolve = done; });
    leavePending.current = { promise, resolve }; setLeavePrompt(true);
    return promise;
  };
  const finishLeave = (leave: boolean) => {
    if (leave) setDraft(null);
    setLeavePrompt(false); leavePending.current?.resolve(leave); leavePending.current = null;
  };
  const perform = (action: () => void) => { void confirmLeave().then(leave => { if (leave && mounted.current) { setDraft(null); setMessage(''); setError(''); action(); } }); };
  const beforeLeave = async () => { if (!await confirmLeave()) throw new Error(''); };
  const latestLeave = React.useRef({ dirty, saving, beforeLeave });
  React.useLayoutEffect(() => { latestLeave.current = { dirty, saving, beforeLeave }; });
  React.useEffect(() => {
    const unload = (event: BeforeUnloadEvent) => { if (latestLeave.current.dirty || latestLeave.current.saving) { event.preventDefault(); event.returnValue = ''; } };
    const click = (event: MouseEvent) => {
      if (dataRouter || !latestLeave.current.dirty && !latestLeave.current.saving || event.defaultPrevented || event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      const anchor = event.target instanceof Element ? event.target.closest<HTMLAnchorElement>('a[href]') : null;
      if (!anchor || anchor.hasAttribute('download') || anchor.target && anchor.target !== '_self') return;
      const next = new URL(anchor.href, window.location.href);
      if (next.origin !== window.location.origin) return;
      event.preventDefault(); event.stopPropagation();
      void latestLeave.current.beforeLeave().then(() => {
        const settings = next.pathname === '/settings' || next.pathname.startsWith('/settings/');
        navigate(`${next.pathname}${next.search}${next.hash}`, settings ? { state: { backgroundLocation: location } } : undefined);
      }).catch(() => {});
    };
    window.addEventListener('beforeunload', unload); document.addEventListener('click', click, true);
    return () => { window.removeEventListener('beforeunload', unload); document.removeEventListener('click', click, true); };
  }, [dataRouter, navigate, location]);

  const save = async () => {
    if (!source || !image || locked) return false;
    let caption = activeDraft.caption;
    if (activeDraft.editing || activeDraft.adding.trim()) {
      const next = [...tags];
      if (activeDraft.editing) next[activeDraft.editing.index] = activeDraft.editing.value;
      caption = serializeTags([...next, ...parseTags(activeDraft.adding)]);
    }
    const description = activeDraft.description;
    setSaving(true); setError(''); setMessage('');
    try {
      await apiClient.put(`/datasets/${source.source.id}/images/${image.hash}/caption`, {
        ...(isJson ? { caption_fields: fieldChanges, caption_revision: structure!.revision } : { caption }),
      }, { params: { rel_path: image.rel_path }, silent: true });
      if (!mounted.current || currentScope.current !== scope) return false;
      setDraft({ key, caption, description, baseCaption: caption, baseDescription: description, fields: {}, adding: '', editing: null });
      setMessage(text('标签已保存', 'Caption saved'));
      await Promise.all([images.refetch(), stats.refetch(), queryClient.invalidateQueries({ queryKey: ['caption-images', projectId, versionId, source.source.id] })]);
      if (mounted.current && currentScope.current === scope) { setDraft(null); onChanged?.(); }
      return true;
    } catch (failure) { if (mounted.current && currentScope.current === scope) setError(formatApiError(failure)); return false; }
    finally { if (mounted.current && currentScope.current === scope) setSaving(false); }
  };
  const addTags = () => { if (activeDraft.adding.trim()) updateDraft({ caption: serializeTags([...tags, ...parseTags(activeDraft.adding)]), adding: '' }); };
  const applyTagEdit = () => { if (activeDraft.editing) { const next = [...tags]; next[activeDraft.editing.index] = activeDraft.editing.value; updateDraft({ caption: serializeTags(next), editing: null }); } };
  const refresh = () => perform(() => { void datasets.refetch(); if (source) { void images.refetch(); void stats.refetch(); } });
  const setFilter = (patch: Partial<typeof navigation>) => perform(() => updateNavigation({ ...patch, page: 1, selected: '' }));

  return <section ref={workspace} className="caption-workspace" aria-label={text('标签工作区', 'Caption workspace')}>
    {dataRouter && <DatasetNavigationGuard shouldBlock={() => dirty || saving} beforeLeave={beforeLeave} onError={failure => { if (failure instanceof Error && failure.message) setError(formatApiError(failure)); }}/>}
    <div className="caption-workspace-toolbar" role="group" aria-label={text('选择图片目录', 'Choose image folder')}>
      <label className="caption-workspace-source"><span><Folder size={17}/>{text('图片目录', 'Image folder')}</span><StudioSelect searchable aria-label={text('图片目录', 'Image folder')} value={source?.source.id || ''} disabled={saving || !source} options={(datasets.data || []).map(item => ({ value: item.source.id, label: nameOf(item.source.path) }))} onValueChange={value => perform(() => setDatasetId(value))}/></label>
      <span className="caption-workspace-source-summary">{source?.source.is_reg ? text('正则图', 'Regularization') : text('训练集', 'Training')} · {stats.data?.images ?? source?.stats?.images ?? '—'} {text('张图片', 'images')}</span>
      <button type="button" className="ui-btn" onClick={refresh} disabled={saving || datasets.isFetching || images.isFetching || stats.isFetching} aria-label={text('刷新标签与统计', 'Refresh captions and statistics')}><RefreshCw size={16}/><span>{text('刷新', 'Refresh')}</span></button>
    </div>
    <div className="caption-workspace-filter-panel" role="search" aria-label={text('筛选当前目录图片', 'Filter images in this folder')}>
      <form onSubmit={event => { event.preventDefault(); setFilter({ search: searchDraft.trim() }); }}><span>{text('图片筛选', 'Image filter')}</span><label className="caption-workspace-search"><Search size={16}/><input aria-label={text('搜索文件名或标签', 'Search filenames or captions')} placeholder={text('搜索文件名或标签', 'Search filenames or captions')} value={searchDraft} onChange={event => updateNavigation({ searchDraft: event.target.value })}/></label><button type="submit" className="ui-btn" disabled={saving}>{text('搜索', 'Search')}</button></form>
    <div className="caption-workspace-filters"><div className="ui-segmented" role="group" aria-label={text('标签状态筛选', 'Caption status filter')}>
      {([['', text('全部图片', 'All images'), stats.data?.images], ['captioned', text('已有标签', 'Captioned'), stats.data?.captioned], ['missing', text('缺少标签', 'Missing captions'), stats.data?.missing], ['invalid', text('标签错误', 'Invalid captions'), stats.data?.invalid]] as const).map(([value, label, count]) => <button type="button" key={value} aria-pressed={status === value} disabled={saving} onClick={() => setFilter({ status: value })}>{label}<strong>{count ?? '—'}</strong></button>)}<SlidingIndicator className="ui-segmented-thumb"/></div>
      {(tag || search) && <button type="button" className="ui-btn ui-btn-quiet ui-btn-sm caption-workspace-clear" disabled={saving} onClick={() => setFilter({ tag: '', search: '', searchDraft: '' })}>{tag ? text(`标签：${tag}`, `Tag: ${tag}`) : text(`搜索：${search}`, `Search: ${search}`)}<X size={12}/></button>}
    </div>
    </div>
    {(datasets.error || images.error || error) && <p role="alert" className="caption-workspace-error">{error || formatApiError(datasets.error || images.error)}</p>}
    {datasets.isPending || source && images.isPending ? <p className="caption-workspace-empty" role="status"><Loader2 size={16} className="animate-spin"/>{text('读取图片与标签…', 'Loading images and captions…')}</p>
      : !source ? <p className="caption-workspace-empty">{text('请先导入图片。', 'Import images first.')}</p>
        : <>
          <div className="caption-workspace-pagination"><span className="caption-workspace-filter-count">{text(`符合条件 ${images.data?.total ?? 0} 张`, `${images.data?.total ?? 0} matching images`)}{images.isFetching && <span role="status" className="caption-workspace-refresh-status"><Loader2 size={12} className="animate-spin"/>{text('正在更新筛选结果…', 'Updating filtered images…')}</span>}</span><div><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('上一页', 'Previous page')} disabled={page <= 1 || images.isFetching || saving} onClick={() => perform(() => updateNavigation({ page: page - 1, selected: '' }))}><ChevronLeft size={14}/></button><span>{page} / {pages}</span><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('下一页', 'Next page')} disabled={page >= pages || images.isFetching || saving} onClick={() => perform(() => updateNavigation({ page: page + 1, selected: '' }))}><ChevronRight size={14}/></button></div></div>
          <div className="caption-workspace-content">
          {!items.length ? <p className="caption-workspace-empty">{source.index_status === 'indexing' ? text('图片正在建立索引，完成后刷新即可查看。', 'Images are being indexed. Refresh when indexing completes.') : text('没有符合条件的图片。', 'No images match these filters.')}</p> : <div className="caption-workspace-body">
            <div className="caption-workspace-visual">
              {image && <>
              <div className="caption-workspace-preview"><img src={apiUrl(`/datasets/${source.source.id}/images/${image.hash}/file`)} alt={text(`大图：${image.rel_path}`, `Full image: ${image.rel_path}`)}/></div>
              <div className="caption-workspace-image-heading"><strong>{image.rel_path}</strong><span>{image.width} × {image.height}</span><span>{image.caption_format?.toUpperCase() || 'TXT'}</span><div><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('上一张', 'Previous image')} disabled={selectedIndex <= 0 || saving} onClick={() => perform(() => updateNavigation({ selected: imageKey(items[selectedIndex - 1]) }))}><ChevronLeft size={14}/></button><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('下一张', 'Next image')} disabled={selectedIndex >= items.length - 1 || saving} onClick={() => perform(() => updateNavigation({ selected: imageKey(items[selectedIndex + 1]) }))}><ChevronRight size={14}/></button></div></div>
              </>}
            <div className="caption-workspace-gallery" ref={grid} role="region" aria-busy={images.isFetching} aria-label={text('图片缩略图', 'Image thumbnails')} tabIndex={0}>{items.map(item => <button type="button" key={imageKey(item)} aria-label={text(`选择图片：${item.rel_path}`, `Select image: ${item.rel_path}`)} aria-pressed={item === image} disabled={saving || images.isFetching} onClick={() => perform(() => updateNavigation({ selected: imageKey(item) }))}><img src={apiUrl(`/datasets/${source.source.id}/images/${item.hash}/thumb?size=256`)} alt={item.rel_path} loading="lazy"/><span title={item.rel_path}>{item.rel_path}</span>{(item.caption_error || item.caption_status === 'invalid') ? <small className="caption-workspace-error">{text('标签错误', 'Invalid caption')}</small> : !item.caption && <small>{text('缺少标签', 'No caption')}</small>}</button>)}</div>
            </div>
            {image && <div ref={editor} className="caption-workspace-editor" aria-label={text('当前图片标签编辑器', 'Current image caption editor')} role="region">
              <div className="caption-workspace-editor-fields">
              {invalid && <p role="alert" className="caption-workspace-error">{text('此标签文件无法解析，修复格式后刷新再编辑。', 'This caption file cannot be parsed. Fix its format, then refresh to edit.')}{image.caption_error && ` ${image.caption_error}`}</p>}
              {isJson ? <StructuredCaptionEditor key={key} structure={structure} draft={activeDraft.fields} onChange={fields => updateDraft({ fields })} disabled={locked} readOnly={readOnly}/> : <>
              <div className="caption-workspace-editor-tabs ui-segmented ui-segmented-sm" role="group" aria-label={text('标签编辑方式', 'Caption editing mode')}><button type="button" aria-pressed={editorMode === 'tags'} onClick={() => setEditorMode('tags')}>{text('标签', 'Tags')}</button><button type="button" aria-pressed={editorMode === 'text'} onClick={() => setEditorMode('text')}>{text('文本 / 自然语言', 'Text / natural language')}</button><SlidingIndicator className="ui-segmented-thumb"/></div>
              <fieldset disabled={locked}>
                {editorMode === 'text' ? <label className="caption-workspace-field">{text('标签文本', 'Caption text')}<textarea aria-label={text('标签文本', 'Caption text')} value={activeDraft.caption} onChange={event => updateDraft({ caption: event.target.value, editing: null })} rows={4} readOnly={readOnly}/></label>
                  : <div className="caption-workspace-tags"><div>{tags.map((item, index) => <span className="caption-workspace-chip" key={`${item}/${index}`}>{activeDraft.editing?.index === index ? <input autoFocus aria-label={text(`编辑标签：${item}`, `Edit tag: ${item}`)} value={activeDraft.editing.value} onChange={event => updateDraft({ editing: { index, value: event.target.value } })} onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); applyTagEdit(); } if (event.key === 'Escape') updateDraft({ editing: null }); }} onBlur={applyTagEdit}/> : <button type="button" aria-label={text(`编辑标签：${item}`, `Edit tag: ${item}`)} onClick={() => updateDraft({ editing: { index, value: item } })}>{item}</button>}{!readOnly && <button type="button" aria-label={text(`删除标签：${item}`, `Remove tag: ${item}`)} onClick={() => updateDraft({ caption: serializeTags(tags.filter((_, i) => i !== index)), editing: null })}><X size={12}/></button>}</span>)}</div>{!readOnly && <div className="caption-workspace-tag-add"><input aria-label={text('添加标签', 'Add tags')} placeholder={text('输入标签，多个用逗号分隔', 'Enter tags separated by commas')} value={activeDraft.adding} onChange={event => updateDraft({ adding: event.target.value })} onKeyDown={event => { if (event.key === 'Enter') { event.preventDefault(); addTags(); } }}/><button type="button" className="ui-btn ui-btn-sm" onClick={addTags}>{text('添加', 'Add')}</button></div>}</div>}
              </fieldset>
              </>}
              </div>
              <div className="caption-workspace-save"><span role="status">{saving ? text('正在保存…', 'Saving…') : dirty ? text('有未保存的修改', 'Unsaved changes') : message || (readOnly ? text('只读', 'Read only') : '')}</span>{!readOnly && <><button type="button" className="ui-btn" disabled={!dirty || saving} onClick={() => perform(() => setDraft(null))}>{text('放弃修改', 'Discard changes')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={!dirty || locked} onClick={() => { void save(); }}><Save size={14}/>{text('保存标签', 'Save caption')}</button></>}</div>
            </div>}
          </div>}
          <aside className="caption-workspace-statistics" aria-label={text('整个数据集的标签统计', 'Statistics for the whole dataset')}>
            <header><div><h3>{text('标签频次', 'Tag frequency')}</h3><span>{text(`整个目录 ${stats.data?.images ?? '—'} 张图片 · ${stats.data?.unique_tags ?? '—'} 个不同标签`, `All ${stats.data?.images ?? '—'} images in this folder · ${stats.data?.unique_tags ?? '—'} unique tags`)}</span></div><input aria-label={text('搜索标签统计', 'Search tag statistics')} placeholder={text('筛选标签', 'Filter tags')} value={statsSearch} onChange={event => { setStatsSearch(event.target.value); setStatsLimit(80); }}/></header>
            {stats.error ? <p role="alert" className="caption-workspace-error">{formatApiError(stats.error)}</p> : stats.isPending ? <p role="status">{text('统计标签…', 'Loading tag statistics…')}</p> : <><div className="caption-workspace-frequencies">{statsTags.slice(0, statsLimit).map(item => <button type="button" key={item.tag} aria-label={text(`筛选标签：${item.tag}，${item.count} 张图片`, `Filter tag: ${item.tag}, ${item.count} images`)} aria-pressed={tag === item.tag} disabled={saving} onClick={() => setFilter({ tag: tag === item.tag ? '' : item.tag })}><span>{item.tag}</span><strong>{item.count}</strong></button>)}{!statsTags.length && <p>{text('没有符合条件的标签。', 'No matching tags.')}</p>}</div>{statsTags.length > statsLimit && <button type="button" className="ui-btn ui-btn-quiet ui-btn-sm" onClick={() => setStatsLimit(limit => limit + 80)}>{text(`显示更多（共 ${statsTags.length} 个）`, `Show more (${statsTags.length} total)`)}</button>}</>}
          </aside>
          </div>
        </>}
    {leavePrompt && <Dialog title={text('标签修改尚未保存', 'Unsaved caption changes')} onClose={() => finishLeave(false)} closeDisabled={saving}><div className="caption-workspace-leave"><p>{text('保存后继续，或放弃这张图片的修改。', 'Save before continuing, or discard changes to this image.')}</p>{error && <p role="alert" className="caption-workspace-error">{error}</p>}<div><button type="button" className="ui-btn" disabled={saving} onClick={() => finishLeave(false)}>{text('继续编辑', 'Keep editing')}</button><button type="button" className="ui-btn ui-btn-danger" disabled={saving} onClick={() => finishLeave(true)}>{text('放弃并继续', 'Discard and continue')}</button><button type="button" disabled={locked} className="ui-btn ui-btn-primary" onClick={() => { void save().then(saved => { if (saved && mounted.current) finishLeave(true); }); }}>{saving ? text('正在保存…', 'Saving…') : text('保存并继续', 'Save and continue')}</button></div></div></Dialog>}
  </section>;
}
