import React from 'react';
import { UNSAFE_DataRouterContext } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ChevronLeft, ChevronRight, Folder, Redo2, RefreshCw, RotateCcw, Save, Search, Undo2 } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import type { DatasetImage, DatasetInfo } from '../../api/types';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import StudioSelect from '../StudioSelect';
import { LazyImage, LoadingNote } from '../Loading';
import { SlidingIndicator } from '../motion';
import ImageEditor, { type ImageEditorHandle, type ImageEditorNavigationTarget } from '../masks/ImageEditor';
import { preloadPaint, invalidatePaintCache, type PaintInfo } from '../masks/paintApi';
import { paintDraftDirty, paintDraftKey, type PaintDraftStore, type PaintEditorState } from '../masks/paintWorkspaceState';
import MaskIcon from '../icons/MaskIcon';
import { useDatasetImageCatalog } from './useDatasetImageCatalog';
import DatasetNavigationGuard from './DatasetNavigationGuard';
import Dialog from '../Dialog';
import './paint-workspace.css';

type EditStatus = 'modified' | 'unmodified' | 'unknown';
type Filter = 'all' | 'unmodified' | 'pending' | 'modified';
const nameOf = (path: string) => path.replace(/\\/g, '/').split('/').filter(Boolean).pop() || path;
const EMPTY: DatasetImage[] = [];
const INITIAL_STATE: PaintEditorState = {dirty:false,canUndo:false,canRedo:false,canRestore:false,busy:false,available:false,message:''};
const PAGE_SIZE = 30;
export interface PaintWorkspaceProps {
  projectId:string; versionId:string; initialDatasetId?:string; readOnly?:boolean; refreshKey?:string;
  tools?:(beforeStart:()=>Promise<boolean>)=>React.ReactNode;
}
export default function PaintWorkspace({projectId,versionId,initialDatasetId='',readOnly=false,refreshKey,tools}:PaintWorkspaceProps) {
  const text=useWorkspaceText();
  const client=useQueryClient();
  const hasDataRouter=!!React.useContext(UNSAFE_DataRouterContext);
  const root=React.useRef<HTMLElement>(null);
  const body=React.useRef<HTMLDivElement>(null);
  const editor=React.useRef<ImageEditorHandle>(null);
  const drafts=React.useRef<PaintDraftStore>(new Map());
  const [,bumpDrafts]=React.useReducer((n:number)=>n+1,0);
  const onDraftsChange=React.useCallback(()=>bumpDrafts(),[]);
  const [state,setState]=React.useState(INITIAL_STATE);
  const [datasetId,setDatasetId]=React.useState(initialDatasetId);
  const [search,setSearch]=React.useState('');
  const [filter,setFilter]=React.useState<Filter>('all');
  const [page,setPage]=React.useState(1);
  const [selected,setSelected]=React.useState<Record<string,string>>({});
  const [leaveOpen,setLeaveOpen]=React.useState(false);
  const leavePending=React.useRef<((leave:boolean)=>void)|null>(null);
  const datasets=useQuery({queryKey:['caption-datasets',projectId,versionId],refetchOnWindowFocus:false,
    queryFn:({signal})=>apiClient.get<DatasetInfo[]>(`/projects/${projectId}/datasets`,{params:{version_id:versionId,include_cache:false},signal,silent:true})});
  const source=datasets.data?.find(item=>item.source.id===datasetId)||datasets.data?.[0];
  const did=source?.source.id;
  const catalog=useDatasetImageCatalog(did,'filename',true);
  const items=catalog.data||EMPTY;
  const image=items.find(item=>item.rel_path===selected[did||''])||items[0];
  const pendingCount=[...drafts.current.values()].filter(paintDraftDirty).length;
  const statusOf=(item:DatasetImage):EditStatus|'pending'=>{
    const draft=did&&drafts.current.get(paintDraftKey(did,item.rel_path));
    if(draft&&paintDraftDirty(draft))return 'pending';
    return (item as DatasetImage&{saved_edit_status?:EditStatus}).saved_edit_status||'unknown';
  };
  const counts={all:items.length,unmodified:0,pending:0,modified:0,unknown:0};
  for(const item of items)counts[statusOf(item)]++;
  const filtered=items.filter(item=>(filter==='all'||statusOf(item)===filter)&&(!search||`${item.rel_path} ${item.caption||''}`.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase())));
  const pages=Math.max(1,Math.ceil(filtered.length/PAGE_SIZE));
  const shownPage=Math.min(page,pages);
  const visible=filtered.slice((shownPage-1)*PAGE_SIZE,shownPage*PAGE_SIZE);
  const hasImage=!!image;
  const imageIndex=image?filtered.findIndex(item=>item.rel_path===image.rel_path):-1;
  const activeDraft=did&&image?drafts.current.get(paintDraftKey(did,image.rel_path)):undefined;
  const activeDirty=state.dirty||!!activeDraft&&paintDraftDirty(activeDraft);
  React.useLayoutEffect(()=>{
    const host=body.current,workspace=root.current;if(!host||!workspace)return;
    let frame=0;
    const measure=()=>{
      frame=0;let top=host.getBoundingClientRect().top+window.scrollY;
      for(let parent=host.parentElement;parent&&parent!==document.body;parent=parent.parentElement)top+=parent.scrollTop;
      workspace.style.setProperty('--paint-workspace-height',`${Math.max(400,Math.floor(window.innerHeight-top-20))}px`);
    };
    const schedule=()=>{if(!frame)frame=requestAnimationFrame(measure);};
    const observer=new ResizeObserver(schedule);observer.observe(workspace);window.addEventListener('resize',schedule);measure();
    return()=>{observer.disconnect();window.removeEventListener('resize',schedule);if(frame)cancelAnimationFrame(frame);};
  },[did,hasImage]);
  const previousRefresh=React.useRef(refreshKey);
  React.useEffect(()=>{
    if(previousRefresh.current===refreshKey)return;
    previousRefresh.current=refreshKey;
    if(did)void client.invalidateQueries({queryKey:['dataset-image-catalog',did]});
  },[refreshKey,did,client]);
  React.useEffect(()=>{
    if(!did||!image||readOnly||!state.available||state.busy)return;
    const controller=new AbortController();
    const at=items.findIndex(item=>item.rel_path===image.rel_path);
    for(const neighbor of [items[at-1],items[at+1]])if(neighbor)void preloadPaint(did,neighbor.hash,neighbor.rel_path,controller.signal).catch(()=>{});
    return()=>controller.abort();
  },[did,image,items,readOnly,refreshKey,state.available,state.busy]);
  React.useEffect(()=>{
    const unload=(event:BeforeUnloadEvent)=>{if(pendingCount||state.busy){event.preventDefault();event.returnValue='';}};
    window.addEventListener('beforeunload',unload);return()=>window.removeEventListener('beforeunload',unload);
  },[pendingCount,state.busy]);
  React.useEffect(()=>()=>{leavePending.current?.(false);},[]);
  const beforeLeave=()=>{
    if(state.busy)return Promise.reject(new Error(''));
    if(!pendingCount)return Promise.resolve();
    return new Promise<void>((resolve,reject)=>{leavePending.current=leave=>leave?resolve():reject(new Error(''));setLeaveOpen(true);});
  };
  const finishLeave=(leave:boolean)=>{
    if(leave){drafts.current.clear();bumpDrafts();}
    setLeaveOpen(false);leavePending.current?.(leave);leavePending.current=null;
  };
  const target=(item:DatasetImage,index:number):ImageEditorNavigationTarget=>({datasetId:did!,imageId:item.hash,relPath:item.rel_path,position:index+1,total:filtered.length});
  const choose=(item:DatasetImage)=>editor.current?.requestImage(target(item,filtered.indexOf(item)));
  const changeSource=(id:string)=>{setDatasetId(id);setState(INITIAL_STATE);setSearch('');setFilter('all');setPage(1);};
  const savedDatasets=React.useRef(new Set<string>());
  const refreshSaved=async()=>{const ids=[...savedDatasets.current];savedDatasets.current.clear();await Promise.all(ids.map(id=>client.invalidateQueries({queryKey:['dataset-image-catalog',id]})));void datasets.refetch();};
  const saved=async(info?:PaintInfo,_signal?:AbortSignal,savedDatasetId=did,batch=false)=>{
    if(!info||!savedDatasetId)return;
    client.setQueriesData<DatasetImage[]>({queryKey:['dataset-image-catalog',savedDatasetId]},previous=>previous?.map(item=>item.rel_path===info.rel_path?{...item,hash:info.image_id,has_mask:info.has_mask,width:info.width,height:info.height}:item));
    void client.invalidateQueries({queryKey:['caption-images',projectId,versionId,savedDatasetId]});
    savedDatasets.current.add(savedDatasetId);if(!batch)await refreshSaved();
  };
  const refresh=()=>{if(did)invalidatePaintCache(did);void datasets.refetch();void catalog.refetch();};
  const gallery=<>
    <div className="paint-library-browser">
      <label className="paint-search"><Search size={14}/><input type="search" aria-label={text('搜索图片','Search images')} placeholder={text('搜索文件名或标签','Search filenames or captions')} value={search} onChange={event=>{setSearch(event.target.value);setPage(1);}}/></label>
      <button type="button" className="ui-btn ui-btn-sm ui-btn-icon" disabled={state.busy||catalog.isFetching} aria-label={text('刷新图片','Refresh images')} title={text('刷新图片','Refresh images')} onClick={refresh}><RefreshCw size={14} className={catalog.isFetching?'animate-spin':undefined}/></button>

    </div>
    <div className="paint-library-filters ui-segmented ui-segmented-sm" role="group" aria-label={text('图片修改状态','Image edit status')}>
      {([['all',text('全部','All')],['unmodified',text('未修改','Unmodified')],['modified',text('已修改','Modified')],['pending',text('待保存','Unsaved')]] as [Filter,string][]).map(([value,label])=><button type="button" key={value} aria-pressed={filter===value} disabled={state.busy} onClick={()=>{setFilter(value);setPage(1);}}>{label}<span>{catalog.isPending?'—':counts[value]}</span></button>)}<SlidingIndicator className="ui-segmented-thumb"/>
    </div>
    <div className="paint-thumbnail-grid" role="region" aria-label={text('图片缩略图','Image thumbnails')} tabIndex={0}>
      {visible.map(item=><button type="button" key={item.rel_path} title={item.rel_path} aria-label={text(`选择图片：${item.rel_path}`,`Select image: ${item.rel_path}`)} aria-pressed={item.rel_path===image?.rel_path} disabled={state.busy&&!state.switching} data-edit-status={statusOf(item)} onClick={()=>choose(item)}><LazyImage src={apiUrl(`/datasets/${did}/images/${item.hash}/thumb?size=256`)} alt={item.rel_path} loading="lazy"/>{item.has_mask&&<span className="paint-mask-mark" role="img" aria-label={text('有遮罩','Has mask')} title={text('有遮罩','Has mask')}><MaskIcon size={12}/></span>}{statusOf(item)==='pending'&&<span className="paint-draft-mark" aria-label={text('待保存','Unsaved')}/>}</button>)}
      {!visible.length&&<p className="paint-library-empty">{text('没有符合条件的图片。','No matching images.')}</p>}
    </div>
    {pages>1&&<div className="paint-library-navigation"><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('上一页','Previous page')} disabled={state.busy||shownPage<=1} onClick={()=>setPage(shownPage-1)}><ChevronLeft size={12}/></button><span>{text('第','Page ')} {shownPage}/{pages}</span><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('下一页','Next page')} disabled={state.busy||shownPage>=pages} onClick={()=>setPage(shownPage+1)}><ChevronRight size={12}/></button></div>}
  </>;
  return <section className="paint-workspace" ref={root} onKeyDown={event=>{if(event.defaultPrevented||!(event.ctrlKey||event.metaKey)||event.target instanceof Element&&event.target.closest('[role="dialog"]'))return;if(event.key.toLowerCase()==='s'){event.preventDefault();if(event.shiftKey)void editor.current?.saveAll();else void editor.current?.saveCurrent();}else if(event.key.toLowerCase()==='z'&&!(event.target instanceof HTMLInputElement)&&!(event.target instanceof HTMLTextAreaElement)){event.preventDefault();if(event.shiftKey)editor.current?.redo();else editor.current?.undo();}}} aria-label={text('涂抹与遮罩','Paint and mask')}>
    {hasDataRouter&&<DatasetNavigationGuard shouldBlock={()=>!!pendingCount||state.busy} beforeLeave={beforeLeave} onError={()=>{}}/>}
    <div className="paint-workspace-toolbar">
      <label className="paint-source"><StudioSelect searchable icon={<Folder size={14}/>} aria-label={text('图片目录','Image folder')} value={did||''} disabled={state.busy||!source} onValueChange={changeSource} options={(datasets.data||[]).map(item=>({value:item.source.id,label:nameOf(item.source.path)}))}/></label>
      {tools?.(()=>editor.current?.prepareForOperation()??Promise.resolve(true))}
      <div className="paint-workspace-actions"><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" disabled={readOnly||state.busy||!state.canUndo} aria-label={text('撤销','Undo')} title={text('撤销','Undo')} onClick={()=>editor.current?.undo()}><Undo2 size={14}/></button><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" disabled={readOnly||state.busy||!state.canRedo} aria-label={text('重做','Redo')} title={text('重做','Redo')} onClick={()=>editor.current?.redo()}><Redo2 size={14}/></button><button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" disabled={readOnly||state.busy||!state.canRestore} aria-label={text('恢复上次保存前','Restore previous save')} title={text('恢复上次保存前','Restore previous save')} onClick={()=>void editor.current?.restore()}><RotateCcw size={14}/></button><button type="button" className="ui-btn ui-btn-quiet ui-btn-sm" disabled={readOnly||state.busy||!activeDirty} onClick={()=>void editor.current?.discard()}>{text('放弃修改','Discard changes')}</button><button type="button" className="ui-btn ui-btn-sm" disabled={readOnly||state.busy||!state.available||!activeDirty} onClick={()=>void editor.current?.saveCurrent()}><Save size={14}/>{text('保存当前图片','Save current image')}</button><button type="button" className={`ui-btn ui-btn-sm${pendingCount?' ui-btn-primary':''}`} disabled={readOnly||state.busy||!pendingCount} onClick={()=>void editor.current?.saveAll()}><Save size={14}/>{text('保存全部','Save all')}{pendingCount?` (${pendingCount})`:''}</button></div>
    </div>
    {(datasets.error||catalog.error)&&<p role="alert" className="paint-workspace-error">{formatApiError(datasets.error||catalog.error)}</p>}
    <div className="paint-workspace-body" ref={body}>
      {did&&image?<ImageEditor key={did} ref={editor} datasetId={did} imageId={image.hash} relPath={image.rel_path} inline readOnly={readOnly} drafts={drafts.current} onDraftsChange={onDraftsChange} onStateChange={setState} refreshKey={refreshKey} gallery={gallery} onClose={()=>{}} onSaved={saved} onBatchSaved={refreshSaved} onReloadList={refresh} navigation={{position:imageIndex+1,total:filtered.length,resolve:async(direction,current)=>{const index=filtered.findIndex(item=>item.rel_path===current.relPath)+direction;const next=filtered[index];return next?target(next,index):null;},onNavigated:next=>{setSelected(previous=>({...previous,[did]:next.relPath}));const index=filtered.findIndex(item=>item.rel_path===next.relPath);if(index>=0)setPage(Math.floor(index/PAGE_SIZE)+1);}}}/>:catalog.isPending||datasets.isPending?<LoadingNote block label={text('读取图片与遮罩…','Loading images and masks…')}/>:<p className="paint-library-empty">{text('请先导入图片。','Import images first.')}</p>}
    </div>
    {leaveOpen&&<Dialog title={text('图片修改尚未保存','Unsaved image changes')} closeDisabled={state.busy} onClose={()=>finishLeave(false)}><p>{text(`有 ${pendingCount} 张图片的修改尚未保存。`,`Edits to ${pendingCount} images are unsaved.`)}</p><div className="workspace-confirm-actions"><button type="button" className="ui-btn" onClick={()=>finishLeave(false)}>{text('继续编辑','Keep editing')}</button><button type="button" className="ui-btn ui-btn-danger" onClick={()=>finishLeave(true)}>{text('放弃并离开','Discard and leave')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={state.busy||readOnly} onClick={()=>void editor.current?.saveAll().then(()=>{if(![...drafts.current.values()].some(paintDraftDirty))finishLeave(true);})}>{text('保存全部并离开','Save all and leave')}</button></div></Dialog>}
  </section>;
}
