import { useCallback, useContext, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Link, useNavigate, UNSAFE_DataRouterContext } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { ChevronLeft, ChevronRight, Folder, Image as ImageIcon, Layers, Loader2, RefreshCw, Search } from 'lucide-react';
import { apiClient, apiUrl, READ_TIMEOUT_MS } from '../../api/client';
import type { DatasetInfo, DatasetImagesPage } from '../../api/types';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import StudioSelect from '../StudioSelect';
import { LazyImage } from '../Loading';
import ImageEditor, { type ImageEditorIdentity, type ImageEditorNavigationTarget } from '../masks/ImageEditor';
import type { PaintInfo } from '../masks/paintApi';
import MaskOverlay from './MaskOverlay';
import ImageSortSelect, { type ImageSort } from './ImageSortSelect';
import { useGridPageSize } from '../../api/hooks/useGridPageSize';
import DatasetNavigationGuard from './DatasetNavigationGuard';
import { projectUrl, versionConfigUrl } from '../../utils/projectVersions';
import './caption-viewer.css';

const datasetName = (path: string) => path.replace(/\\/g,'/').split('/').filter(Boolean).pop()?.replace(/^(?:d_[0-9a-f]+-)+/,'') || path;
type LocatedImage = { data:DatasetImagesPage; index:number };
type EditorSession = { scope:string; datasetId:string; search:string; sort:ImageSort; page:number; controller:AbortController; target?:{ image:ImageEditorNavigationTarget; located:LocatedImage; signal:AbortSignal } };

export default function CaptionViewer({projectId,versionId,readOnly=false,editing=false,initialDatasetId=''}: {projectId:string;versionId:string;readOnly?:boolean;editing?:boolean;initialDatasetId?:string}) {
  const text=useWorkspaceText();
  const queryClient=useQueryClient();
  const navigate=useNavigate();
  const hasDataRouter=!!useContext(UNSAFE_DataRouterContext);
  const [editor,setEditor]=useState<ImageEditorNavigationTarget|null>(null);
  const editorSession=useRef<EditorSession|null>(null);
  const [editError,setEditError]=useState('');
  const [showMask,setShowMask]=useState(true);
  const allowedDestination=useRef('');
  const gridRef=useRef<HTMLDivElement | null>(null);
  const gridPage=useGridPageSize(40);
  const [sort,setSort]=useState<ImageSort>('filename');
  const displayedPage=useRef('');
  const [datasetId,setDatasetId]=useState(initialDatasetId);
  const measureGrid = gridPage.gridRef;
  const bindGrid = useCallback((node: HTMLDivElement | null)=>{gridRef.current=node;measureGrid(node);},[measureGrid]);
  const [navigation,setNavigation]=useState({scope:'',pageSize:40,page:1,draftSearch:'',search:'',selected:'',selectionIndex:0});
  const datasets=useQuery({queryKey:['caption-datasets',projectId,versionId],queryFn:({signal})=>apiClient.get<DatasetInfo[]>(`/projects/${projectId}/datasets`,{params:{version_id:versionId,include_cache:false},signal,silent:true})});
  const source=datasets.data?.find(item=>item.source.id===datasetId) || datasets.data?.[0];
  const windowsPaths=/^(?:[a-z]:[\\/]|\\\\)/i.test(source?.source.path||'');
  const sameImagePath=(left:string,right:string)=>windowsPaths?left.replace(/\\/g,'/')===right.replace(/\\/g,'/'):left===right;
  // Derive reset state before querying: a reused viewer must never request the old page in a new source/version.
  const scope=JSON.stringify([projectId,versionId,source?.source.id]);
  const empty={scope,pageSize:gridPage.pageSize,page:1,draftSearch:'',search:'',selected:'',selectionIndex:0};
  const fitNavigation=(previous:typeof navigation)=>{
    const current=previous.scope===scope?previous:empty;
    if(current.pageSize===gridPage.pageSize)return current;
    // Resizing changes page boundaries, not the image being viewed.
    const index=(current.page-1)*current.pageSize+current.selectionIndex;
    return {...current,pageSize:gridPage.pageSize,page:Math.floor(index/gridPage.pageSize)+1,selectionIndex:index%gridPage.pageSize};
  };
  const current=fitNavigation(navigation);
  const {draftSearch,search,page,selected}=current;
  const updateNavigation=(patch:Partial<typeof navigation>)=>setNavigation(previous=>({...fitNavigation(previous),...patch}));
  const setPage=(page:number)=>updateNavigation({page,selected:'',selectionIndex:0});
  const setDraftSearch=(draftSearch:string)=>updateNavigation({draftSearch});
  const setSearch=(search:string)=>updateNavigation({search});
  const setSelected=(selected:string)=>updateNavigation({selected,selectionIndex:Math.max(0,items.findIndex(item=>sameImagePath(item.rel_path,selected)))});
  const changeSort=(value:ImageSort)=>{
    setSort(value);
    setNavigation(previous=>({...previous.scope===scope?previous:empty,pageSize:gridPage.pageSize,page:1,selected:'',selectionIndex:0}));
  };
  // The first request waits for the grid's column count; later page-size changes keep the shown page until the new one arrives.
  const query=useQuery<DatasetImagesPage>({
    queryKey:['caption-images',projectId,versionId,source?.source.id,page,search,sort,gridPage.pageSize],enabled:!!source && gridPage.ready,
    placeholderData:(previous,previousQuery)=>previousQuery?.queryKey[1]===projectId && previousQuery.queryKey[2]===versionId && previousQuery.queryKey[3]===source?.source.id && previousQuery.queryKey[5]===search && previousQuery.queryKey[6]===sort ? previous : undefined,
    queryFn:({signal})=>apiClient.get<DatasetImagesPage>(`/datasets/${source!.source.id}/images`,{params:{page,page_size:gridPage.pageSize,q:search||undefined,sort},signal,silent:true}),
  });
  const items=query.data?.items || [];
  // Position of the viewed image on the page on screen, which is the previous page size's page while a resized page loads.
  const viewed=(current.page-1)*current.pageSize+current.selectionIndex-((query.data?.page??page)-1)*(query.data?.page_size??gridPage.pageSize);
  const selectedIndex=selected?items.findIndex(item=>sameImagePath(item.rel_path,selected)):viewed>=0&&viewed<items.length?viewed:Math.min(current.selectionIndex,Math.max(0,items.length-1));
  const image=items[selectedIndex];
  const pages=Math.max(1,Math.ceil((query.data?.total||0)/gridPage.pageSize));
  const imageKey=image?`${image.hash}/${image.rel_path}`:'';
  const actualPage=query.data?.page;
  useLayoutEffect(()=>{
    const grid=gridRef.current;
    if(!grid || actualPage===undefined || query.isPlaceholderData)return;
    const nextPage=JSON.stringify([scope,search,actualPage]);
    if(displayedPage.current!==nextPage){grid.scrollTop=0;displayedPage.current=nextPage;}
    const selectedCard=grid.querySelector<HTMLButtonElement>('[aria-pressed="true"]');
    if(!selectedCard)return;
    const viewport=grid.getBoundingClientRect(),card=selectedCard.getBoundingClientRect();
    if(card.top<viewport.top)grid.scrollTop=Math.max(0,grid.scrollTop+card.top-viewport.top);
    else if(card.bottom>viewport.bottom)grid.scrollTop+=card.bottom-viewport.bottom;
  },[scope,search,actualPage,query.isPlaceholderData,imageKey,gridPage.columns]);
  useEffect(()=>{
    if(!query.data || query.isFetching || query.isError || page<=pages)return;
    // Refresh can remove the last page after curation. Only correct the scope that returned this result.
    setNavigation(previous=>previous.scope===scope?{...previous,pageSize:gridPage.pageSize,page:pages,selected:'',selectionIndex:0}:previous);
  },[query.data,query.isFetching,query.isError,page,pages,scope,gridPage.pageSize]);
  const loading=datasets.isPending || (!!source && query.isPending);
  const error=datasets.error || query.error;
  const refresh=()=>{void datasets.refetch();if(source)void query.refetch();};
  const editorScope=JSON.stringify([scope,search,sort]);
  const latest=useRef({editorScope,pageSize:gridPage.pageSize});
  latest.current={editorScope,pageSize:gridPage.pageSize};
  useEffect(()=>{
    const session=editorSession.current;
    if(session&&session.scope!==editorScope)session.controller.abort();
  },[editorScope]);
  useEffect(()=>()=>editorSession.current?.controller.abort(),[]);
  const checkSession=(session:EditorSession,signal:AbortSignal)=>{
    if(signal.aborted||session.controller.signal.aborted||editorSession.current!==session||latest.current.editorScope!==session.scope)throw new DOMException('Aborted','AbortError');
  };
  const withSession=async<T,>(session:EditorSession,signal:AbortSignal,run:(signal:AbortSignal)=>Promise<T>)=>{
    const controller=new AbortController();
    const abort=()=>controller.abort();
    signal.addEventListener('abort',abort,{once:true});session.controller.signal.addEventListener('abort',abort,{once:true});
    try{checkSession(session,signal);const result=await run(controller.signal);checkSession(session,signal);return result;}
    finally{signal.removeEventListener('abort',abort);session.controller.signal.removeEventListener('abort',abort);}
  };
  const readPage=(session:EditorSession,page:number,pageSize:number,signal:AbortSignal)=>apiClient.get<DatasetImagesPage>(`/datasets/${session.datasetId}/images`,{params:{page,page_size:pageSize,q:session.search||undefined,sort:session.sort},signal,silent:true,timeout:READ_TIMEOUT_MS});
  const locateImage=async(session:EditorSession,relPath:string,preferredPage:number,pageSize:number,signal:AbortSignal):Promise<LocatedImage>=>{
    // The modified-time order can change after saving, so a page number cannot identify an image.
    const first=await readPage(session,preferredPage,pageSize,signal);checkSession(session,signal);
    const firstIndex=first.items.findIndex(item=>sameImagePath(item.rel_path,relPath));
    if(firstIndex>=0)return {data:first,index:firstIndex};
    const pageCount=Math.ceil(first.total/pageSize);
    for(let p=1;p<=pageCount;p++){
      if(p===preferredPage)continue;
      const data=await readPage(session,p,pageSize,signal);checkSession(session,signal);
      const index=data.items.findIndex(item=>sameImagePath(item.rel_path,relPath));
      if(index>=0)return {data,index};
    }
    throw new Error(text('当前图片已不在筛选结果中，请关闭编辑器并刷新列表。','This image is no longer in the filtered results. Close the editor and refresh the list.'));
  };
  const commitSelection=(session:EditorSession,located:LocatedImage,identity:ImageEditorIdentity)=>{
    checkSession(session,session.controller.signal);
    const data={...located.data,items:located.data.items.map((item,index)=>index===located.index?{...item,hash:identity.imageId}:item)};
    queryClient.setQueryData(['caption-images',projectId,versionId,session.datasetId,data.page,session.search,session.sort,data.page_size],data);
    session.page=data.page;
    updateNavigation({pageSize:data.page_size,page:data.page,selected:data.items[located.index].rel_path,selectionIndex:located.index});
    setEditor({...identity,position:(data.page-1)*data.page_size+located.index+1,total:data.total});
  };
  const resolveImage=async(direction:-1|1,identity:ImageEditorIdentity,signal:AbortSignal):Promise<ImageEditorNavigationTarget|null>=>{
    const session=editorSession.current;
    if(!session||session.datasetId!==identity.datasetId)throw new DOMException('Aborted','AbortError');
    return withSession(session,signal,async activeSignal=>{
      session.target=undefined;
      const size=latest.current.pageSize;
      const current=await locateImage(session,identity.relPath,session.page,size,activeSignal);
      const position=(current.data.page-1)*current.data.page_size+current.index+direction;
      if(position<0||position>=current.data.total){
        session.page=current.data.page;
        queryClient.setQueryData(['caption-images',projectId,versionId,session.datasetId,current.data.page,session.search,session.sort,current.data.page_size],current.data);
        updateNavigation({pageSize:current.data.page_size,page:current.data.page,selected:current.data.items[current.index].rel_path,selectionIndex:current.index});
        setEditor(previous=>previous?{...previous,position:(current.data.page-1)*current.data.page_size+current.index+1,total:current.data.total}:previous);
        return null;
      }
      const page=Math.floor(position/size)+1,index=position%size;
      const data=page===current.data.page?current.data:await readPage(session,page,size,activeSignal);
      checkSession(session,activeSignal);
      const item=data.items[index];
      if(!item||sameImagePath(item.rel_path,identity.relPath))throw new Error(text('图片列表已变化，请重试。','The image list changed. Try again.'));
      const target={datasetId:session.datasetId,imageId:item.hash,relPath:item.rel_path,position:position+1,total:data.total};
      session.target={image:target,located:{data,index},signal};
      return target;
    });
  };
  const savedImage=async(info?:PaintInfo,signal?:AbortSignal)=>{
    const session=editorSession.current;
    if(!session||!info)throw new Error(text('未取得已保存图片的信息，请重新读取。','The saved image information is unavailable. Reload the image.'));
    await withSession(session,signal||session.controller.signal,async signal=>{
      setEditor(previous=>previous&&sameImagePath(previous.relPath,info.rel_path)?{...previous,imageId:info.image_id}:previous);
      await queryClient.cancelQueries({queryKey:['caption-images',projectId,versionId,session.datasetId]});
      checkSession(session,signal);
      queryClient.setQueriesData<DatasetImagesPage>({queryKey:['caption-images',projectId,versionId,session.datasetId]},previous=>previous?{...previous,items:previous.items.map(item=>sameImagePath(item.rel_path,info.rel_path)?{...item,hash:info.image_id,has_mask:info.has_mask,width:info.width,height:info.height}:item)}:previous);
      const size=latest.current.pageSize;
      const located=await locateImage(session,info.rel_path,session.page,size,signal);
      commitSelection(session,located,{datasetId:session.datasetId,imageId:info.image_id,relPath:info.rel_path});
      const target=session.target;
      if(target&&!target.signal.aborted){
        // A save-and-next operation locks its target before the save changes modified-time ordering.
        const located=await locateImage(session,target.image.relPath,target.located.data.page,size,signal);
        checkSession(session,signal);
        if(session.target===target)target.located=located;
      }
      void datasets.refetch();
    });
  };
  const navigatedImage=(target:ImageEditorNavigationTarget)=>{
    const session=editorSession.current;
    if(!session||!session.target||!sameImagePath(session.target.image.relPath,target.relPath)||session.datasetId!==target.datasetId)throw new DOMException('Aborted','AbortError');
    checkSession(session,session.target.signal);
    commitSelection(session,session.target.located,target);session.target=undefined;
  };
  const closeEditor=()=>{editorSession.current?.controller.abort();editorSession.current=null;setEditor(null);setEditError('');};
  const openEditor=()=>{if(!readOnly&&source&&image&&query.data&&!query.isPlaceholderData){
    editorSession.current?.controller.abort();
    editorSession.current={scope:editorScope,datasetId:source.source.id,search,sort,page:query.data.page,controller:new AbortController()};
    setEditError('');setEditor({datasetId:source.source.id,imageId:image.hash,relPath:image.rel_path,position:(query.data.page-1)*query.data.page_size+selectedIndex+1,total:query.data.total});
  }};
  const enableTraining=async()=>{
    const endpoint=versionConfigUrl(projectId,versionId);
    const config=await apiClient.get<{dataset?:Record<string,unknown>;[key:string]:unknown}>(endpoint,{silent:true});
    await apiClient.put(endpoint,{...config,dataset:{...config.dataset,masked_loss:true}},{silent:true});
    const destination=projectUrl(projectId,versionId,'train');allowedDestination.current=destination;closeEditor();navigate(destination);
  };
  return <section className={`caption-viewer${editing ? ' is-editing' : ''}`} aria-label={editing?text('涂抹与遮罩','Paint and mask'):text('图片与已有标签','Images and existing captions')}>
    {hasDataRouter&&<DatasetNavigationGuard shouldBlock={destination=>{
      if(allowedDestination.current===`${destination.pathname}${destination.search}${destination.hash}`){allowedDestination.current='';return false;}
      return !!editor;
    }} beforeLeave={async()=>{throw new Error(text('请先保存或关闭图片编辑器，再离开当前页面。','Save or close the image editor before leaving this page.'));}} onError={error=>setEditError(formatApiError(error))}/>}
    <div className="caption-viewer-controls"><div className="caption-viewer-toolbar" role="group" aria-label={text('选择图片目录','Choose image folder')}>
      <label className="caption-viewer-dataset"><span><Folder size={17}/>{text('图片目录','Image folder')}</span><StudioSelect searchable aria-label={text('图片目录','Image folder')} disabled={!source} value={source?.source.id||''} onValueChange={setDatasetId} options={(datasets.data||[]).map(item=>({value:item.source.id,label:datasetName(item.source.path)}))}/></label>
      <span className="caption-viewer-source-summary">{source?.source.is_reg ? text('正则图','Regularization') : text('训练集','Training')} · {source?.stats?.images ?? '—'} {text('张图片','images')}</span><button type="button" className="ui-btn" aria-label={editing ? text('刷新图片','Refresh images') : text('刷新标签','Refresh captions')} onClick={refresh} disabled={datasets.isFetching||query.isFetching}><RefreshCw size={16}/><span>{text('刷新','Refresh')}</span></button>
    </div>
    <form className="caption-viewer-filter-row" role="search" aria-label={text('筛选当前目录图片','Filter images in this folder')} onSubmit={event=>{event.preventDefault();setSearch(draftSearch.trim());setPage(1);setSelected('');}}><span>{text('图片筛选','Image filter')}</span><label className="caption-viewer-search"><Search size={16} aria-hidden="true"/><input aria-label={editing ? text('搜索图片','Search images') : text('搜索文件名或标签','Search filenames or captions')} value={draftSearch} onChange={event=>setDraftSearch(event.target.value)} placeholder={editing ? text('搜索文件名或标签','Search filenames or captions') : text('搜索文件名或标签','Search filenames or captions')}/></label><button type="submit" className="ui-btn">{text('搜索','Search')}</button>{search && <button type="button" className="ui-btn ui-btn-quiet" onClick={()=>{setSearch('');setDraftSearch('');setPage(1);}}>{text('清除','Clear')}</button>}<ImageSortSelect value={sort} onChange={changeSort}/><small>{text(`符合条件 ${query.data?.total ?? '—'} 张`,`${query.data?.total ?? '—'} matching images`)}</small></form>
      {pages>1 && <div className="caption-viewer-pagination" aria-busy={query.isFetching}><button type="button" className="ui-btn ui-btn-sm" disabled={page<=1||query.isFetching} onClick={()=>{setPage(page-1);setSelected('');}}>{text('上一页','Previous page')}</button><span>{query.isPlaceholderData ? query.data?.page : page} / {pages}</span>{query.isFetching && <Loader2 size={13} className="animate-spin" aria-label={text('读取中','Loading')}/>}<button type="button" className="ui-btn ui-btn-sm" disabled={page>=pages||query.isFetching} onClick={()=>{setPage(page+1);setSelected('');}}>{text('下一页','Next page')}</button></div>}
    </div>
    {error && <p role="alert" className="pipeline-error-text">{formatApiError(error)}</p>}
    {editError&&!editor&&<p role="alert" className="pipeline-error-text">{editError}</p>}
    {/* While loading, the grid stays in place with its status inside, so its column count sets the first page size. */}
    {!loading && !source ? <p className="caption-viewer-empty">{editing ? text('请先导入图片。','Import images first.') : text('请先导入图片。','Import images first.')}</p> : !loading && !items.length ? <p className="caption-viewer-empty">{search ? text('没有符合搜索条件的图片。','No images match the search.') : source?.index_status==='indexing' ? text('图片正在建立索引，完成后刷新即可查看。','Images are being indexed. Refresh when indexing finishes.') : text('此数据集暂无可查看的图片。','This dataset has no images to display.')}</p> : <>
      <div className="caption-viewer-columns" aria-busy={loading || query.isFetching}>
        <div className="caption-viewer-grid" ref={bindGrid} role="region" aria-label={text('图片缩略图','Image thumbnails')} tabIndex={0}>{loading ? <p role="status" className="caption-viewer-empty"><Loader2 size={18} className="animate-spin"/>{editing ? text('读取图片…','Loading images…') : text('读取图片与标签…','Loading images and captions…')}</p> : source && items.map(item=><button key={`${item.hash}/${item.rel_path}`} type="button" aria-label={editing ? text(`选择图片: ${item.rel_path}`,`Select image: ${item.rel_path}`) : text(`查看标签: ${item.rel_path}`,`View caption: ${item.rel_path}`)} aria-pressed={item===image} onClick={()=>setSelected(item.rel_path)}><div className="caption-viewer-thumbnail"><LazyImage src={apiUrl(`/datasets/${source.source.id}/images/${item.hash}/thumb?size=256`)} alt={item.rel_path} loading="lazy"/></div><span title={item.rel_path}>{item.rel_path}</span>{!editing && !item.caption && <small>{text('暂无标签','No caption')}</small>}</button>)}</div>
        {image && source && !loading && <div className="caption-viewer-detail" role="region" aria-label={editing ? text('图片详情','Image details') : text('图片与标签详情','Image and caption details')} tabIndex={0}>
          <div className="caption-viewer-image"><LazyImage src={apiUrl(`/datasets/${source.source.id}/images/${image.hash}/file`)} alt={text(`大图: ${image.rel_path}`,`Full image: ${image.rel_path}`)}/>{editing&&showMask&&image.has_mask&&<MaskOverlay key={`${imageKey}/${query.dataUpdatedAt}`} src={`${apiUrl(`/datasets/${source.source.id}/images/${image.hash}/mask`)}?rel_path=${encodeURIComponent(image.rel_path)}&v=${query.dataUpdatedAt}`} label={text('遮罩：红色区域不参与训练','Mask: red areas are left out of training')}/>}</div>
          <div className="caption-viewer-image-heading"><strong title={image.rel_path}>{image.rel_path}</strong><span>{image.width} × {image.height}</span>{editing&&image.has_mask&&<button type="button" className="ui-btn ui-btn-sm" aria-pressed={showMask} title={text('红色区域不参与训练','Red areas are left out of training')} onClick={()=>setShowMask(!showMask)}><Layers size={13}/>{text('显示遮罩','Show mask')}</button>}<div><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('上一张','Previous image')} disabled={selectedIndex===0} onClick={()=>{const previous=items[selectedIndex-1];setSelected(previous.rel_path);}}><ChevronLeft size={14}/></button><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('下一张','Next image')} disabled={selectedIndex===items.length-1} onClick={()=>{const next=items[selectedIndex+1];setSelected(next.rel_path);}}><ChevronRight size={14}/></button></div></div>
          {!editing && <div className="caption-viewer-caption"><h4>{text('已有标签','Existing caption')}{image.caption_format && <small> · {image.caption_format.toUpperCase()}</small>}</h4>{image.caption_error && <p role="alert" className="pipeline-error-text">{text('标签解析失败：','Caption parsing failed: ')}{image.caption_error}</p>}{image.caption ? <pre data-testid="existing-caption">{image.caption}</pre> : <p data-testid="existing-caption">{text('此图片暂无标签。','This image has no caption.')}</p>}</div>}
          {editing&&!readOnly&&<button type="button" className="ui-btn ui-btn-primary caption-viewer-editor" disabled={query.isFetching||query.isPlaceholderData} onClick={openEditor}><ImageIcon size={13}/><span>{text('打开涂抹与遮罩编辑器','Open paint and mask editor')}</span></button>}
          {(!editing||readOnly)&&<Link to={`/datasets/${source.source.id}?project=${encodeURIComponent(projectId)}&version=${encodeURIComponent(versionId)}`} className="ui-btn caption-viewer-editor"><ImageIcon size={13}/>{readOnly ? text('查看数据集详情','View dataset details') : text('打开逐图标签 / 遮罩编辑器','Open individual caption / mask editor')}</Link>}
        </div>}
      </div>

    </>}
    {editor&&<ImageEditor {...editor} navigation={{position:editor.position,total:editor.total,resolve:resolveImage,onNavigated:navigatedImage}} navigationError={editError} onClose={closeEditor} onSaved={savedImage} onReloadList={refresh} onEnableTraining={enableTraining}/>}
  </section>;
}
