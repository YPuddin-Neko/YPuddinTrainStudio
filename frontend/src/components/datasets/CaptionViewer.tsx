import { useCallback, useContext, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Link, useNavigate, UNSAFE_DataRouterContext } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ChevronLeft, ChevronRight, Folder, Image as ImageIcon, Layers, Loader2, RefreshCw, Search } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import type { DatasetInfo, DatasetImagesPage } from '../../api/types';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import StudioSelect from '../StudioSelect';
import { LazyImage } from '../Loading';
import ImageEditor from '../masks/ImageEditor';
import MaskOverlay from './MaskOverlay';
import ImageSortSelect, { type ImageSort } from './ImageSortSelect';
import { useGridPageSize } from '../../api/hooks/useGridPageSize';
import DatasetNavigationGuard from './DatasetNavigationGuard';
import { projectUrl, versionConfigUrl } from '../../utils/projectVersions';
import './caption-viewer.css';

const datasetName = (path: string) => path.replace(/\\/g,'/').split('/').filter(Boolean).pop()?.replace(/^(?:d_[0-9a-f]+-)+/,'') || path;

export default function CaptionViewer({projectId,versionId,readOnly=false,editing=false,initialDatasetId=''}: {projectId:string;versionId:string;readOnly?:boolean;editing?:boolean;initialDatasetId?:string}) {
  const text=useWorkspaceText();
  const navigate=useNavigate();
  const hasDataRouter=!!useContext(UNSAFE_DataRouterContext);
  const [editor,setEditor]=useState<{datasetId:string;imageId:string;relPath:string}|null>(null);
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
  const setSelected=(selected:string)=>updateNavigation({selected,selectionIndex:Math.max(0,items.findIndex(item=>`${item.hash}/${item.rel_path}`===selected))});
  const changeSort=(value:ImageSort)=>{
    setSort(value);
    setNavigation(previous=>({...previous.scope===scope?previous:empty,pageSize:gridPage.pageSize,page:1,selected:'',selectionIndex:0}));
  };
  const query=useQuery<DatasetImagesPage>({
    queryKey:['caption-images',projectId,versionId,source?.source.id,page,search,sort,gridPage.pageSize],enabled:!!source,
    placeholderData:(previous,previousQuery)=>previousQuery?.queryKey[1]===projectId && previousQuery.queryKey[2]===versionId && previousQuery.queryKey[3]===source?.source.id && previousQuery.queryKey[5]===search && previousQuery.queryKey[6]===sort && previousQuery.queryKey[7]===gridPage.pageSize ? previous : undefined,
    queryFn:({signal})=>apiClient.get<DatasetImagesPage>(`/datasets/${source!.source.id}/images`,{params:{page,page_size:gridPage.pageSize,q:search||undefined,sort},signal,silent:true}),
  });
  const items=query.data?.items || [];
  const selectedIndex=selected?Math.max(0,items.findIndex(item=>`${item.hash}/${item.rel_path}`===selected)):Math.min(current.selectionIndex,Math.max(0,items.length-1));
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
  const openEditor=()=>{if(!readOnly&&source&&image&&!query.isPlaceholderData){setEditError('');setEditor({datasetId:source.source.id,imageId:image.hash,relPath:image.rel_path});}};
  const enableTraining=async()=>{
    const endpoint=versionConfigUrl(projectId,versionId);
    const config=await apiClient.get<{dataset?:Record<string,unknown>;[key:string]:unknown}>(endpoint,{silent:true});
    await apiClient.put(endpoint,{...config,dataset:{...config.dataset,masked_loss:true}},{silent:true});
    const destination=projectUrl(projectId,versionId,'train');allowedDestination.current=destination;setEditor(null);navigate(destination);
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
    {loading ? <p role="status" className="caption-viewer-empty"><Loader2 size={18} className="animate-spin"/>{editing ? text('读取图片…','Loading images…') : text('读取图片与标签…','Loading images and captions…')}</p> : !source ? <p className="caption-viewer-empty">{editing ? text('请先导入图片。','Import images first.') : text('请先导入图片。','Import images first.')}</p> : !items.length ? <p className="caption-viewer-empty">{search ? text('没有符合搜索条件的图片。','No images match the search.') : source.index_status==='indexing' ? text('图片正在建立索引，完成后刷新即可查看。','Images are being indexed. Refresh when indexing finishes.') : text('此数据集暂无可查看的图片。','This dataset has no images to display.')}</p> : <>
      <div className="caption-viewer-columns" aria-busy={query.isFetching}>
        <div className="caption-viewer-grid" ref={bindGrid} role="region" aria-label={text('图片缩略图','Image thumbnails')} tabIndex={0}>{items.map(item=><button key={`${item.hash}/${item.rel_path}`} type="button" aria-label={editing ? text(`选择图片: ${item.rel_path}`,`Select image: ${item.rel_path}`) : text(`查看标签: ${item.rel_path}`,`View caption: ${item.rel_path}`)} aria-pressed={item===image} onClick={()=>setSelected(`${item.hash}/${item.rel_path}`)}><div className="caption-viewer-thumbnail"><LazyImage src={apiUrl(`/datasets/${source.source.id}/images/${item.hash}/thumb?size=256`)} alt={item.rel_path} loading="lazy"/></div><span title={item.rel_path}>{item.rel_path}</span>{!editing && !item.caption && <small>{text('暂无标签','No caption')}</small>}</button>)}</div>
        {image && <div className="caption-viewer-detail" role="region" aria-label={editing ? text('图片详情','Image details') : text('图片与标签详情','Image and caption details')} tabIndex={0}>
          <div className="caption-viewer-image"><LazyImage src={apiUrl(`/datasets/${source.source.id}/images/${image.hash}/file`)} alt={text(`大图: ${image.rel_path}`,`Full image: ${image.rel_path}`)}/>{editing&&showMask&&image.has_mask&&<MaskOverlay key={`${imageKey}/${query.dataUpdatedAt}`} src={`${apiUrl(`/datasets/${source.source.id}/images/${image.hash}/mask`)}?rel_path=${encodeURIComponent(image.rel_path)}&v=${query.dataUpdatedAt}`} label={text('遮罩：红色区域不参与训练','Mask: red areas are left out of training')}/>}</div>
          <div className="caption-viewer-image-heading"><strong title={image.rel_path}>{image.rel_path}</strong><span>{image.width} × {image.height}</span>{editing&&image.has_mask&&<button type="button" className="ui-btn ui-btn-sm" aria-pressed={showMask} title={text('红色区域不参与训练','Red areas are left out of training')} onClick={()=>setShowMask(!showMask)}><Layers size={13}/>{text('显示遮罩','Show mask')}</button>}<div><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('上一张','Previous image')} disabled={selectedIndex===0} onClick={()=>{const previous=items[selectedIndex-1];setSelected(`${previous.hash}/${previous.rel_path}`);}}><ChevronLeft size={14}/></button><button type="button" className="ui-btn ui-btn-sm ui-btn-icon" aria-label={text('下一张','Next image')} disabled={selectedIndex===items.length-1} onClick={()=>{const next=items[selectedIndex+1];setSelected(`${next.hash}/${next.rel_path}`);}}><ChevronRight size={14}/></button></div></div>
          {!editing && <div className="caption-viewer-caption"><h4>{text('已有标签','Existing caption')}{image.caption_format && <small> · {image.caption_format.toUpperCase()}</small>}</h4>{image.caption_error && <p role="alert" className="pipeline-error-text">{text('标签解析失败：','Caption parsing failed: ')}{image.caption_error}</p>}{image.caption ? <pre data-testid="existing-caption">{image.caption}</pre> : <p data-testid="existing-caption">{text('此图片暂无标签。','This image has no caption.')}</p>}</div>}
          {editing&&!readOnly&&<button type="button" className="ui-btn ui-btn-primary caption-viewer-editor" disabled={query.isFetching||query.isPlaceholderData} onClick={openEditor}><ImageIcon size={13}/><span>{text('打开涂抹与遮罩编辑器','Open paint and mask editor')}</span></button>}
          {(!editing||readOnly)&&<Link to={`/datasets/${source.source.id}?project=${encodeURIComponent(projectId)}&version=${encodeURIComponent(versionId)}`} className="ui-btn caption-viewer-editor"><ImageIcon size={13}/>{readOnly ? text('查看数据集详情','View dataset details') : text('打开逐图标签 / 遮罩编辑器','Open individual caption / mask editor')}</Link>}
        </div>}
      </div>

    </>}
    {editor&&<ImageEditor {...editor} navigationError={editError} onClose={()=>{setEditor(null);setEditError('');}} onSaved={()=>refresh()} onReloadList={refresh} onEnableTraining={enableTraining}/>}
  </section>;
}
