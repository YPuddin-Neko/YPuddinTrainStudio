import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ChevronLeft, ChevronRight, Image as ImageIcon, Loader2, RefreshCw, Search } from 'lucide-react';
import { apiClient, apiUrl } from '../../api/client';
import type { DatasetInfo, DatasetImagesPage } from '../../api/types';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import StudioSelect from '../StudioSelect';
import './caption-viewer.css';

const datasetName = (path: string) => path.replace(/\\/g,'/').split('/').filter(Boolean).pop()?.replace(/^(?:d_[0-9a-f]+-)+/,'') || path;

export default function CaptionViewer({projectId,versionId,readOnly=false}: {projectId:string;versionId:string;readOnly?:boolean}) {
  const text=useWorkspaceText();
  const gridRef=useRef<HTMLDivElement>(null);
  const displayedPage=useRef('');
  const [datasetId,setDatasetId]=useState('');
  const [navigation,setNavigation]=useState({scope:'',page:1,draftSearch:'',search:'',selected:''});
  const datasets=useQuery({queryKey:['caption-datasets',projectId,versionId],queryFn:({signal})=>apiClient.get<DatasetInfo[]>(`/projects/${projectId}/datasets`,{params:{version_id:versionId},signal,silent:true})});
  const source=datasets.data?.find(item=>item.source.id===datasetId) || datasets.data?.[0];
  // Derive reset state before querying: a reused viewer must never request the old page in a new source/version.
  const scope=JSON.stringify([projectId,versionId,source?.source.id]);
  const empty={scope,page:1,draftSearch:'',search:'',selected:''};
  const {page,draftSearch,search,selected}=navigation.scope===scope?navigation:empty;
  const updateNavigation=(patch:Partial<typeof navigation>)=>setNavigation(previous=>({...(previous.scope===scope?previous:empty),...patch}));
  const setPage=(page:number)=>updateNavigation({page});
  const setDraftSearch=(draftSearch:string)=>updateNavigation({draftSearch});
  const setSearch=(search:string)=>updateNavigation({search});
  const setSelected=(selected:string)=>updateNavigation({selected});
  const query=useQuery<DatasetImagesPage>({
    queryKey:['caption-images',projectId,versionId,source?.source.id,page,search],enabled:!!source,
    placeholderData:(previous,previousQuery)=>previousQuery?.queryKey[1]===projectId && previousQuery.queryKey[2]===versionId && previousQuery.queryKey[3]===source?.source.id && previousQuery.queryKey[5]===search ? previous : undefined,
    queryFn:({signal})=>apiClient.get<DatasetImagesPage>(`/datasets/${source!.source.id}/images`,{params:{page,page_size:40,q:search||undefined},signal,silent:true}),
  });
  const items=query.data?.items || [];
  const selectedIndex=Math.max(0,items.findIndex(item=>`${item.hash}/${item.rel_path}`===selected));
  const image=items[selectedIndex];
  const pages=Math.max(1,Math.ceil((query.data?.total||0)/40));
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
  },[scope,search,actualPage,query.isPlaceholderData,imageKey]);
  useEffect(()=>{
    if(!query.data || query.isFetching || query.isError || page<=pages)return;
    // Refresh can remove the last page after curation. Only correct the scope that returned this result.
    setNavigation(previous=>previous.scope===scope?{...previous,page:pages,selected:''}:previous);
  },[query.data,query.isFetching,query.isError,page,pages,scope]);
  const loading=datasets.isPending || (!!source && query.isPending);
  const error=datasets.error || query.error;
  const refresh=()=>{void datasets.refetch();if(source)void query.refetch();};
  return <section className="caption-viewer" aria-label={text('图片与已有标签','Images and existing captions')}>
    <div className="caption-viewer-controls"><div className="caption-viewer-toolbar">
      {(datasets.data?.length||0)>1 && <label>{text('数据集','Dataset')}<StudioSelect aria-label={text('数据集','Dataset')} value={source?.source.id||''} onValueChange={setDatasetId} options={(datasets.data||[]).map(item=>({value:item.source.id,label:datasetName(item.source.path)}))}/></label>}
      <form onSubmit={event=>{event.preventDefault();setSearch(draftSearch.trim());setPage(1);setSelected('');}}><Search size={14}/><input aria-label={text('搜索文件名或标签','Search filenames or captions')} value={draftSearch} onChange={event=>setDraftSearch(event.target.value)} placeholder={text('搜索文件名或标签','Search filenames or captions')}/><button type="submit">{text('搜索','Search')}</button>{search && <button type="button" onClick={()=>{setSearch('');setDraftSearch('');setPage(1);}}>{text('清除','Clear')}</button>}</form>
      <span>{query.data?.total ?? '—'} {text('张图片','images')}</span><button type="button" aria-label={text('刷新标签','Refresh captions')} onClick={refresh} disabled={datasets.isFetching||query.isFetching}><RefreshCw size={14}/></button>
    </div>
      {pages>1 && <div className="caption-viewer-pagination" aria-busy={query.isFetching}><button type="button" disabled={page<=1||query.isFetching} onClick={()=>{setPage(page-1);setSelected('');}}>{text('上一页','Previous page')}</button><span>{query.isPlaceholderData ? query.data?.page : page} / {pages}</span>{query.isFetching && <Loader2 size={13} className="animate-spin" aria-label={text('读取中','Loading')}/>}<button type="button" disabled={page>=pages||query.isFetching} onClick={()=>{setPage(page+1);setSelected('');}}>{text('下一页','Next page')}</button></div>}
    </div>
    {error && <p role="alert" className="pipeline-error-text">{formatApiError(error)}</p>}
    {loading ? <p role="status" className="caption-viewer-empty"><Loader2 size={18} className="animate-spin"/>{text('读取图片与标签…','Loading images and captions…')}</p> : !source ? <p className="caption-viewer-empty">{text('导入图片后，即可在这里查看对应标签。','Import images to view their captions here.')}</p> : !items.length ? <p className="caption-viewer-empty">{search ? text('没有符合搜索条件的图片。','No images match the search.') : source.index_status==='indexing' ? text('图片正在建立索引，完成后刷新即可查看。','Images are being indexed. Refresh when indexing finishes.') : text('此数据集暂无可查看的图片。','This dataset has no images to display.')}</p> : <>
      <div className="caption-viewer-columns" aria-busy={query.isFetching}>
        <div className="caption-viewer-grid" ref={gridRef} role="region" aria-label={text('图片缩略图','Image thumbnails')} tabIndex={0}>{items.map(item=><button key={`${item.hash}/${item.rel_path}`} type="button" aria-label={text(`查看标签: ${item.rel_path}`,`View caption: ${item.rel_path}`)} aria-pressed={item===image} onClick={()=>setSelected(`${item.hash}/${item.rel_path}`)}><img src={apiUrl(`/datasets/${source.source.id}/images/${item.hash}/thumb?size=256`)} alt={item.rel_path} loading="lazy"/><span title={item.rel_path}>{item.rel_path}</span>{!item.caption && <small>{text('暂无标签','No caption')}</small>}</button>)}</div>
        {image && <div className="caption-viewer-detail" role="region" aria-label={text('图片与标签详情','Image and caption details')} tabIndex={0}>
          <div className="caption-viewer-image"><img src={apiUrl(`/datasets/${source.source.id}/images/${image.hash}/file`)} alt={text(`大图: ${image.rel_path}`,`Full image: ${image.rel_path}`)}/></div>
          <div className="caption-viewer-image-heading"><strong title={image.rel_path}>{image.rel_path}</strong><span>{image.width} × {image.height}</span><div><button type="button" aria-label={text('上一张','Previous image')} disabled={selectedIndex===0} onClick={()=>{const previous=items[selectedIndex-1];setSelected(`${previous.hash}/${previous.rel_path}`);}}><ChevronLeft size={14}/></button><button type="button" aria-label={text('下一张','Next image')} disabled={selectedIndex===items.length-1} onClick={()=>{const next=items[selectedIndex+1];setSelected(`${next.hash}/${next.rel_path}`);}}><ChevronRight size={14}/></button></div></div>
          <div className="caption-viewer-caption"><h4>{text('已有标签','Existing caption')}</h4>{image.caption ? <pre data-testid="existing-caption">{image.caption}</pre> : <p data-testid="existing-caption">{text('此图片暂无标签。','This image has no caption.')}</p>}</div>
          <Link to={`/datasets/${source.source.id}`} className="caption-viewer-editor"><ImageIcon size={13}/>{readOnly ? text('查看数据集详情','View dataset details') : text('打开逐图标签 / 遮罩编辑器','Open individual caption / mask editor')}</Link>
        </div>}
      </div>

    </>}
  </section>;
}
