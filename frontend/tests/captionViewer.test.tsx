import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import CaptionViewer from '../src/components/datasets/CaptionViewer';
import { apiClient, apiUrl } from '../src/api/client';
import i18n from '../src/i18n';

const source=(id:string)=>({source:{id,path:`D:\\素材\\${id}`},index_status:'ready'});
const picture=(hash:string,caption:string)=>({hash,rel_path:`${hash}.png`,width:512,height:768,caption,has_mask:false});
const caption='portrait, blue eyes\nA natural sentence, with punctuation.  原始标签';
let calls: {url:string;params:any}[];
function show(versionId='v_2',readOnly=false) {
  return render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><MemoryRouter><CaptionViewer projectId="p_1" versionId={versionId} readOnly={readOnly}/></MemoryRouter></QueryClientProvider>);
}
beforeEach(async()=>{
  await i18n.changeLanguage('zh-CN');calls=[];
  vi.spyOn(apiClient,'post');vi.spyOn(apiClient,'put');
  vi.spyOn(apiClient,'get').mockImplementation(async(url,options)=>{
    const params=options?.params as any;calls.push({url,params});
    if(url==='/projects/p_1/datasets')return [source('d_a'),source('d_b')] as any;
    if(url==='/datasets/d_b/images')return {items:[picture('other','another dataset')],total:1,page:1,page_size:40} as any;
    if(params?.q)return {items:[picture('searched',`match ${params.q}`)],total:1,page:1,page_size:40} as any;
    if(params?.page===2)return {items:[picture('last','page two')],total:41,page:2,page_size:40} as any;
    return {items:[picture('first',caption),picture('second','')],total:41,page:1,page_size:40} as any;
  });
});
afterEach(()=>vi.restoreAllMocks());
describe('existing caption viewer',()=>{
  it('reads the selected version and shows full unchanged captions with their images, including missing captions',async()=>{
    show('v_2',true);
    const original=await screen.findByTestId('existing-caption');
    expect(original.textContent).toBe(caption);
    expect(calls[0]).toEqual({url:'/projects/p_1/datasets',params:{version_id:'v_2'}});
    expect(screen.getByRole('img',{name:'大图: first.png'})).toHaveAttribute('src',apiUrl('/datasets/d_a/images/first/file'));
    expect(screen.getByRole('link',{name:'查看数据集详情'})).toHaveAttribute('href','/datasets/d_a');
    fireEvent.click(screen.getByRole('button',{name:'下一张'}));
    expect(screen.getByTestId('existing-caption')).toHaveTextContent('此图片暂无标签');
    expect(screen.getByRole('img',{name:'大图: second.png'})).toHaveAttribute('src',apiUrl('/datasets/d_a/images/second/file'));
    fireEvent.click(screen.getByRole('button',{name:'上一张'}));
    expect(screen.getByTestId('existing-caption').textContent).toBe(caption);
    expect(apiClient.post).not.toHaveBeenCalled();expect(apiClient.put).not.toHaveBeenCalled();
  });
  it('paginates, searches the actual server caption index and scopes a different dataset without writes',async()=>{
    show();await screen.findByTestId('existing-caption');
    fireEvent.click(screen.getByRole('button',{name:'下一页'}));
    await waitFor(()=>expect(screen.getByTestId('existing-caption')).toHaveTextContent('page two'));
    expect(calls.at(-1)).toEqual({url:'/datasets/d_a/images',params:{page:2,page_size:40,q:undefined}});
    fireEvent.change(screen.getByRole('textbox',{name:'搜索文件名或标签'}),{target:{value:'蓝色'}});
    fireEvent.click(screen.getByRole('button',{name:'搜索'}));
    await waitFor(()=>expect(screen.getByTestId('existing-caption')).toHaveTextContent('match 蓝色'));
    expect(calls.at(-1)).toEqual({url:'/datasets/d_a/images',params:{page:1,page_size:40,q:'蓝色'}});
    fireEvent.click(screen.getByRole('combobox',{name:'数据集'}));
    fireEvent.click(screen.getByRole('option',{name:'d_b'}));
    await waitFor(()=>expect(screen.getByTestId('existing-caption')).toHaveTextContent('another dataset'));
    expect(calls.at(-1)?.url).toBe('/datasets/d_b/images');
    expect(screen.getByRole('link',{name:'打开逐图标签 / 遮罩编辑器'})).toHaveAttribute('href','/datasets/d_b');
    expect(apiClient.post).not.toHaveBeenCalled();expect(apiClient.put).not.toHaveBeenCalled();
  });
  it('resets page, search and selection before querying a reused viewer in another version',async()=>{
    vi.mocked(apiClient.get).mockImplementation(async(url,options)=>{
      const params=options?.params as any;calls.push({url,params});
      if(url==='/projects/p_1/datasets')return [source(params.version_id==='v_2'?'d_old':'d_new')] as any;
      return {items:[picture(url.includes('d_new')?'fresh':'old',url.includes('d_new')?'new version caption':'old version caption')],total:url.includes('d_new')?1:41,page:params.page,page_size:40} as any;
    });
    const client=new QueryClient({defaultOptions:{queries:{retry:false}}});
    const view=(versionId:string)=><QueryClientProvider client={client}><MemoryRouter><CaptionViewer projectId="p_1" versionId={versionId}/></MemoryRouter></QueryClientProvider>;
    const mounted=render(view('v_2'));await screen.findByTestId('existing-caption');
    fireEvent.change(screen.getByRole('textbox',{name:'搜索文件名或标签'}),{target:{value:'old search'}});
    fireEvent.click(screen.getByRole('button',{name:'搜索'}));
    await waitFor(()=>expect(calls.at(-1)?.params.q).toBe('old search'));
    await waitFor(()=>expect(screen.getByRole('button',{name:'下一页'})).toBeEnabled());
    fireEvent.click(screen.getByRole('button',{name:'下一页'}));
    await waitFor(()=>expect(calls.at(-1)?.params.page).toBe(2));
    const previousRequests=calls.length;
    mounted.rerender(view('v_3'));
    await waitFor(()=>expect(screen.getByTestId('existing-caption')).toHaveTextContent('new version caption'));
    expect(screen.getByRole('textbox',{name:'搜索文件名或标签'})).toHaveValue('');
    expect(screen.getByRole('img',{name:'大图: fresh.png'})).toHaveAttribute('src',apiUrl('/datasets/d_new/images/fresh/file'));
    expect(calls.slice(previousRequests).filter(call=>call.url.includes('/images'))).toEqual([{url:'/datasets/d_new/images',params:{page:1,page_size:40,q:undefined}}]);
    expect(apiClient.post).not.toHaveBeenCalled();
  });
  it('clears a removed source without querying it again and starts a replacement source at page one',async()=>{
    let sources=[source('d_a')];
    vi.mocked(apiClient.get).mockImplementation(async(url,options)=>{
      const params=options?.params as any;calls.push({url,params});
      if(url==='/projects/p_1/datasets')return sources as any;
      return {items:[picture('one',url)],total:41,page:params.page,page_size:40} as any;
    });
    show();await screen.findByTestId('existing-caption');
    fireEvent.click(screen.getByRole('button',{name:'下一页'}));
    await waitFor(()=>expect(calls.at(-1)?.params.page).toBe(2));
    await waitFor(()=>expect(screen.getByRole('button',{name:'刷新标签'})).toBeEnabled());
    sources=[];fireEvent.click(screen.getByRole('button',{name:'刷新标签'}));
    await screen.findByText('导入图片后，即可在这里查看对应标签。');
    expect(screen.queryByTestId('existing-caption')).not.toBeInTheDocument();
    const previousRequests=calls.length;
    sources=[source('d_replacement')];fireEvent.click(screen.getByRole('button',{name:'刷新标签'}));
    await screen.findByTestId('existing-caption');
    expect(calls.slice(previousRequests).filter(call=>call.url.includes('/images'))).toEqual([{url:'/datasets/d_replacement/images',params:{page:1,page_size:40,q:undefined}}]);
  });
  it.each([40,0])('recovers an out-of-range last page after the same dataset shrinks to %i images',async(totalAfter)=>{
    let total=41;
    vi.mocked(apiClient.get).mockImplementation(async(url,options)=>{
      const params=options?.params as any;calls.push({url,params});
      if(url==='/projects/p_1/datasets')return [source('d_a')] as any;
      return {items:total>(params.page-1)*40?[picture(`page${params.page}`,`caption page ${params.page}`)]:[],total,page:params.page,page_size:40} as any;
    });
    show();await screen.findByTestId('existing-caption');
    fireEvent.click(screen.getByRole('button',{name:'下一页'}));
    await waitFor(()=>expect(screen.getByTestId('existing-caption')).toHaveTextContent('caption page 2'));
    await waitFor(()=>expect(screen.getByRole('button',{name:'刷新标签'})).toBeEnabled());
    total=totalAfter;
    fireEvent.click(screen.getByRole('button',{name:'刷新标签'}));
    if(totalAfter)await waitFor(()=>expect(screen.getByTestId('existing-caption')).toHaveTextContent('caption page 1'));
    else await screen.findByText('此数据集暂无可查看的图片。');
    await waitFor(()=>expect(calls.at(-1)).toEqual({url:'/datasets/d_a/images',params:{page:1,page_size:40,q:undefined}}));
    expect(screen.queryByRole('button',{name:'上一页'})).not.toBeInTheDocument();
  });
  it('handles an empty version without requesting an unrelated dataset or requiring inspection',async()=>{
    vi.mocked(apiClient.get).mockResolvedValue([] as any);
    show('v_empty');await screen.findByText('导入图片后，即可在这里查看对应标签。');
    fireEvent.click(screen.getByRole('button',{name:'刷新标签'}));
    await waitFor(()=>expect(apiClient.get).toHaveBeenCalledTimes(2));
    expect(vi.mocked(apiClient.get).mock.calls.every(([url])=>url==='/projects/p_1/datasets')).toBe(true);
    expect(apiClient.post).not.toHaveBeenCalled();
  });
});
