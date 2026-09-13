import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createMemoryRouter, MemoryRouter, RouterProvider } from 'react-router-dom';
import CaptionViewer from '../src/components/datasets/CaptionViewer';
import { apiClient } from '../src/api/client';
import '../src/i18n';
vi.mock('../src/components/masks/ImageEditor',()=>({default:(props:any)=><div role="dialog"><p>{props.datasetId} / {props.imageId} / {props.relPath}</p>{props.navigationError&&<p role="alert">{props.navigationError}</p>}<button onClick={props.onClose}>close editor</button><button onClick={()=>{props.onReloadList();props.onClose();}}>close and refresh list</button><button onClick={()=>props.onSaved({image_id:'newhash'})}>saved pixels</button><button onClick={()=>void props.onEnableTraining()}>enable masked training</button></div>}));
const config={model:{family:'anima',dit_path:'D:/weights/model.safetensors'},dataset:{sources:[{path:'D:/source',repeats:3}],masked_loss:false},loop:{epochs:8}};
let indexed=false;
beforeEach(()=>{
  indexed=false;vi.spyOn(apiClient,'put').mockResolvedValue(config);
  vi.spyOn(apiClient,'get').mockImplementation(async(url)=>{
    if(url==='/projects/p/config?version_id=v2')return config as any;
    if(url==='/projects/p/datasets')return [{source:{id:'d_v2',path:'D:/source'},index_status:'ready'}] as any;
    if(url==='/datasets/d_v2/images')return {items:[{hash:indexed?'newhash':'hash',rel_path:'person.png',width:64,height:64,caption:'original caption'}],total:1,page:1,page_size:40} as any;
    throw new Error(`Unexpected request ${url}`);
  });
});
afterEach(()=>vi.restoreAllMocks());
const client=()=>new QueryClient({defaultOptions:{queries:{retry:false}}});
describe('version-scoped paint entry',()=>{
  it('reopens the exact selected path with its refreshed hash after closing a conflicting editor',async()=>{
    render(<QueryClientProvider client={client()}><MemoryRouter><CaptionViewer projectId="p" versionId="v2" editing/></MemoryRouter></QueryClientProvider>);
    fireEvent.click(await screen.findByRole('button',{name:'打开涂抹与遮罩编辑器'}));expect(screen.getByRole('dialog')).toHaveTextContent('d_v2 / hash / person.png');
    indexed=true;fireEvent.click(screen.getByRole('button',{name:'close and refresh list'}));expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    await waitFor(()=>expect(screen.getByRole('img',{name:'大图: person.png'})).toHaveAttribute('src',expect.stringContaining('/newhash/file')));
    await waitFor(()=>expect(screen.getByRole('button',{name:'打开涂抹与遮罩编辑器'})).toBeEnabled());fireEvent.click(screen.getByRole('button',{name:'打开涂抹与遮罩编辑器'}));
    expect(screen.getByRole('dialog')).toHaveTextContent('d_v2 / newhash / person.png');expect(apiClient.put).not.toHaveBeenCalled();
  });
  it('opens the selected image without pipeline inspection and refreshes its changed hash without dropping the editor',async()=>{
    render(<QueryClientProvider client={client()}><MemoryRouter><CaptionViewer projectId="p" versionId="v2" editing/></MemoryRouter></QueryClientProvider>);
    fireEvent.click(await screen.findByRole('button',{name:'打开涂抹与遮罩编辑器'}));expect(screen.getByRole('dialog')).toHaveTextContent('d_v2 / hash / person.png');
    indexed=true;fireEvent.click(screen.getByRole('button',{name:'saved pixels'}));
    await waitFor(()=>expect(screen.getByRole('img',{name:'大图: person.png'})).toHaveAttribute('src',expect.stringContaining('/newhash/file')));
    expect(screen.getByRole('dialog')).toBeInTheDocument();expect(screen.queryByTestId('existing-caption')).not.toBeInTheDocument();
    expect(vi.mocked(apiClient.get).mock.calls.some(([url])=>url.includes('pipeline'))).toBe(false);expect(apiClient.put).not.toHaveBeenCalled();
  });
  it('retains the editor on browser Back and explains the save-or-close requirement inside it',async()=>{
    const router=createMemoryRouter([{path:'/previous',element:<p>Previous workspace</p>},{path:'/data',element:<CaptionViewer projectId="p" versionId="v2" editing/>}],{initialEntries:['/previous','/data'],initialIndex:1});
    render(<QueryClientProvider client={client()}><RouterProvider router={router}/></QueryClientProvider>);
    fireEvent.click(await screen.findByRole('button',{name:'打开涂抹与遮罩编辑器'}));await act(()=>router.navigate(-1));
    expect(await screen.findByRole('alert')).toHaveTextContent('请先保存或关闭图片编辑器');expect(router.state.location.pathname).toBe('/data');expect(screen.getByRole('dialog')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button',{name:'close editor'}));await act(()=>router.navigate(-1));expect(await screen.findByText('Previous workspace')).toBeInTheDocument();
  });
  it('enables masked training only in the explicit version, preserves other configuration and navigates after saving',async()=>{
    const router=createMemoryRouter([{path:'/data',element:<CaptionViewer projectId="p" versionId="v2" editing/>},{path:'/projects/p/v/v2/train',element:<p>Training parameters</p>}],{initialEntries:['/data']});
    render(<QueryClientProvider client={client()}><RouterProvider router={router}/></QueryClientProvider>);
    fireEvent.click(await screen.findByRole('button',{name:'打开涂抹与遮罩编辑器'}));fireEvent.click(screen.getByRole('button',{name:'enable masked training'}));
    await screen.findByText('Training parameters');expect(apiClient.put).toHaveBeenCalledWith('/projects/p/config?version_id=v2',{...config,dataset:{...config.dataset,masked_loss:true}},{silent:true});
  });
  it('keeps archived paint pages read-only without showing a caption editor',async()=>{
    render(<QueryClientProvider client={client()}><MemoryRouter><CaptionViewer projectId="p" versionId="v2" editing readOnly/></MemoryRouter></QueryClientProvider>);
    await screen.findByRole('img',{name:'大图: person.png'});expect(screen.queryByTestId('existing-caption')).not.toBeInTheDocument();expect(screen.queryByRole('button',{name:'打开涂抹与遮罩编辑器'})).not.toBeInTheDocument();expect(screen.getByRole('link',{name:'查看数据集详情'})).toBeInTheDocument();expect(apiClient.put).not.toHaveBeenCalled();
  });
});
