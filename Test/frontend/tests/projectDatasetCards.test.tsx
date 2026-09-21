import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import ProjectDatasetCards, { type WorkspaceDataset } from '../../../frontend/src/components/datasets/ProjectDatasetCards';
import i18n from '../../../frontend/src/i18n';

vi.mock('../../../frontend/src/events/useEventStream', () => ({ useEventStream: () => {} }));
const server = setupServer();
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
beforeEach(async () => { await i18n.changeLanguage('zh-CN'); });
afterEach(() => { server.resetHandlers(); vi.restoreAllMocks(); });
afterAll(() => server.close());
const source = { id:'d_portraits', project_id:'p_owner', version_id:'v_old', path:'D:/studio/train/人物与服装', repeats:2, caption_ext:'auto', is_reg:false, prior_weight:1, class_prompt:null, created_at:1 };
const row = (id = source.id, status = 'ready', count = 20): WorkspaceDataset => ({ source:{...source,id}, index_status:status, stats:{images:count,captioned:18,masks:4} });
function show(datasets: WorkspaceDataset[], onRefresh = vi.fn()) {
  return render(<QueryClientProvider client={new QueryClient({ defaultOptions:{queries:{retry:false}} })}><MemoryRouter><ProjectDatasetCards datasets={datasets} projectId="p_current" versionId="v_current" onRefresh={onRefresh}/></MemoryRouter></QueryClientProvider>);
}

describe('project dataset preview cards', () => {
  it('requests one complete cover and keeps the dataset kind outside the image area', async () => {
    const urls: URL[] = [];
    server.use(http.get('/api/datasets/d_portraits/images', ({request}) => { urls.push(new URL(request.url)); return HttpResponse.json({total:20,items:Array.from({length:4},(_,i)=>({hash:`image${i}`,rel_path:`人物/${i}.png`}))}); }));
    show([row()]);
    const card = screen.getByTestId('dataset-card-d_portraits');
    await waitFor(() => expect(within(card).getAllByRole('img')).toHaveLength(1));
    expect(urls).toHaveLength(1); expect(urls[0].searchParams.get('page_size')).toBe('1'); expect(urls[0].searchParams.get('page')).toBe('1');
    expect(card).toHaveAttribute('href','/datasets/d_portraits?project=p_owner&version=v_old');
    expect(card).toHaveAccessibleName('打开数据集：人物与服装');
    expect(card).toHaveTextContent('20 张图片'); expect(card).toHaveTextContent('18 份标签'); expect(card).toHaveTextContent('4 张遮罩');
    const image = within(card).getByRole('img', { name: '人物/0.png' });
    expect(image).toHaveAttribute('loading','lazy'); expect(image.getAttribute('src')).toMatch(/\/thumb\?size=512$/);
    expect(within(card).getByText('训练集').closest('.project-dataset-card-body')).not.toBeNull();
    expect(card.querySelector('.project-dataset-preview')).not.toHaveTextContent('训练集');
    expect(card.querySelector('.project-dataset-kind svg')).toBeNull();
  });

  it('distinguishes an empty library, an empty folder, indexing and failure without fetching unusable previews', () => {
    const mounted = show([]);
    expect(screen.getByTestId('datasets-empty')).toHaveTextContent('还没有训练图片');
    mounted.unmount();
    show([row('empty','ready',0), row('working','indexing',0), {...row('failed','failed',0),stats:{images:0,captioned:0,masks:0,error:'图片格式无法读取'}}]);
    expect(within(screen.getByTestId('dataset-card-empty')).getByText('目录中还没有图片')).toBeInTheDocument();
    expect(screen.getByTestId('dataset-card-working')).toHaveTextContent('索引中');
    expect(screen.getByTestId('dataset-card-failed')).toHaveTextContent('索引失败');
    expect(screen.getByTestId('dataset-card-failed')).toHaveTextContent('图片格式无法读取');
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
  });

  it('keeps failed preview loads navigable and retries them with the library refresh', async () => {
    let fail = true; const onRefresh = vi.fn();
    server.use(http.get('/api/datasets/d_portraits/images', () => fail ? HttpResponse.json({error:{message:'Unavailable'}},{status:503}) : HttpResponse.json({total:20,items:[{hash:'recovered',rel_path:'恢复.png'}]})));
    show([row()],onRefresh);
    expect(await screen.findByText('预览暂不可用')).toBeInTheDocument();
    expect(screen.getByTestId('dataset-card-d_portraits')).toHaveAttribute('href',expect.stringContaining('/datasets/d_portraits'));
    fail=false; fireEvent.click(screen.getByRole('button',{name:'刷新索引状态'}));
    expect(await screen.findByRole('img',{name:'恢复.png'})).toBeInTheDocument(); expect(onRefresh).toHaveBeenCalledOnce();
  });

  it('keeps the full long folder name accessible and refreshes after indexing completes', async () => {
    const longName='人物甲与服装细节'.repeat(12);
    const pending={...row('d_long','indexing',0),source:{...source,id:'d_long',path:`/train/${longName}`,is_reg:true}};
    const client=new QueryClient({defaultOptions:{queries:{retry:false}}});
    const view=(dataset:WorkspaceDataset)=><QueryClientProvider client={client}><MemoryRouter><ProjectDatasetCards datasets={[dataset]} projectId="p_owner" onRefresh={vi.fn()}/></MemoryRouter></QueryClientProvider>;
    server.use(http.get('/api/datasets/d_long/images',()=>HttpResponse.json({total:1,items:[{hash:'ready',rel_path:'就绪.png'}]})));
    const mounted=render(view(pending));
    expect(screen.getByRole('heading',{name:longName})).toHaveAttribute('title',longName);
    const card = screen.getByTestId('dataset-card-d_long');
    expect(within(card).getByText('正则图').closest('.project-dataset-card-body')).not.toBeNull();
    expect(card.querySelector('.project-dataset-preview')).not.toHaveTextContent('正则图');
    mounted.rerender(view({...pending,index_status:'ready',stats:{images:1,captioned:1,masks:0}}));
    expect(await screen.findByRole('img',{name:'就绪.png'})).toBeInTheDocument();
  });
});
