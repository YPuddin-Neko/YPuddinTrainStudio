import {act,fireEvent,render,screen,waitFor,within} from '@testing-library/react';
import {afterAll,afterEach,beforeAll,expect,it} from 'vitest';
import {http,HttpResponse} from 'msw';
import {setupServer} from 'msw/node';
import {MemoryRouter,Route,Routes} from 'react-router-dom';
import {QueryClient,QueryClientProvider} from '@tanstack/react-query';
import {handlers} from '../mocks/handlers';
import TrainConfig from '../../../frontend/src/pages/TrainConfig/TrainConfig';
import '../../../frontend/src/i18n';

const server=setupServer(...handlers);
beforeAll(()=>server.listen({onUnhandledRequest:'error'}));
afterEach(()=>server.resetHandlers());
afterAll(()=>server.close());
const result=(repeats:number)=>({ok:true,errors:[],warnings:[],params:{},source_balance:[
  {source_index:0,path:'/data/角色',is_reg:false,images:3,repeats,repeated_images:3*repeats,resolution_variants:1,items:3*repeats},
  {source_index:1,path:'/data/正则',is_reg:true,images:2,repeats:1,repeated_images:2,resolution_variants:1,items:2},
]});

it('retains the last distribution, retries failures and ignores an older pending response',async()=>{
  const replies:Array<(response:Response)=>void>=[];
  server.use(http.post('/api/plan',()=>new Promise<Response>(resolve=>replies.push(resolve))));
  render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><MemoryRouter initialEntries={['/projects/p_test/train']}><Routes><Route path="/projects/:id/train" element={<TrainConfig/>}/></Routes></MemoryRouter></QueryClientProvider>);
  const epochs=within(await screen.findByTestId('field-loop.epochs')).getByRole('spinbutton');
  expect(screen.getByText('正在计算数据分布…')).toBeInTheDocument();
  await waitFor(()=>expect(replies).toHaveLength(1));
  await act(async()=>replies[0](HttpResponse.json(result(2))));
  expect(await screen.findByRole('button',{name:'角色 · 3 张 × 2 次 = 6 项 · 75.0%'})).toBeInTheDocument();
  fireEvent.change(epochs,{target:{value:'11'}});
  expect(await screen.findByText('正在更新数据分布，以下为上次结果…')).toBeInTheDocument();
  await waitFor(()=>expect(replies).toHaveLength(2));
  await act(async()=>replies[1](HttpResponse.json({detail:'Distribution request failed'},{status:503})));
  expect(await screen.findByText('数据分布计算失败')).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'开始训练'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'角色 · 3 张 × 2 次 = 6 项 · 75.0%'})).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'重新计算'}));
  await waitFor(()=>expect(replies).toHaveLength(3));
  fireEvent.change(epochs,{target:{value:'12'}});
  await waitFor(()=>expect(replies).toHaveLength(4));
  await act(async()=>replies[3](HttpResponse.json(result(6))));
  expect(await screen.findByRole('button',{name:'角色 · 3 张 × 6 次 = 18 项 · 90.0%'})).toBeInTheDocument();
  await act(async()=>replies[2](HttpResponse.json(result(1))));
  expect(screen.getByRole('button',{name:'角色 · 3 张 × 6 次 = 18 项 · 90.0%'})).toBeInTheDocument();
  expect(screen.queryByRole('button',{name:'角色 · 3 张 × 1 次 = 3 项 · 60.0%'})).not.toBeInTheDocument();
  expect(screen.queryByText('数据分布计算失败')).not.toBeInTheDocument();
  expect(screen.queryByText(/Distribution request failed/)).not.toBeInTheDocument();
});
