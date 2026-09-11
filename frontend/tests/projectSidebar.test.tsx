import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes, useLocation, useNavigate, useParams, type Location } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { handlers } from '../src/mocks/handlers';
import Layout from '../src/components/Layout';
import ProjectWorkspaceHeader from '../src/components/projects/ProjectWorkspaceHeader';
import SettingsDrawer from '../src/components/SettingsDrawer';
import type { ProjectVersion, VersionedProject } from '../src/utils/projectVersions';
import i18n from '../src/i18n';

vi.mock('../src/events/useEventStream', () => ({ useEventStream: () => {}, useEventStreamStatus: () => 'connected' }));
const server = setupServer(...handlers);
const versions = ['v1', 'v2', 'archived'].map((id, index) => ({id,project_id:'p_sidebar',name:id==='archived'?'Old experiment':id,status:'ready',archived:id==='archived',number:index+1,paths:{root:`/project/v${index+1}`},stats:{images:0,datasets:0,jobs:0,artifacts:0}})) as ProjectVersion[];
let flush=vi.fn<() => Promise<void>>();
beforeAll(() => server.listen({onUnhandledRequest:'error'}));
afterAll(() => server.close());
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN'); flush=vi.fn().mockResolvedValue(undefined);
  server.use(http.get('/api/settings',()=>HttpResponse.json({paths:{},server:{},ui:{theme:'light',language:'zh-CN'}})));
});
afterEach(() => {server.resetHandlers();vi.restoreAllMocks();});

function ProjectPage() {
  const {versionId='v1'}=useParams(); const location=useLocation(); const [draft,setDraft]=React.useState('');
  const active=new URLSearchParams(location.search).get('step')==='results'?'results':'data';
  const project={id:'p_sidebar',name:'Character workspace',active_version_id:versionId} as VersionedProject;
  return <div className="project-workspace" data-testid="workspace"><ProjectWorkspaceHeader project={project} versionId={versionId} versions={versions} current={versions.find(item=>item.id===versionId)} active={active} refresh={async()=>{}} beforeAction={flush}/><input aria-label="Local draft" value={draft} onChange={event=>setDraft(event.target.value)}/></div>;
}
function Router() {
  const location=useLocation();const navigate=useNavigate();const background=(location.state as {backgroundLocation?:Location}|null)?.backgroundLocation;
  return <><Routes location={background||location}><Route element={<Layout/>}><Route path="/projects/:id/v/:versionId" element={<ProjectPage/>}/><Route path="/queue" element={<p>Global queue</p>}/></Route></Routes>
    {background&&<SettingsDrawer onClose={()=>navigate(`${background.pathname}${background.search}`,{replace:true})}><p>Settings content</p></SettingsDrawer>}
    <output data-testid="route">{location.pathname}{location.search}</output></>;
}
function show(){render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><MemoryRouter initialEntries={['/projects/p_sidebar/v/v1']}><Router/></MemoryRouter></QueryClientProvider>);}

describe('project navigation belongs to the global sidebar',()=>{
  it('portals project identity, version actions and stages into the sidebar, keeps a single content heading, then cleans the slot when leaving',async()=>{
    show();const slot=await screen.findByTestId('project-sidebar-slot');
    expect(within(slot).getByRole('link',{name:/Character workspace/})).toBeInTheDocument();
    expect(within(slot).getByRole('combobox',{name:'项目版本'})).toHaveTextContent('v1');
    expect(within(slot).getByRole('navigation',{name:'项目训练步骤'})).toBeInTheDocument();
    expect(within(slot).getByRole('button',{name:'新版本'})).toBeInTheDocument();
    const frame=screen.getByTestId('app-page-frame');expect(within(frame).getAllByRole('heading',{level:1})).toHaveLength(1);
    expect(within(frame).getByRole('heading',{name:'训练数据'})).toBeInTheDocument();
    expect(within(frame).queryByRole('combobox',{name:'项目版本'})).not.toBeInTheDocument();
    expect(within(frame).queryByRole('navigation',{name:'项目训练步骤'})).not.toBeInTheDocument();
    fireEvent.click(within(screen.getByRole('navigation',{name:'主导航'})).getByRole('link',{name:'任务队列'}));
    await screen.findByText('Global queue');expect(slot).toBeEmptyDOMElement();
  });

  it('waits for version saving, closes the mobile menu only after success, and exposes archived versions in the selector area',async()=>{
    let complete=()=>{};flush.mockImplementation(()=>new Promise<void>(resolve=>{complete=resolve;}));
    show();fireEvent.click(screen.getByRole('button',{name:i18n.t('hardware.openMenu')}));
    const sidebar=screen.getByRole('complementary');expect(sidebar).not.toHaveClass('hidden');
    fireEvent.click(screen.getByRole('combobox',{name:'项目版本'}));
    expect(sidebar).not.toHaveClass('hidden');
    fireEvent.click(screen.getByRole('option',{name:'v2'}));
    await waitFor(()=>expect(flush).toHaveBeenCalledOnce());
    expect(screen.getByTestId('route')).toHaveTextContent('/projects/p_sidebar/v/v1');
    expect(sidebar).not.toHaveClass('hidden');
    expect(screen.getByRole('combobox',{name:'项目版本'})).toBeDisabled();
    await act(async()=>complete());
    await waitFor(()=>expect(screen.getByTestId('route')).toHaveTextContent('/projects/p_sidebar/v/v2'));
    expect(sidebar).toHaveClass('hidden');
    fireEvent.click(screen.getByRole('checkbox',{name:'显示已归档版本'}));
    fireEvent.click(screen.getByRole('combobox',{name:'项目版本'}));
    expect(screen.getByRole('option',{name:'Old experiment · 已归档'})).toBeInTheDocument();
  });

  it('keeps the same sidebar and local draft through a settings background drawer',async()=>{
    show();const slot=screen.getByTestId('project-sidebar-slot');const controls=within(slot).getByRole('region',{name:'当前项目工作区'});
    const input=screen.getByRole('textbox',{name:'Local draft'});fireEvent.change(input,{target:{value:'Unsaved local value'}});
    fireEvent.click(within(screen.getByRole('navigation',{name:'主导航'})).getByRole('link',{name:'系统设置'}));
    const drawer=await screen.findByRole('dialog',{name:'系统设置'});
    expect(slot).toContainElement(controls);expect(input).toHaveValue('Unsaved local value');
    fireEvent.click(within(drawer).getByRole('button',{name:'关闭设置，返回工作区'}));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();expect(slot).toContainElement(controls);
    expect(screen.getByRole('textbox',{name:'Local draft'})).toBe(input);expect(flush).not.toHaveBeenCalled();
  });
});
