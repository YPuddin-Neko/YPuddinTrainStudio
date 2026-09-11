import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes, useParams } from 'react-router-dom';
import Projects from '../src/pages/Projects/Projects';
import { apiClient } from '../src/api/client';
import i18n from '../src/i18n';
function Destination(){const {id}=useParams();return <output data-testid="created-id">{id}</output>;}
function show(){return render(<MemoryRouter initialEntries={['/projects']}><Routes><Route path="/projects" element={<Projects/>}/><Route path="/projects/:id" element={<Destination/>}/></Routes></MemoryRouter>);}
beforeEach(async()=>{
  await i18n.changeLanguage('zh-CN');
  vi.spyOn(apiClient,'get').mockResolvedValue([]);
  vi.spyOn(apiClient,'post').mockImplementation(async(_url,body)=>body as any);
});
afterEach(()=>vi.restoreAllMocks());
describe('project display name and permanent directory ID',()=>{
  it.each([{language:'zh-CN',name:'中文角色 Étoile',id:'Character_01'},{language:'en',name:'Character experiment',id:'English_2'}])('submits independent multilingual name and hand-entered ID in $language',async({language,name,id})=>{
    await i18n.changeLanguage(language);show();
    fireEvent.click(screen.getByRole('button',{name:i18n.t('projects.newProject')}));
    const modal=screen.getByTestId('create-project-modal');
    fireEvent.change(screen.getByTestId('project-name-input'),{target:{value:name}});
    expect(screen.getByTestId('project-id-input')).toHaveValue('');
    expect(within(modal).getByRole('button',{name:i18n.t('projects.create')})).toBeDisabled();
    fireEvent.change(screen.getByTestId('project-id-input'),{target:{value:id}});
    expect(within(modal).getByText(`studio_data/project/${id}/v1/`)).toBeInTheDocument();
    fireEvent.click(within(modal).getByRole('button',{name:i18n.t('projects.create')}));
    await waitFor(()=>expect(screen.getByTestId('created-id')).toHaveTextContent(id));
    expect(apiClient.post).toHaveBeenCalledWith('/projects',{id,name,note:''},{silent:true});
  });
  it('rejects unsupported ID characters without rewriting the input or sending a request',()=>{
    show();fireEvent.click(screen.getByRole('button',{name:'新建项目'}));
    fireEvent.change(screen.getByTestId('project-name-input'),{target:{value:'中文名称'}});
    for(const id of ['中文目录','wrong-name','has space','../escape']){
      fireEvent.change(screen.getByTestId('project-id-input'),{target:{value:id}});
      expect(screen.getByTestId('project-id-input')).toHaveValue(id);
      expect(screen.getByTestId('project-id-input')).toHaveAttribute('aria-invalid','true');
      expect(screen.getByRole('alert')).toHaveTextContent('只能包含英文字母、数字和下划线');
      expect(within(screen.getByTestId('create-project-modal')).getByRole('button',{name:'创建'})).toBeDisabled();
    }
    expect(apiClient.post).not.toHaveBeenCalled();
  });
  it('keeps both fields editable and reports server directory conflicts beside the ID',async()=>{
    vi.mocked(apiClient.post).mockRejectedValueOnce({message:'project already exists',details:{errors:[{loc:['body','id'],msg:'Character_01 conflicts with an existing directory'}]}});
    show();fireEvent.click(screen.getByRole('button',{name:'新建项目'}));
    fireEvent.change(screen.getByTestId('project-name-input'),{target:{value:'新的中文项目'}});
    fireEvent.change(screen.getByTestId('project-id-input'),{target:{value:'Character_01'}});
    fireEvent.click(within(screen.getByTestId('create-project-modal')).getByRole('button',{name:'创建'}));
    expect(await screen.findByRole('alert')).toHaveTextContent('body.id: Character_01 conflicts');
    expect(screen.getByTestId('project-name-input')).toHaveValue('新的中文项目');
    expect(screen.getByTestId('project-id-input')).toHaveValue('Character_01');
    fireEvent.change(screen.getByTestId('project-id-input'),{target:{value:'Character_02'}});
    fireEvent.click(within(screen.getByTestId('create-project-modal')).getByRole('button',{name:'创建'}));
    await screen.findByTestId('created-id');
    expect(apiClient.post).toHaveBeenLastCalledWith('/projects',{id:'Character_02',name:'新的中文项目',note:''},{silent:true});
  });
});
