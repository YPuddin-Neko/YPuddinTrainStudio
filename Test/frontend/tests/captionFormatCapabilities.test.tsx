import {fireEvent, render, screen} from '@testing-library/react';
import {MemoryRouter} from 'react-router-dom';
import {beforeEach, describe, expect, it, vi} from 'vitest';
import CaptionFormatSelect from '../../../frontend/src/components/CaptionFormatSelect';
import ProjectDataImport from '../../../frontend/src/pages/ProjectDetail/ProjectDataImport';
import {SchemaForm} from '../../../frontend/src/schema/SchemaForm/SchemaForm';
import schema from '../../../frontend/src/schema/train-schema.json';
import type {FamilyInfo} from '../../../frontend/src/api/types';
import i18n from '../../../frontend/src/i18n';

beforeEach(async()=>{await i18n.changeLanguage('zh-CN');});
describe('family caption format capabilities',()=>{
  it.each(['anima','sdxl','krea2','flux2'])('keeps structured JSON available for %s',name=>{
    render(<SchemaForm schema={schema} value={{model:{family:name},dataset:{sources:[{path:'/data/portraits',caption_ext:'auto'}]}}} onChange={vi.fn()} groupFilter={['caption']} family={{name,caption_formats:['txt','json']} as unknown as FamilyInfo}/>);
    fireEvent.click(screen.getByRole('combobox',{name:'标签格式 · portraits'}));
    expect(screen.getByRole('option',{name:'仅 JSON'})).toBeVisible();
    expect(screen.getByRole('option',{name:'自动 · JSON 优先，其次 TXT'})).toBeVisible();
  });

  it('hides unsupported JSON without rewriting a legacy selection and offers explicit repair',()=>{
    const onChange=vi.fn();
    render(<CaptionFormatSelect value=".JSON" formats={['txt']} onChange={onChange}/>);
    expect(screen.getByRole('alert')).toHaveTextContent('当前模型不支持 JSON 标签');
    expect(screen.getByRole('combobox')).toHaveAttribute('aria-invalid','true');
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('combobox'));
    expect(screen.queryByRole('option',{name:'仅 JSON'})).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('option',{name:'自动 · 仅查找 TXT'}));
    expect(onChange).toHaveBeenCalledWith('auto');
  });

  it('uses the same capabilities in the training form and import picker',()=>{
    const {unmount}=render(<SchemaForm schema={schema} value={{model:{family:'toy'},dataset:{sources:[{path:'/data/portraits',caption_ext:'auto'}]}}} onChange={vi.fn()} groupFilter={['caption']} family={{name:'toy',caption_formats:['txt']} as unknown as FamilyInfo}/>);
    expect(screen.getByRole('combobox',{name:'标签格式 · portraits'})).toHaveTextContent('自动 · 仅查找 TXT');
    fireEvent.click(screen.getByRole('combobox',{name:'标签格式 · portraits'}));
    expect(screen.queryByRole('option',{name:'仅 JSON'})).not.toBeInTheDocument();
    unmount();
    render(<MemoryRouter><ProjectDataImport projectId="p" versionId="v" onImported={vi.fn()} captionFormats={['txt']}/></MemoryRouter>);
    fireEvent.click(screen.getByText('导入选项',{selector:'summary'}));
    fireEvent.click(screen.getByRole('combobox',{name:'标签格式'}));
    expect(screen.getByRole('option',{name:'自动 · 仅查找 TXT'})).toBeVisible();
    expect(screen.queryByRole('option',{name:'仅 JSON'})).not.toBeInTheDocument();
  });
});
