import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import StructuredCaptionEditor from '../../../frontend/src/components/datasets/StructuredCaptionEditor';
import { captionFieldChanges, captionMetadata, type CaptionStructure, type CaptionFieldDraft } from '../../../frontend/src/utils/captionStructure';
import i18n from '../../../frontend/src/i18n';

function fixture(): CaptionStructure {
  return {format:'full',editable:true,legacy_override:false,revision:'source-sha',document:{fixed:{quality:'quality, detail',score:0.8},ai_output:{appearance:['long hair','red, blue dress'],nl:'A sentence, with punctuation.'},meta:{seed:42,checked:true,notes:{extra:['keep',3]}}},fields:[
    {path:['fixed','quality'],role:'quality',value:'quality, detail',present:true},
    {path:['ai_output','appearance'],role:'appearance',value:['long hair','red, blue dress'],present:true},
    {path:['ai_output','nl'],role:'nl',value:'A sentence, with punctuation.',present:true},
    {path:['ai_output','environment'],role:'environment',value:[],present:false},
  ]};
}
function Editor({structure=fixture(),readOnly=false}:{structure?:CaptionStructure;readOnly?:boolean}) {
  const [draft,setDraft]=React.useState<CaptionFieldDraft>({});
  return <><StructuredCaptionEditor structure={structure} draft={draft} onChange={setDraft} readOnly={readOnly}/><output data-testid="changes">{JSON.stringify(captionFieldChanges(structure,draft))}</output></>;
}
const changes=()=>JSON.parse(screen.getByTestId('changes').textContent!);
beforeEach(async()=>{await i18n.changeLanguage('zh-CN');});
describe('structured JSON caption editing',()=>{
  it('changes only field paths, preserves array members with commas, and retains original scalar types',()=>{
    const structure=fixture(); const original=JSON.stringify(structure);
    render(<Editor structure={structure}/>);
    fireEvent.change(screen.getByRole('textbox',{name:'画面质量'}),{target:{value:'new quality, details'}});
    fireEvent.change(screen.getByRole('textbox',{name:'外观与服装 · 标签 1'}),{target:{value:'short hair'}});
    fireEvent.change(screen.getByRole('textbox',{name:'自然语言描述'}),{target:{value:''}});
    expect(changes()).toEqual([{path:['fixed','quality'],value:'new quality, details'},{path:['ai_output','appearance'],value:['short hair','red, blue dress']},{path:['ai_output','nl'],value:''}]);
    expect(JSON.stringify(structure)).toBe(original);
    expect(screen.queryByRole('textbox',{name:'标签文本'})).not.toBeInTheDocument();
  });
  it('adds into an empty category, removes only that array entry, and drops reverted changes',()=>{
    render(<Editor/>);
    fireEvent.click(screen.getByRole('button',{name:'添加环境与背景标签'}));
    fireEvent.change(screen.getByRole('textbox',{name:'环境与背景 · 标签 1'}),{target:{value:'forest'}});
    fireEvent.click(screen.getByRole('button',{name:'删除外观与服装标签 1'}));
    expect(changes()).toEqual([{path:['ai_output','appearance'],value:['red, blue dress']},{path:['ai_output','environment'],value:['forest']}]);
    fireEvent.click(screen.getByRole('button',{name:'删除环境与背景标签 1'}));
    expect(changes()).toEqual([{path:['ai_output','appearance'],value:['red, blue dress']}]);
  });
  it('exposes unknown metadata read-only without coercing numbers, booleans, nested arrays or unknown keys',()=>{
    const structure=fixture();render(<Editor structure={structure} readOnly/>);
    expect(captionMetadata(structure)).toEqual({fixed:{score:0.8},meta:{seed:42,checked:true,notes:{extra:['keep',3]}}});
    fireEvent.click(screen.getByText('其他元数据'));
    const details=screen.getByText('其他元数据').closest('details')!;
    expect(details).toHaveTextContent('"seed": 42');expect(details).toHaveTextContent('"checked": true');
    expect(within(details).queryByRole('textbox')).not.toBeInTheDocument();
    expect(screen.getByRole('textbox',{name:'画面质量'})).toBeDisabled();
    expect(screen.queryByRole('button',{name:/添加/})).not.toBeInTheDocument();
  });
  it('shows active legacy fields without implying historical categories are restored',()=>{
    const structure=fixture(); structure.format='legacy_override';structure.legacy_override=true;
    structure.document={tags:['active'],_ypuddin_caption_edit:1,fixed:{quality:'historical'}};
    structure.fields=[{path:['tags'],role:'tags',value:['active'],present:true}];
    render(<Editor structure={structure}/>);
    expect(screen.getByText(/编辑当前使用的标签/)).toBeInTheDocument();
    expect(screen.queryByRole('textbox',{name:'画面质量'})).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole('textbox',{name:'画面内容 · 标签 1'}),{target:{value:'updated'}});
    expect(changes()).toEqual([{path:['tags'],value:['updated']}]);
  });
});
