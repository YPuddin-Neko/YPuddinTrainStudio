import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import VisualCropEditor from '../src/components/datasets/VisualCropEditor';
import '../src/i18n';

it('draws a source-pixel rectangle, enforces a ratio and submits actual geometry', () => {
  const apply=vi.fn();
  render(<VisualCropEditor image={{dataset_id:'d_1',hash:'hash',width:80,height:40,rel_path:'portrait.png'}} busy={false} onApply={apply}/>);
  const svg=screen.getByRole('img',{name:'可拖动裁剪区域'});
  vi.spyOn(svg,'getBoundingClientRect').mockReturnValue({x:0,y:0,left:0,top:0,right:80,bottom:40,width:80,height:40,toJSON:()=>({})});
  fireEvent.pointerDown(svg,{button:0,pointerId:1,clientX:10,clientY:5});
  fireEvent.pointerMove(svg,{pointerId:1,clientX:50,clientY:30});
  fireEvent.pointerUp(svg,{pointerId:1});
  expect(screen.getByLabelText('裁剪 X')).toHaveValue(10);
  expect(screen.getByLabelText('裁剪 Y')).toHaveValue(5);
  expect(screen.getByLabelText('裁剪宽度')).toHaveValue(40);
  expect(screen.getByLabelText('裁剪高度')).toHaveValue(25);
  fireEvent.click(screen.getByRole('button',{name:'应用此裁剪'}));
  expect(apply).toHaveBeenLastCalledWith({x:10,y:5,width:40,height:25});
  fireEvent.change(screen.getByLabelText('裁剪比例'),{target:{value:'1'}});
  expect(screen.getByLabelText('裁剪宽度')).toHaveValue(25);
  expect(screen.getByLabelText('裁剪高度')).toHaveValue(25);
  fireEvent.click(screen.getByRole('button',{name:'重置区域'}));
  expect(screen.getByLabelText('裁剪宽度')).toHaveValue(80);
  expect(screen.getByLabelText('裁剪高度')).toHaveValue(40);
});

it('moves and resizes a rectangle while staying inside the source',()=>{
  const apply=vi.fn();
  const {container}=render(<VisualCropEditor image={{dataset_id:'d_1',hash:'hash',width:80,height:40,rel_path:'portrait.png'}} busy={false} onApply={apply}/>);
  const svg=screen.getByRole('img',{name:'可拖动裁剪区域'});
  vi.spyOn(svg,'getBoundingClientRect').mockReturnValue({x:0,y:0,left:0,top:0,right:80,bottom:40,width:80,height:40,toJSON:()=>({})});
  fireEvent.change(screen.getByLabelText('裁剪宽度'),{target:{value:'30'}});
  fireEvent.change(screen.getByLabelText('裁剪高度'),{target:{value:'20'}});
  fireEvent.pointerDown(container.querySelector('[data-crop-handle="move"]')!,{button:0,pointerId:1,clientX:10,clientY:10});
  fireEvent.pointerMove(svg,{pointerId:1,clientX:75,clientY:35});
  fireEvent.pointerUp(svg,{pointerId:1});
  expect(screen.getByLabelText('裁剪 X')).toHaveValue(50);
  expect(screen.getByLabelText('裁剪 Y')).toHaveValue(20);
  fireEvent.pointerDown(container.querySelector('[data-crop-handle="nw"]')!,{button:0,pointerId:2,clientX:50,clientY:20});
  fireEvent.pointerMove(svg,{pointerId:2,clientX:40,clientY:10});
  fireEvent.pointerUp(svg,{pointerId:2});
  fireEvent.click(screen.getByRole('button',{name:'应用此裁剪'}));
  expect(apply).toHaveBeenLastCalledWith({x:40,y:10,width:40,height:30});
});
