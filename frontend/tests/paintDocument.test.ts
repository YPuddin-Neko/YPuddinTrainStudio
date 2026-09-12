import { describe, expect, it } from 'vitest';
import { PaintDocument } from '../src/components/masks/paintDocument';
import { imagePoint } from '../src/components/masks/maskDocument';
const pixels=()=>new Uint8ClampedArray(Array.from({length:64},()=>[20,40,60,128]).flat());
describe('independent color painting document',()=>{
  it('paints original-pixel coordinates, preserves alpha and allows eyedropper and erase back to the original',()=>{
    const doc=new PaintDocument(8,8,pixels());
    const point=imagePoint(35,35,{left:0,top:0,width:80,height:80},8,8);
    const operation={kind:'stroke' as const,points:[point,{x:4.5,y:3.5}],diameter:1,color:'#ff0000',erase:false};
    doc.apply(operation);
    expect(doc.colorAt(point)).toBe('#ff0000');expect(Array.from(doc.pixels.slice((3*8+3)*4,(3*8+3)*4+4))).toEqual([255,0,0,128]);
    expect(doc.colorAt({x:0,y:0})).toBe('#14283c');
    doc.apply({...operation,points:[point],erase:true});expect(doc.colorAt(point)).toBe('#14283c');
    expect(doc.colorAt({x:4.5,y:3.5})).toBe('#ff0000');
    doc.undo();expect(doc.colorAt(point)).toBe('#ff0000');doc.redo();expect(doc.colorAt(point)).toBe('#14283c');
  });
  it('retains correct dirty state after saving, branching undo and clear/redo without retaining unbounded full-image snapshots',()=>{
    const doc=new PaintDocument(8,8,pixels());
    const operation={kind:'stroke' as const,points:[{x:3.5,y:3.5}],diameter:1,color:'#ff0000',erase:false};
    doc.apply(operation);doc.markSaved();expect(doc.dirty).toBe(false);
    doc.undo();expect(doc.dirty).toBe(true);doc.apply({...operation,color:'#00ff00'});expect(doc.canRedo).toBe(false);expect(doc.dirty).toBe(true);
    doc.apply({kind:'clear'});expect(doc.pixels).toEqual(pixels());doc.undo();expect(doc.colorAt(operation.points[0])).toBe('#00ff00');doc.redo();expect(doc.pixels).toEqual(pixels());
    expect(doc.base).toEqual(pixels());
  });
  it('rejects wrong image geometry and invalid colors and clips a large brush at the edges',()=>{
    expect(()=>new PaintDocument(2,2,pixels())).toThrow('dimensions');
    const doc=new PaintDocument(8,8,pixels());
    expect(()=>doc.segment({x:0,y:0},{x:1,y:1},2,'red',false)).toThrow('RGB');
    doc.apply({kind:'stroke',points:[{x:0,y:0}],diameter:100,color:'#abcdef',erase:false});
    expect(doc.colorAt({x:8,y:8})).toBe('#abcdef');expect(doc.pixels.filter((_,index)=>index%4===3).every(value=>value===128)).toBe(true);
  });
});
