import {useEffect, useId, useRef, useState} from 'react';
import {Check, Copy, X} from 'lucide-react';
import type {Plan} from '../../api/types';
import {useWorkspaceText} from '../../utils/workspaceText';
import './source-balance.css';

const colors=['#3b82f6','#8b5cf6','#14b8a6','#f59e0b','#ec4899','#06b6d4','#84cc16','#f97316'];
const nameOf=(path:string,index:number)=>path.replace(/\\/g,'/').replace(/\/$/,'').split('/').pop() || `#${index+1}`;

export default function SourceBalance({sources,loading,hasSources=false}: {sources:Plan['source_balance'];loading:boolean;hasSources?:boolean}) {
  const text=useWorkspaceText();
  const [active,setActive]=useState<number|null>(null);
  const root = useRef<HTMLElement>(null);
  const detailsId = useId();
  const [copyResult, setCopyResult] = useState<{path:string;ok:boolean} | null>(null);
  useEffect(() => {
    if (active === null) return;
    const outside = (event: PointerEvent) => { if (event.target instanceof Node && !root.current?.contains(event.target)) setActive(null); };
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') { const trigger = root.current?.querySelector<HTMLButtonElement>('.source-balance-source.is-active'); setActive(null); trigger?.focus({preventScroll:true}); } };
    document.addEventListener('pointerdown', outside);
    document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('pointerdown', outside); document.removeEventListener('keydown', escape); };
  }, [active]);
  const known=Array.isArray(sources) && sources.every(source=>[source.images,source.repeats,source.repeated_images,source.resolution_variants,source.items].every(value=>Number.isFinite(value) && value>=0));
  const rows=known ? sources! : [];
  const total=rows.reduce((sum,source)=>sum+source.items,0);
  const ratio=(items:number)=>total>0 ? items/total*100 : 0;
  const formula=(source:typeof rows[number])=>text(`${source.images} 张 × ${source.repeats} 次${source.resolution_variants>1 ? ` × ${source.resolution_variants} 种尺寸` : ''} = ${source.items} 项`,`${source.images} images × ${source.repeats} repeats${source.resolution_variants>1 ? ` × ${source.resolution_variants} sizes` : ''} = ${source.items} items`);
  const selected=rows.find(source=>source.source_index===active);
  const copyPath = async (path: string) => {
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(path);
      else {
        const input = document.createElement('textarea');
        input.value = path; input.readOnly = true; input.style.cssText = 'position:fixed;left:-9999px;top:0';
        const focused = document.activeElement as HTMLElement | null;
        document.body.append(input);
        try { input.select(); if (!document.execCommand('copy')) throw new Error('copy failed'); }
        finally { input.remove(); focused?.focus({preventScroll:true}); }
      }
      setCopyResult({path,ok:true});
    } catch { setCopyResult({path,ok:false}); }
  };
  return <section ref={root} className="source-balance" aria-label={text('训练集配平','Dataset balance')}>
    <div className="source-balance-heading"><h4>{text('训练集配平','Dataset balance')}</h4>{known && <small>{text(`${total} 项 / 轮`,`${total} items / epoch`)}</small>}</div>
    {!known ? <p className="inspector-note">{loading ? text('统计各来源的图片与重复次数…','Counting source images and repeats…') : hasSources ? text('配平统计尚不可用，请完成数据配置后重新计算。','Balance is unavailable. Complete the dataset configuration and recalculate.') : text('添加训练数据后显示各来源占比。','Add training data to see each source’s share.')}</p> : total===0 ? <p className="inspector-note">{text('当前没有可参与训练的样本。','No samples currently participate in training.')}</p> : <>
      <div className="source-balance-bar" role="img" aria-label={rows.map(source=>`${nameOf(source.path,source.source_index)} ${ratio(source.items).toFixed(1)}%`).join(' · ')}>{rows.map(source=><span aria-hidden="true" key={source.source_index} style={{width:`${ratio(source.items)}%`,background:colors[source.source_index%colors.length]}} onMouseEnter={()=>setActive(source.source_index)}/>)}</div>
      {[false,true].map(isReg=>{
        const group=rows.filter(source=>source.is_reg===isReg);
        if (!group.length) return null;
        return <div className={`source-balance-group ${isReg?'is-regularization':''}`} key={String(isReg)}>
          <h5>{isReg?text('正则集','Regularization'):text('训练集','Training')}<span>{ratio(group.reduce((sum,source)=>sum+source.items,0)).toFixed(1)}%</span></h5>
          {group.map(source=><button type="button" className={`source-balance-source ${active===source.source_index?'is-active':''}`} key={source.source_index} onMouseEnter={()=>setActive(source.source_index)} onClick={()=>setActive(source.source_index)} aria-expanded={active===source.source_index} aria-controls={active===source.source_index ? detailsId : undefined} aria-label={`${nameOf(source.path,source.source_index)} · ${formula(source)} · ${ratio(source.items).toFixed(1)}%`}>
            <span className="source-balance-swatch" style={{background:colors[source.source_index%colors.length]}}/><span className="source-balance-name">{nameOf(source.path,source.source_index)}</span><strong>{ratio(source.items).toFixed(1)}%</strong><small>{formula(source)}</small>
          </button>)}
        </div>;
      })}
      {selected && <div role="region" aria-label={text('数据源详情','Source details')} id={detailsId} className="source-balance-tooltip">
        <header><button type="button" className="ui-btn ui-btn-quiet ui-btn-sm" onClick={() => void copyPath(selected.path)}>{copyResult?.path === selected.path && copyResult.ok ? <Check size={13}/> : <Copy size={13}/>}<span aria-live="polite">{copyResult?.path === selected.path && copyResult.ok ? text('已复制','Copied') : text('复制路径','Copy path')}</span></button><button type="button" className="ui-btn ui-btn-quiet ui-btn-sm ui-btn-icon" aria-label={text('关闭数据源详情','Close source details')} onClick={() => setActive(null)}><X size={13}/></button></header>
        <strong className="source-balance-path">{selected.path}</strong><span>{formula(selected)} · {ratio(selected.items).toFixed(1)}%</span>
        {copyResult?.path === selected.path && !copyResult.ok && <p role="alert">{text('复制失败，请选中路径复制。','Copy failed. Select and copy the path.')}</p>}
      </div>}
    </>}
  </section>;
}
