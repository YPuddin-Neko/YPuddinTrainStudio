import {useState} from 'react';
import type {Plan} from '../../api/types';
import {useWorkspaceText} from '../../utils/workspaceText';
import './source-balance.css';

const colors=['#3b82f6','#8b5cf6','#14b8a6','#f59e0b','#ec4899','#06b6d4','#84cc16','#f97316'];
const nameOf=(path:string,index:number)=>path.replace(/\\/g,'/').replace(/\/$/,'').split('/').pop() || `#${index+1}`;

export default function SourceBalance({sources,loading,hasSources=false}: {sources:Plan['source_balance'];loading:boolean;hasSources?:boolean}) {
  const text=useWorkspaceText();
  const [active,setActive]=useState<number|null>(null);
  const known=Array.isArray(sources) && sources.every(source=>[source.images,source.repeats,source.repeated_images,source.resolution_variants,source.items].every(value=>Number.isFinite(value) && value>=0));
  const rows=known ? sources! : [];
  const total=rows.reduce((sum,source)=>sum+source.items,0);
  const ratio=(items:number)=>total>0 ? items/total*100 : 0;
  const formula=(source:typeof rows[number])=>text(`${source.images} 张 × ${source.repeats} 次${source.resolution_variants>1 ? ` × ${source.resolution_variants} 种尺寸` : ''} = ${source.items} 项`,`${source.images} images × ${source.repeats} repeats${source.resolution_variants>1 ? ` × ${source.resolution_variants} sizes` : ''} = ${source.items} items`);
  const selected=rows.find(source=>source.source_index===active);
  return <section className="source-balance" aria-label={text('训练集配平','Dataset balance')}>
    <div className="source-balance-heading"><h4>{text('训练集配平','Dataset balance')}</h4>{known && <small>{text(`${total} 项 / 轮`,`${total} items / epoch`)}</small>}</div>
    {!known ? <p className="inspector-note">{loading ? text('统计各来源的图片与重复次数…','Counting source images and repeats…') : hasSources ? text('配平统计尚不可用，请完成数据配置后重新计算。','Balance is unavailable. Complete the dataset configuration and recalculate.') : text('添加训练数据后显示各来源占比。','Add training data to see each source’s share.')}</p> : total===0 ? <p className="inspector-note">{text('当前没有可参与训练的样本。','No samples currently participate in training.')}</p> : <>
      <div className="source-balance-bar" role="img" aria-label={rows.map(source=>`${nameOf(source.path,source.source_index)} ${ratio(source.items).toFixed(1)}%`).join(' · ')}>{rows.map(source=><span aria-hidden="true" key={source.source_index} style={{width:`${ratio(source.items)}%`,background:colors[source.source_index%colors.length]}} title={`${source.path}\n${formula(source)} · ${ratio(source.items).toFixed(1)}%`} onMouseEnter={()=>setActive(source.source_index)} onMouseLeave={()=>setActive(null)}/>)}</div>
      {[false,true].map(isReg=>{
        const group=rows.filter(source=>source.is_reg===isReg);
        if (!group.length) return null;
        return <div className={`source-balance-group ${isReg?'is-regularization':''}`} key={String(isReg)}>
          <h5>{isReg?text('正则集','Regularization'):text('训练集','Training')}<span>{ratio(group.reduce((sum,source)=>sum+source.items,0)).toFixed(1)}%</span></h5>
          {group.map(source=><button type="button" className={`source-balance-source ${active===source.source_index?'is-active':''}`} key={source.source_index} onMouseEnter={()=>setActive(source.source_index)} onMouseLeave={()=>setActive(null)} onFocus={()=>setActive(source.source_index)} onBlur={()=>setActive(null)} onClick={()=>setActive(source.source_index)} aria-label={`${nameOf(source.path,source.source_index)} · ${formula(source)} · ${ratio(source.items).toFixed(1)}%`} title={`${source.path}\n${formula(source)} · ${ratio(source.items).toFixed(1)}%`}>
            <span className="source-balance-swatch" style={{background:colors[source.source_index%colors.length]}}/><span className="source-balance-name">{nameOf(source.path,source.source_index)}</span><strong>{ratio(source.items).toFixed(1)}%</strong><small>{formula(source)}</small>
          </button>)}
        </div>;
      })}
      {selected && <div role="tooltip" className="source-balance-tooltip"><strong>{selected.path}</strong><span>{formula(selected)} · {ratio(selected.items).toFixed(1)}%</span></div>}
      <p className="inspector-note">{text('已排除验证集，占比包含正则集。多卡训练末尾可能略过少量样本，此处未扣除；占比不代表损失权重。','Validation images are excluded. Shares include regularization and count items before multi-GPU tail dropping, not loss weights.')}</p>
    </>}
  </section>;
}
