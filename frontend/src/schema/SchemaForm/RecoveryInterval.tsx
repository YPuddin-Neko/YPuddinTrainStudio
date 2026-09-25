import {useEffect, useRef, useState} from 'react';
import StudioSelect from '../../components/StudioSelect';

type Interval = {save_state_every_steps:number|null;save_state_every_epochs:number|null};
type Unit = 'steps'|'epochs';

export default function RecoveryInterval({id,label,value,onChange,english,invalid}: {
  id:string;label:string;value:Interval;onChange:(value:Interval)=>void;english:boolean;invalid?:boolean;
}) {
  const [emptyUnit,setEmptyUnit] = useState<Unit>(value.save_state_every_epochs != null ? 'epochs' : 'steps');
  const remembered = useRef({steps:value.save_state_every_steps ?? 100,epochs:value.save_state_every_epochs ?? 1});
  const steps=value.save_state_every_steps;
  const epochs=value.save_state_every_epochs;
  const lastEdit=useRef<Interval|null>(null);
  useEffect(()=>{
    if(lastEdit.current?.save_state_every_steps===steps && lastEdit.current?.save_state_every_epochs===epochs) {
      lastEdit.current=null; return;
    }
    remembered.current={steps:steps ?? 100,epochs:epochs ?? 1};
    if(steps==null && epochs==null)setEmptyUnit('steps');
  },[steps,epochs]);
  const emit=(next:Interval)=>{lastEdit.current=next;onChange(next);};
  const mode=steps!=null && epochs!=null ? 'both' : epochs!=null ? 'epochs' : steps!=null ? 'steps' : emptyUnit;
  const update = (unit:Unit, raw:string) => {
    const next=raw==='' ? null : Number(raw);
    if(next!=null) remembered.current[unit]=next;
    setEmptyUnit(unit);
    emit(unit==='steps' ? {save_state_every_steps:next,save_state_every_epochs:mode==='both'?epochs:null} : {save_state_every_steps:mode==='both'?steps:null,save_state_every_epochs:next});
  };
  const input=(unit:Unit) => <input id={unit===mode || mode==='both' && unit==='steps' ? id : undefined} type="number" min={1} step={1}
    aria-label={mode==='both' ? `${label} · ${unit==='steps'?'Step':'Epoch'}` : label}
    aria-invalid={invalid || undefined} aria-describedby={`${id}-hint`}
    placeholder={english?'Disabled':'关闭'} value={(unit==='steps'?steps:epochs) ?? ''} onChange={event=>update(unit,event.target.value)}/>;
  return <div className={`recovery-interval${mode==='both'?' recovery-interval-both':''}`}>
    {mode==='both' ? <div className="recovery-interval-values"><label>Step{input('steps')}</label><label>Epoch{input('epochs')}</label></div> : input(mode)}
    <StudioSelect aria-label={english?'Recovery save unit':'恢复点保存单位'} value={mode}
      options={[{value:'steps',label:english?'Step':'Step (训练步)'},{value:'epochs',label:english?'Epoch':'Epoch (训练轮)'},...(mode==='both'?[{value:'both',label:english?'Step + Epoch':'Step + Epoch (两种间隔)',disabled:true}]:[])]}
      onValueChange={raw=>{
        const unit=raw as Unit;
        if(steps!=null)remembered.current.steps=steps;
        if(epochs!=null)remembered.current.epochs=epochs;
        setEmptyUnit(unit);
        emit(unit==='steps'?{save_state_every_steps:remembered.current.steps,save_state_every_epochs:null}:{save_state_every_steps:null,save_state_every_epochs:remembered.current.epochs});
      }}/>
  </div>;
}
