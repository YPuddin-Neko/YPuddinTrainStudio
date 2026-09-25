import {useEffect, useRef, useState, type ReactNode} from 'react';
import StudioSelect from '../../components/StudioSelect';
import Switch from '../../components/Switch';

type Interval = {save_state_every_steps:number|null;save_state_every_epochs:number|null};
type Unit = 'steps'|'epochs';

function IntervalInput({value,onChange,...props}: {value:number;onChange:(value:number)=>void} & Omit<React.InputHTMLAttributes<HTMLInputElement>,'value'|'onChange'>) {
  const [draft,setDraft]=useState(String(value));
  useEffect(()=>setDraft(String(value)),[value]);
  return <input {...props} type="number" min={1} step={1} value={draft}
    onChange={event=>{
      const raw=event.target.value;setDraft(raw);
      if(raw!=='')onChange(Number(raw));
    }} onBlur={()=>setDraft(String(value))}/>;
}

export default function RecoveryInterval({id,label,value,onChange,english,invalid,help}: {
  id:string;label:string;value:Interval;onChange:(value:Interval)=>void;english:boolean;invalid?:boolean;help?:ReactNode;
}) {
  const steps=value.save_state_every_steps;
  const epochs=value.save_state_every_epochs;
  const enabled=steps!=null || epochs!=null;
  const remembered=useRef({steps:steps ?? 100,epochs:epochs ?? 1});
  const lastEnabled=useRef<Interval>(enabled ? value : {save_state_every_steps:100,save_state_every_epochs:null});
  const lastEdit=useRef<Interval|null>(null);
  useEffect(()=>{
    const ownEdit=lastEdit.current?.save_state_every_steps===steps && lastEdit.current?.save_state_every_epochs===epochs;
    lastEdit.current=null;
    if(!ownEdit)remembered.current={steps:steps ?? 100,epochs:epochs ?? 1};
    if(steps!=null || epochs!=null)lastEnabled.current={save_state_every_steps:steps,save_state_every_epochs:epochs};
    else if(!ownEdit)lastEnabled.current={save_state_every_steps:100,save_state_every_epochs:null};
  },[steps,epochs]);
  const emit=(next:Interval)=>{lastEdit.current=next;onChange(next);};
  const mode=steps!=null && epochs!=null ? 'both' : epochs!=null ? 'epochs' : 'steps';
  const input=(unit:Unit) => <IntervalInput id={unit===mode || mode==='both' && unit==='steps' ? id : undefined}
    aria-label={mode==='both' ? `${label} · ${unit==='steps'?'Step':'Epoch'}` : label}
    aria-invalid={invalid || undefined} aria-describedby={`${id}-hint`}
    value={(unit==='steps'?steps:epochs) ?? remembered.current[unit]}
    onChange={next=>{
      remembered.current[unit]=next;
      emit(unit==='steps' ? {save_state_every_steps:next,save_state_every_epochs:mode==='both'?epochs:null} : {save_state_every_steps:mode==='both'?steps:null,save_state_every_epochs:next});
    }}/>;
  return <>
    <div className="recovery-switch-row">
      <Switch checked={enabled} aria-controls={`${id}-settings`} onCheckedChange={checked=>emit(checked ? lastEnabled.current : {save_state_every_steps:null,save_state_every_epochs:null})}>
        {english?'Save recovery points periodically':'定期保存恢复点'}
      </Switch>{help}
    </div>
    {enabled && <div id={`${id}-settings`} className="recovery-settings">
      <div className={`recovery-interval${mode==='both'?' recovery-interval-both':''}`}>
        {mode==='both' ? <div className="recovery-interval-values"><label>Step{input('steps')}</label><label>Epoch{input('epochs')}</label></div> : input(mode)}
        <StudioSelect aria-label={english?'Recovery save unit':'恢复点保存单位'} value={mode}
          options={[{value:'steps',label:english?'Step':'Step (训练步)'},{value:'epochs',label:english?'Epoch':'Epoch (训练轮)'},...(mode==='both'?[{value:'both',label:english?'Step + Epoch':'Step + Epoch (两种间隔)',disabled:true}]:[])]}
          onValueChange={raw=>{
            const unit=raw as Unit;
            if(steps!=null)remembered.current.steps=steps;
            if(epochs!=null)remembered.current.epochs=epochs;
            emit(unit==='steps'?{save_state_every_steps:remembered.current.steps,save_state_every_epochs:null}:{save_state_every_steps:null,save_state_every_epochs:remembered.current.epochs});
          }}/>
      </div>
    </div>}
  </>;
}
