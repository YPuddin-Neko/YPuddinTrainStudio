import {useEffect, useRef, useState, type ReactNode} from 'react';
import StudioSelect from '../../components/StudioSelect';
import ParameterToggleCard from './ParameterToggleCard';

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

export default function RecoveryInterval({id,label,value,onChange,english,invalid,help,error}: {
  id:string;label:string;value:Interval;onChange:(value:Interval)=>void;english:boolean;invalid?:boolean;help?:ReactNode;error?:string;
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
  const displayed=enabled ? value : lastEnabled.current;
  const displaySteps=displayed.save_state_every_steps;
  const displayEpochs=displayed.save_state_every_epochs;
  const mode=displaySteps!=null && displayEpochs!=null ? 'both' : displayEpochs!=null ? 'epochs' : 'steps';
  const hint=mode==='both'
    ? (english ? `Save recovery points every ${displaySteps} steps and every ${displayEpochs} epochs.` : `每 ${displaySteps} 步和每 ${displayEpochs} 轮分别保存恢复点。`)
    : mode==='epochs'
      ? (english ? `Save training state every ${displayEpochs} epochs.` : `每 ${displayEpochs} 轮保存一次完整训练状态。`)
      : (english ? `Save training state every ${displaySteps} steps.` : `每 ${displaySteps} 步保存一次完整训练状态。`);
  const input=(unit:Unit) => <IntervalInput id={unit===mode || mode==='both' && unit==='steps' ? id : undefined}
    aria-label={mode==='both' ? `${label} · ${unit==='steps'?'Step':'Epoch'}` : label}
    aria-invalid={invalid || undefined} aria-describedby={`${id}-hint`}
    value={(unit==='steps'?displaySteps:displayEpochs) ?? remembered.current[unit]}
    onChange={next=>{
      remembered.current[unit]=next;
      emit(unit==='steps' ? {save_state_every_steps:next,save_state_every_epochs:mode==='both'?epochs:null} : {save_state_every_steps:mode==='both'?steps:null,save_state_every_epochs:next});
    }}/>;
  return <ParameterToggleCard id={id} title={english?'Save recovery points periodically':'定期保存恢复点'}
    description={english?'Keep the complete training state so an interrupted run can resume.':'保存完整训练状态，供中断后继续。'}
    checked={enabled} help={help} onCheckedChange={checked=>emit(checked ? lastEnabled.current : {save_state_every_steps:null,save_state_every_epochs:null})}>
    <div className="recovery-settings">
      <label htmlFor={id}>{label}</label>
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
      <p id={`${id}-hint`} className="config-field-hint">{hint}</p>
      {error && <p className="config-field-error">{error}</p>}
    </div>
  </ParameterToggleCard>;
}
