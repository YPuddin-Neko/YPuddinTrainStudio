import {useEffect, useState} from 'react';
import {apiClient} from '../../api/client';
import type {components} from '../../api/generated';
import Dialog from '../../components/Dialog';
import {configFieldLabel} from '../../utils/configPresentation';
import {formatApiError} from '../../utils/errors';
import {useWorkspaceText} from '../../utils/workspaceText';

type Result = components['schemas']['ConfigInspection'];
const fieldKey = (field:Result['fields'][number]) => JSON.stringify(field.path);

export default function ConfigInspection({config,onApply,onClose}: {config:Record<string,any>;onApply:(config:Record<string,any>)=>void;onClose:()=>void}) {
  const text = useWorkspaceText();
  const encoded = JSON.stringify(config);
  const [result,setResult] = useState<{encoded:string;data:Result}|null>(null);
  const [selected,setSelected] = useState<Set<string>>(new Set());
  const [error,setError] = useState('');
  const [attempt,setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setResult(null); setError(''); setSelected(new Set());
    void apiClient.post<Result>('/config/inspect', {config:JSON.parse(encoded)}, {signal:controller.signal,silent:true}).then(data => {
      if (controller.signal.aborted) return;
      setResult({encoded,data});
      setSelected(new Set(data.fields.filter(field=>field.kind==='unknown').map(fieldKey)));
    }).catch(error => {if (!controller.signal.aborted) setError(formatApiError(error));});
    return () => controller.abort();
  }, [encoded,attempt]);
  const current = result?.encoded === encoded ? result.data : null;
  const apply = () => {
    if (!current || !selected.size) return;
    const next = structuredClone(config);
    for (const field of current.fields.filter(field=>selected.has(fieldKey(field)))) {
      let node:any = next;
      for (const key of field.path.slice(0,-1)) node = node?.[key];
      if (node && typeof node === 'object') delete node[field.path[field.path.length-1]];
    }
    onApply(next);
  };
  return <Dialog title={text('参数检查','Parameter check')} wide onClose={onClose}>
    <div className="config-inspection">
      {error ? <p role="alert">{error}<button type="button" className="ui-btn ui-btn-sm" onClick={()=>setAttempt(value=>value+1)}>{text('重试','Retry')}</button></p> : !current ? <p role="status">{text('正在检查参数…','Checking parameters…')}</p> : <>
        {current.fields.length ? <><p>{text('选择要移除的参数。暂不使用的参数可以保留，方便以后切换配置。','Select fields to remove. Inactive fields can be kept for future configuration changes.')}</p>
          <div className="config-inspection-list">{current.fields.map(field=>{const label=configFieldLabel(field.loc,field.loc,text('zh','en')==='en');return <label className="config-inspection-field" key={fieldKey(field)}>
            <input type="checkbox" checked={selected.has(fieldKey(field))} aria-label={text(`移除 ${field.loc}`,`Remove ${field.loc}`)} onChange={event=>setSelected(previous=>{const next=new Set(previous);if(event.target.checked)next.add(fieldKey(field));else next.delete(fieldKey(field));return next;})}/>
            <span>{label!==field.loc && <strong>{label}</strong>}<code>{field.loc}</code><small>{field.kind==='unknown'?text('当前版本不识别','Not recognized by this version'):text('当前条件下不使用','Inactive in this configuration')}</small><pre>{JSON.stringify(field.value,null,2)}</pre></span>
          </label>;})}</div></> : <p role="status">{text('没有可移除的参数。','No removable parameters.')}</p>}
        {!!current.errors.length && <div className="config-inspection-errors" role="alert"><strong>{text('以下参数需要修改取值','These parameter values need correction')}</strong>{current.errors.map((issue,index)=><p key={index}><code>{issue.loc}</code> {issue.msg}</p>)}</div>}
      </>}
      <footer><button type="button" className="ui-btn" onClick={onClose}>{text('关闭','Close')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={!current || !selected.size} onClick={apply}>{text(`移除所选参数${selected.size?` (${selected.size})`:''}`,`Remove selected fields${selected.size?` (${selected.size})`:''}`)}</button></footer>
    </div>
  </Dialog>;
}
