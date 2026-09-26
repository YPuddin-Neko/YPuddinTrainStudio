import React from 'react';
import { Check, Copy } from 'lucide-react';
import { copyText } from '../utils/clipboard';
import { useWorkspaceText } from '../utils/workspaceText';

/** An icon button that copies a value and confirms in its tooltip. */
export default function CopyButton({ value, label }: { value: string; label: string }) {
  const text = useWorkspaceText();
  const [state, setState] = React.useState<'' | 'done' | 'failed'>('');
  const timer = React.useRef<number>();
  React.useEffect(() => () => window.clearTimeout(timer.current), []);
  const copy = async () => {
    try { await copyText(value); setState('done'); } catch { setState('failed'); }
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setState(''), 1500);
  };
  return <button type="button" className="ui-btn ui-btn-icon ui-btn-sm ui-btn-quiet" onClick={() => void copy()} aria-label={label}
    title={state === 'done' ? text('已复制', 'Copied') : state === 'failed' ? text('复制失败', 'Copy failed') : label}>{state === 'done' ? <Check size={14}/> : <Copy size={14}/>}</button>;
}
