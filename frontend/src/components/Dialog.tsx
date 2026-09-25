import React from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import './dialog.css';
import { useWorkspaceText } from '../utils/workspaceText';
import { useAnimatedClose } from './useAnimatedClose';

export default function Dialog({ title, onClose, children, wide = false, closeDisabled = false }: { title: string; onClose: () => void; children: React.ReactNode; wide?: boolean; closeDisabled?: boolean }) {
  const text = useWorkspaceText();
  const panel = React.useRef<HTMLDivElement>(null);
  const {closing,requestClose}=useAnimatedClose(onClose,closeDisabled);
  const closeRef = React.useRef(requestClose); closeRef.current = requestClose;
  const id = React.useId();
  React.useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const element = panel.current;
    element?.querySelector<HTMLElement>('input,select,button,textarea')?.focus();
    const keydown = (event: KeyboardEvent) => {
      // A nested path picker handles its own Escape and focus loop. This native
      // listener runs before React's delegated handler, so leave that event alone.
      if (!(event.target instanceof Element) || event.target.closest('[role="dialog"]') !== element) return;
      if (event.key === 'Escape' && (event.target as HTMLElement).closest('[role="combobox"][aria-expanded="true"]')) return;
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closeRef.current(); }
      if (event.key !== 'Tab') return;
      const nodes = [...(element?.querySelectorAll<HTMLElement>('button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled),textarea:not(:disabled),[tabindex="0"]') || [])];
      const first = nodes[0], last = nodes[nodes.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    element?.addEventListener('keydown', keydown);
    return () => { element?.removeEventListener('keydown', keydown); previous?.focus(); };
  }, []);
  return createPortal(<div className={`workspace-dialog-backdrop${closing ? ' is-closing' : ''}`} onMouseDown={event => { if (event.target === event.currentTarget) closeRef.current(); }}><div ref={panel} role="dialog" aria-modal="true" aria-labelledby={id} className={`workspace-dialog ${wide ? 'workspace-dialog-wide' : ''}`}><header><h2 id={id}>{title}</h2><button type="button" className="ui-btn ui-btn-quiet ui-btn-icon" disabled={closeDisabled} aria-label={text('关闭','Close')} onClick={() => closeRef.current()}><X size={18}/></button></header>{children}</div></div>, document.body);
}
