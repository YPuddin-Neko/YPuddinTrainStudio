import React from 'react';
import { X } from 'lucide-react';
import { useWorkspaceText } from '../utils/workspaceText';
import '../styles/settings-drawer.css';
import { useAnimatedClose } from './useAnimatedClose';
import { LoadingNote } from './Loading';

export default function SettingsDrawer({ children, onClose }: {children: React.ReactNode; onClose: () => void}) {
  const text = useWorkspaceText();
  const ref = React.useRef<HTMLElement>(null);
  const {closing,requestClose}=useAnimatedClose(onClose);
  const closeRef = React.useRef(requestClose); closeRef.current = requestClose;
  React.useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const panel = ref.current;
    panel?.querySelector<HTMLElement>('button')?.focus();
    const keydown = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement).closest('[role="dialog"]') !== panel) return;
      if (event.key === 'Escape') {
        // A select consumes the first Escape to dismiss its portalled list.
        if ((event.target as HTMLElement).closest('[role="combobox"][aria-expanded="true"]')) return;
        event.preventDefault(); closeRef.current();
      }
      if (event.key !== 'Tab') return;
      const nodes = [...(panel?.querySelectorAll<HTMLElement>('a[href],button,input,select,textarea,[tabindex]') || [])].filter(node => {
        if (node.tabIndex < 0 || node.matches(':disabled') || node.closest('[hidden], [inert]') || !node.getClientRects().length) return false;
        const style = window.getComputedStyle(node);
        return style.visibility !== 'hidden' && style.visibility !== 'collapse';
      });
      const first = nodes[0], last = nodes[nodes.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    panel?.addEventListener('keydown',keydown);
    return () => {panel?.removeEventListener('keydown',keydown);previous?.focus();};
  }, []);
  return <div className={`settings-drawer-backdrop${closing ? ' is-closing' : ''}`} onMouseDown={event => {if(event.target === event.currentTarget)requestClose();}}><section ref={ref} role="dialog" aria-modal="true" aria-label={text('系统设置','System settings')} className="settings-drawer"><button type="button" className="ui-btn ui-btn-quiet ui-btn-icon settings-drawer-close" aria-label={text('关闭设置，返回工作区','Close settings and return to workspace')} onClick={requestClose}><X size={19}/></button><React.Suspense fallback={<LoadingNote block className="settings-drawer-loading" label={text('正在读取设置…','Loading settings…')}/>}>{children}</React.Suspense></section></div>;
}
