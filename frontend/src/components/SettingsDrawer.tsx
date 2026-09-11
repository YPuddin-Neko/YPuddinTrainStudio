import React from 'react';
import { X } from 'lucide-react';
import { useWorkspaceText } from '../utils/workspaceText';
import '../styles/settings-drawer.css';

export default function SettingsDrawer({ children, onClose }: {children: React.ReactNode; onClose: () => void}) {
  const text = useWorkspaceText();
  const ref = React.useRef<HTMLElement>(null);
  const closeRef = React.useRef(onClose); closeRef.current = onClose;
  React.useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const panel = ref.current;
    panel?.querySelector<HTMLElement>('button')?.focus();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); closeRef.current(); }
      if (event.key !== 'Tab') return;
      const nodes = [...(panel?.querySelectorAll<HTMLElement>('a[href],button:not(:disabled),input:not(:disabled),select:not(:disabled),textarea:not(:disabled),[tabindex="0"]') || [])];
      const first = nodes[0], last = nodes[nodes.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    panel?.addEventListener('keydown',keydown);
    return () => {panel?.removeEventListener('keydown',keydown);previous?.focus();};
  }, []);
  return <div className="settings-drawer-backdrop" onMouseDown={event => {if(event.target === event.currentTarget)onClose();}}><section ref={ref} role="dialog" aria-modal="true" aria-label={text('系统设置','System settings')} className="settings-drawer"><button className="settings-drawer-close" aria-label={text('关闭设置，返回工作区','Close settings and return to workspace')} onClick={onClose}><X size={19}/></button><React.Suspense fallback={<div role="status" className="settings-drawer-loading">{text('正在读取设置…','Loading settings…')}</div>}>{children}</React.Suspense></section></div>;
}
