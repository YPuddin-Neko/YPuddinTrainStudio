import { useCallback, useEffect, useRef, useState } from 'react';
import Dialog from './Dialog';
import { useWorkspaceText } from '../utils/workspaceText';

type Confirmation = { title: string; message: string; confirmLabel?: string; danger?: boolean };

export function useConfirmation() {
  const text = useWorkspaceText();
  const [request, setRequest] = useState<Confirmation | null>(null);
  const pending = useRef<((confirmed: boolean) => void) | null>(null);
  const settle = useCallback((confirmed: boolean) => {
    const resolve = pending.current;
    pending.current = null;
    setRequest(null);
    resolve?.(confirmed);
  }, []);
  useEffect(() => () => {
    pending.current?.(false);
    pending.current = null;
  }, []);
  const confirm = useCallback((options: Confirmation): Promise<boolean> => {
    if (pending.current) return Promise.resolve(false);
    return new Promise(resolve => {
      pending.current = resolve;
      setRequest(options);
    });
  }, []);
  const confirmation = request && <Dialog title={request.title} nested onClose={() => settle(false)}>
    <p className="workspace-confirm-message">{request.message}</p>
    <footer className="workspace-confirm-actions">
      <button type="button" className="ui-btn" onClick={() => settle(false)}>{text('取消', 'Cancel')}</button>
      <button type="button" className={`ui-btn ${request.danger ? 'ui-btn-danger' : 'ui-btn-primary'}`} onClick={() => settle(true)}>{request.confirmLabel || text('确认', 'Confirm')}</button>
    </footer>
  </Dialog>;
  return {confirm, confirmation};
}
