import React from 'react';
import { createPortal } from 'react-dom';
import { useInRouterContext, useLocation } from 'react-router-dom';
import { HelpCircle } from 'lucide-react';
import './config-help.css';

const OPEN_EVENT = 'studio:config-help-open';
const INSET = 8;
const GAP = 6;

function RouteDismiss({ close }: { close: () => void }) {
  const location = useLocation();
  React.useEffect(close, [close, location.key, location.pathname, location.search, location.hash]);
  return null;
}

/** Readable within the page viewport, independent of a field's clipping ancestors. */
/**
 * `anchor` names an enclosing element (a stat card) that the explanation opens below and starts at, so it does not
 * cover what that element shows; without it the explanation opens below the button.
 */
export default function ConfigHelp({ label, children, anchor: anchorSelector }: { label: string; children: string; anchor?: string }) {
  const id = React.useId();
  const trigger = React.useRef<HTMLButtonElement>(null);
  const panel = React.useRef<HTMLDivElement>(null);
  const [open, setOpen] = React.useState(false);
  const [position, setPosition] = React.useState<React.CSSProperties>({ visibility: 'hidden' });
  const inRouter = useInRouterContext();
  const close = React.useCallback(() => setOpen(false), []);

  React.useLayoutEffect(() => {
    if (!open) return;
    const button = trigger.current;
    const popup = panel.current;
    if (!button || !popup) return;

    const place = () => {
      const viewport = window.visualViewport;
      const bounds = {
        left: viewport?.offsetLeft ?? 0,
        top: viewport?.offsetTop ?? 0,
        right: (viewport?.offsetLeft ?? 0) + (viewport?.width ?? window.innerWidth),
        bottom: (viewport?.offsetTop ?? 0) + (viewport?.height ?? window.innerHeight),
      };
      // Keep the explanation in its main/scrolling workspace, not over the sidebar.
      // Cosmetic overflow:hidden cards are intentionally not boundaries: the portal
      // exists so their rounded corners cannot cut off the explanation.
      for (let parent = button.parentElement; parent && parent !== document.body; parent = parent.parentElement) {
        const style = window.getComputedStyle(parent);
        if (!parent.matches('main, [role="main"]') && !/(auto|scroll)/.test(`${style.overflow} ${style.overflowX} ${style.overflowY}`)) continue;
        const rect = parent.getBoundingClientRect();
        if (!rect.width || !rect.height) continue;
        bounds.left = Math.max(bounds.left, rect.left + parent.clientLeft);
        bounds.top = Math.max(bounds.top, rect.top + parent.clientTop);
        bounds.right = Math.min(bounds.right, parent.clientWidth ? rect.left + parent.clientLeft + parent.clientWidth : rect.right);
        bounds.bottom = Math.min(bounds.bottom, parent.clientHeight ? rect.top + parent.clientTop + parent.clientHeight : rect.bottom);
      }
      const anchor = ((anchorSelector && button.closest(anchorSelector)) || button).getBoundingClientRect();
      if (anchor.bottom < bounds.top || anchor.top > bounds.bottom || anchor.right < bounds.left || anchor.left > bounds.right) {
        close();
        return;
      }
      const width = Math.max(1, Math.min(320, bounds.right - bounds.left - 2 * INSET));
      // Set the constrained width before measuring wrapped text, then publish both
      // coordinates together in layout effect so no unpositioned frame is painted.
      popup.style.width = `${width}px`;
      const wantedHeight = Math.min(320, popup.scrollHeight || popup.getBoundingClientRect().height);
      const below = Math.max(0, bounds.bottom - INSET - anchor.bottom - GAP);
      const above = Math.max(0, anchor.top - GAP - bounds.top - INSET);
      const upwards = below < wantedHeight && above > below;
      const maxHeight = Math.max(1, Math.min(320, upwards ? above : below));
      const height = Math.min(wantedHeight, maxHeight);
      const top = upwards ? anchor.top - GAP - height : anchor.bottom + GAP;
      setPosition({
        width,
        maxHeight,
        left: Math.max(bounds.left + INSET, Math.min(anchorSelector ? anchor.left : anchor.right - width, bounds.right - INSET - width)),
        top: Math.max(bounds.top + INSET, Math.min(top, bounds.bottom - INSET - height)),
        visibility: 'visible',
      });
    };
    const outside = (event: Event) => {
      const target = event.target;
      if (target instanceof Node && !button.contains(target) && !popup.contains(target)) close();
    };
    const scrolled = (event: Event) => {
      // Scrolling a long explanation must remain possible without dismissing it.
      if (!(event.target instanceof Node && popup.contains(event.target))) close();
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      event.stopPropagation();
      close();
      if (popup.contains(document.activeElement)) button.focus({ preventScroll: true });
    };
    place();
    document.addEventListener(OPEN_EVENT, close);
    document.addEventListener('pointerdown', outside, true);
    document.addEventListener('focusin', outside, true);
    document.addEventListener('keydown', escape, true);
    window.addEventListener('scroll', scrolled, true);
    window.addEventListener('resize', place);
    window.visualViewport?.addEventListener('resize', place);
    window.visualViewport?.addEventListener('scroll', scrolled);
    return () => {
      document.removeEventListener(OPEN_EVENT, close);
      document.removeEventListener('pointerdown', outside, true);
      document.removeEventListener('focusin', outside, true);
      document.removeEventListener('keydown', escape, true);
      window.removeEventListener('scroll', scrolled, true);
      window.removeEventListener('resize', place);
      window.visualViewport?.removeEventListener('resize', place);
      window.visualViewport?.removeEventListener('scroll', scrolled);
    };
  }, [open, close, children, anchorSelector]);

  return <>
    {inRouter && <RouteDismiss close={close} />}
    <button ref={trigger} type="button" className="config-help-trigger" aria-label={label}
      aria-expanded={open} aria-controls={open ? id : undefined} aria-describedby={open ? id : undefined}
      onClick={event => {
        event.currentTarget.focus({ preventScroll: true });
        if (open) { close(); return; }
        document.dispatchEvent(new Event(OPEN_EVENT));
        setOpen(true);
      }}>
      <HelpCircle size={14} aria-hidden="true" />
    </button>
    {open && createPortal(<div ref={panel} id={id} role="tooltip" className="config-help-popover" style={position}>
      {children}
    </div>, document.body)}
  </>;
}
