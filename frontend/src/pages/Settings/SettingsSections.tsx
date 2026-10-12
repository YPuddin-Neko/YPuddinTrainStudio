import React from 'react';
import { useLocation } from 'react-router-dom';
import { useWorkspaceText } from '../../utils/workspaceText';

function isVisibleSection(section: Element) {
  if (section.closest('[hidden], [inert]')) return false;
  for (let node: Element | null = section; node; node = node.parentElement) {
    const style = window.getComputedStyle(node);
    if (style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse') return false;
  }
  return true;
}

export function SettingsSections({ sections, children }: { sections: { id: string; label: string }[]; children: React.ReactNode }) {
  const text = useWorkspaceText();
  const location = useLocation();
  const root = React.useRef<HTMLDivElement>(null);
  const endNavigation = React.useRef<{ target: HTMLElement; reachedEnd: boolean; fromHash: boolean } | null>(null);
  const [current, setCurrent] = React.useState(sections[0]?.id);
  const selectSection = React.useCallback((id: string, behavior: ScrollBehavior) => {
    const target = [...root.current?.querySelectorAll<HTMLElement>('[data-settings-section]') || []].find(section => section.id === id);
    if (!target || !isVisibleSection(target)) return false;
    const container = root.current?.closest<HTMLElement>('.settings-scroll');
    const offset = container ? target.getBoundingClientRect().top - container.getBoundingClientRect().top - (parseFloat(window.getComputedStyle(container).scrollPaddingTop) || 0) - (parseFloat(window.getComputedStyle(target).scrollMarginTop) || 0) : 0;
    endNavigation.current = container && container.scrollTop + offset > container.scrollHeight - container.clientHeight ? { target, reachedEnd: false, fromHash: behavior === 'auto' } : null;
    setCurrent(id);
    if (container) {
      if (behavior === 'auto') container.scrollTop += offset;
      else container.scrollTo({ top: container.scrollTop + offset, behavior });
    }
    target.focus({ preventScroll: true });
    return true;
  }, []);
  const sectionIds = sections.map(section => section.id).join('|');
  React.useEffect(() => {
    if (!location.hash) return;
    let id: string;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    if (!sectionIds.split('|').includes(id) || selectSection(id, 'auto')) return;
    const container = root.current;
    if (!container) return;
    const observer = new MutationObserver(() => { if (selectSection(id, 'auto')) observer.disconnect(); });
    observer.observe(container, { childList: true, subtree: true });
    const timer = window.setTimeout(() => observer.disconnect(), 5000);
    return () => { observer.disconnect(); window.clearTimeout(timer); };
  }, [location.hash, sectionIds, selectSection]);
  React.useEffect(() => {
    setCurrent(previous => sections.some(section => section.id === previous) ? previous : sections[0]?.id);
  }, [sections]);
  React.useEffect(() => {
    if (!window.IntersectionObserver) return;
    const ids = new Set(sections.map(section => section.id));
    const observed = new Set([...root.current?.querySelectorAll('[data-settings-section]') || []].filter(section => ids.has(section.id) && isVisibleSection(section)));
    const container = root.current?.closest<HTMLElement>('.settings-scroll');
    const intersecting = new Set<Element>();
    let active = true;
    const updateCurrent = () => {
      if (!active) return;
      const viewport = container?.getBoundingClientRect();
      const padding = container ? parseFloat(window.getComputedStyle(container).scrollPaddingTop) || 0 : 0;
      const alignmentTop = (section: Element) => (viewport?.top || 0) + padding + (parseFloat(window.getComputedStyle(section).scrollMarginTop) || 0);
      const navigation = endNavigation.current;
      if (navigation) {
        const target = navigation.target;
        const rect = target.getBoundingClientRect();
        if (container && observed.has(target) && isVisibleSection(target) && (navigation.fromHash || Math.ceil(container.scrollTop + container.clientHeight) >= container.scrollHeight) && rect.top > alignmentTop(target) && rect.top < viewport!.bottom) {
          navigation.reachedEnd = true;
          setCurrent(target.id);
          return;
        }
        if (navigation.reachedEnd || !observed.has(target) || !isVisibleSection(target)) endNavigation.current = null;
      }
      const visible = [...intersecting].filter(isVisibleSection).map(target => ({ target, rect: target.getBoundingClientRect() }))
        .filter(({ target, rect }) => rect.bottom > alignmentTop(target) && rect.top < (viewport?.bottom ?? window.innerHeight))
        .sort((a, b) => a.rect.top - b.rect.top);
      if (visible[0]) setCurrent(visible[0].target.id);
    };
    const observer = new IntersectionObserver(entries => {
      if (!active) return;
      for (const entry of entries) {
        if (entry.isIntersecting && observed.has(entry.target)) intersecting.add(entry.target);
        else intersecting.delete(entry.target);
      }
      updateCurrent();
    }, { root: container, rootMargin: '0px 0px -65% 0px', threshold: 0 });
    observed.forEach(section => observer.observe(section));
    const scrollTarget = container || window;
    const clearEndNavigation = () => { endNavigation.current = null; };
    const inputEvents = ['wheel', 'touchstart', 'pointerdown', 'keydown'];
    scrollTarget.addEventListener('scroll', updateCurrent, { passive: true });
    inputEvents.forEach(event => scrollTarget.addEventListener(event, clearEndNavigation, { passive: true, capture: true }));
    return () => {
      active = false;
      observer.disconnect();
      scrollTarget.removeEventListener('scroll', updateCurrent);
      inputEvents.forEach(event => scrollTarget.removeEventListener(event, clearEndNavigation, true));
    };
  }, [sections]);
  return <div ref={root} className="settings-columns">
    <div className="settings-main">{children}</div>
    <nav className="settings-index" aria-label={text('当前页章节', 'Page sections')}>
      <p>{text('本页内容', 'On this page')}</p>
      {sections.map(section => <button key={section.id} type="button" aria-current={current === section.id ? 'location' : undefined} onClick={() => selectSection(section.id, 'smooth')}>{section.label}</button>)}
    </nav>
  </div>;
}
