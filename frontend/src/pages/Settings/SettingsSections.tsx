import React from 'react';
import { useWorkspaceText } from '../../utils/workspaceText';

export function SettingsSections({ sections, children }: { sections: { id: string; label: string }[]; children: React.ReactNode }) {
  const text = useWorkspaceText();
  const root = React.useRef<HTMLDivElement>(null);
  const [current, setCurrent] = React.useState(sections[0]?.id);
  const first = sections[0]?.id;
  React.useEffect(() => { setCurrent(first); }, [first]);
  React.useEffect(() => {
    if (!window.IntersectionObserver) return;
    const container = root.current?.closest('.settings-scroll');
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible[0]) setCurrent(visible[0].target.id);
    }, { root: container, rootMargin: '0px 0px -65% 0px', threshold: 0 });
    root.current?.querySelectorAll('[data-settings-section]').forEach(section => observer.observe(section));
    return () => observer.disconnect();
  }, [sections]);
  return <div ref={root} className="settings-columns">
    <div className="settings-main">{children}</div>
    <nav className="settings-index" aria-label={text('当前页章节', 'Page sections')}>
      <p>{text('本页内容', 'On this page')}</p>
      {sections.map(section => <button key={section.id} type="button" aria-current={current === section.id ? 'location' : undefined} onClick={() => {
        setCurrent(section.id);
        const target = root.current?.querySelector<HTMLElement>(`[id="${section.id}"]`);
        target?.scrollIntoView?.({ behavior: 'smooth', block: 'start' });
        target?.focus({ preventScroll: true });
      }}>{section.label}</button>)}
    </nav>
  </div>;
}
