import React from 'react';
import { Archive, MoreHorizontal, Pencil, Trash2 } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';

export default function ProjectCardMenu({ name, archived, busy, onEdit, onArchive, onDelete }: {
  name: string; archived: boolean; busy: boolean; onEdit: () => void; onArchive: () => void; onDelete: () => void;
}) {
  const text = useWorkspaceText();
  const [open, setOpen] = React.useState(false);
  const root = React.useRef<HTMLDivElement>(null);
  const trigger = React.useRef<HTMLButtonElement>(null);
  const menu = React.useRef<HTMLDivElement>(null);
  const id = React.useId();
  React.useEffect(() => {
    if (!open) return;
    menu.current?.querySelector<HTMLButtonElement>('button')?.focus();
    const outside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [open]);
  const run = (action: () => void) => { setOpen(false); trigger.current?.focus(); action(); };
  return <div ref={root} className="project-card-menu" onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false); }}>
    <button ref={trigger} type="button" className="project-more-button" aria-label={text(`更多操作：${name}`, `More actions: ${name}`)}
      aria-haspopup="menu" aria-expanded={open && !busy} aria-controls={open && !busy ? id : undefined} disabled={busy}
      onClick={() => setOpen(value => !value)} onKeyDown={event => { if (event.key === 'ArrowDown') { event.preventDefault(); setOpen(true); } }}><MoreHorizontal size={17}/></button>
    {open && !busy && <div ref={menu} id={id} role="menu" aria-label={text(`项目操作：${name}`, `Project actions: ${name}`)} className="project-action-menu" onKeyDown={event => {
      const buttons = Array.from(menu.current?.querySelectorAll<HTMLButtonElement>('button') || []);
      const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setOpen(false); trigger.current?.focus(); }
      if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (current + (event.key === 'ArrowDown' ? 1 : buttons.length - 1)) % buttons.length;
        buttons[next]?.focus();
      }
    }}>
      <button role="menuitem" type="button" onClick={() => run(onEdit)}><Pencil size={14}/>{text('编辑项目', 'Edit project')}</button>
      <button role="menuitem" type="button" onClick={() => run(onArchive)}><Archive size={14}/>{archived ? text('恢复项目', 'Restore project') : text('归档项目', 'Archive project')}</button>
      <button role="menuitem" type="button" className="project-remove" onClick={() => run(onDelete)}><Trash2 size={14}/>{text('删除项目', 'Delete project')}</button>
    </div>}
  </div>;
}
