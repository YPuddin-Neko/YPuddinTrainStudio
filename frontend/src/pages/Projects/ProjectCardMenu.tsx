import React from 'react';
import { Link } from 'react-router-dom';
import { Archive, ArchiveRestore, MoreHorizontal, Pencil, Settings2, Trash2 } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';
import './projects.css';

export default function ProjectCardMenu({ name, archived, busy, onEdit, onArchive, onDelete, scope = 'project', archiveDisabledReason, parentArchiveUrl, workspace = false }: {
  name: string; archived: boolean; busy: boolean; onEdit?: () => void; onArchive: () => void; onDelete: () => void;
  scope?: 'project' | 'version'; archiveDisabledReason?: string; parentArchiveUrl?: string; workspace?: boolean;
}) {
  const text = useWorkspaceText();
  const [open, setOpen] = React.useState(false);
  const root = React.useRef<HTMLDivElement>(null);
  const trigger = React.useRef<HTMLButtonElement>(null);
  const menu = React.useRef<HTMLDivElement>(null);
  const id = React.useId();
  React.useEffect(() => {
    if (!open) return;
    menu.current?.querySelector<HTMLElement>('button:not(:disabled),a')?.focus();
    const outside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false); };
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [open]);
  const run = (action: () => void) => { setOpen(false); trigger.current?.focus(); action(); };
  return <div ref={root} className={workspace ? 'project-workspace-menu' : 'project-card-menu'} onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false); }}>
    <button ref={trigger} type="button" className={workspace ? 'ui-btn ui-btn-icon' : 'ui-btn ui-btn-sm ui-btn-icon project-more-button'} aria-label={workspace ? text('版本操作', 'Version actions') : text(`更多操作：${name}`, `More actions: ${name}`)} title={workspace ? text('版本操作', 'Version actions') : undefined}
      aria-haspopup="menu" aria-expanded={open && !busy} aria-controls={open && !busy ? id : undefined} disabled={busy}
      onClick={() => setOpen(value => !value)} onKeyDown={event => { if (event.key === 'ArrowDown') { event.preventDefault(); setOpen(true); } }}>{workspace ? <Settings2 size={14}/> : <MoreHorizontal size={17}/>}</button>
    {open && !busy && <div ref={menu} id={id} role="menu" aria-label={scope === 'version' ? text(`版本操作：${name}`, `Version actions: ${name}`) : text(`项目操作：${name}`, `Project actions: ${name}`)} className="project-action-menu" onKeyDown={event => {
      const buttons = Array.from(menu.current?.querySelectorAll<HTMLElement>('button:not(:disabled),a') || []);
      const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setOpen(false); trigger.current?.focus(); }
      if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (current + (event.key === 'ArrowDown' ? 1 : buttons.length - 1)) % buttons.length;
        buttons[next]?.focus();
      }
    }}>
      {!archived && onEdit && <button role="menuitem" type="button" onClick={() => run(onEdit)}><Pencil size={14}/>{text('编辑项目', 'Edit project')}</button>}
      {!(workspace && parentArchiveUrl) && <button role="menuitem" type="button" disabled={!!archiveDisabledReason} title={archiveDisabledReason} onClick={() => run(onArchive)}>{archived ? <ArchiveRestore size={14}/> : <Archive size={14}/>}{scope === 'version' ? archived ? text('恢复版本','Restore version') : text('归档版本','Archive version') : archived ? text('恢复项目', 'Restore project') : text('归档项目', 'Archive project')}</button>}
      {parentArchiveUrl && <Link role="menuitem" to={parentArchiveUrl} onClick={()=>setOpen(false)}>{workspace ? <ArchiveRestore size={14}/> : <Archive size={14}/>}{workspace ? text('恢复项目','Restore project') : text('查看归档项目','View archived project')}</Link>}
      {archived && <button role="menuitem" type="button" className="project-remove" onClick={() => run(onDelete)}><Trash2 size={14}/>{scope === 'version' ? text('永久删除版本','Permanently delete version') : text('永久删除项目', 'Permanently delete project')}</button>}
    </div>}
  </div>;
}
