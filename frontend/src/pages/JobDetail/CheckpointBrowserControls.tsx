import React from 'react';
import { Trash2 } from 'lucide-react';
import type { JobCheckpoint } from '../../api/types';
import StudioSelect from '../../components/StudioSelect';
import { useWorkspaceText } from '../../utils/workspaceText';

import { PAGE_SIZES, useCheckpointBrowser } from './useCheckpointBrowser';

type Browser = ReturnType<typeof useCheckpointBrowser>;
/** Page size and pager when the list is long enough to need them; batch management always. */
export function CheckpointPagination({ browser, kind, disabled = false }: { browser: Browser; kind: 'outputs' | 'resume'; disabled?: boolean }) {
  const text = useWorkspaceText();
  return <div className="checkpoint-pagination">
    {browser.paged && <nav className="checkpoint-pager" aria-label={kind === 'outputs' ? text('产物分页', 'Output pagination') : text('恢复点分页', 'Resume point pagination')}>
      <StudioSelect aria-label={kind === 'outputs' ? text('每页产物数', 'Outputs per page') : text('每页恢复点数', 'Resume points per page')} value={String(browser.pageSize)} options={PAGE_SIZES.map(value => ({ value: String(value), label: text(`${value} 个 / 页`, `${value} / page`) }))} onValueChange={browser.changeSize} disabled={disabled}/>
      {browser.pages > 1 && <>
        <button type="button" className="ui-btn ui-btn-sm" disabled={disabled || browser.page <= 1} onClick={() => browser.setPage(browser.page - 1)}>{text('上一页', 'Previous')}</button>
        <span className="checkpoint-page-number">{browser.page} / {browser.pages}</span>
        <button type="button" className="ui-btn ui-btn-sm" disabled={disabled || browser.page >= browser.pages} onClick={() => browser.setPage(browser.page + 1)}>{text('下一页', 'Next')}</button>
      </>}
    </nav>}
    <button type="button" className="ui-btn ui-btn-sm" disabled={disabled} aria-pressed={browser.managing} onClick={browser.toggleManaging}>{browser.managing ? text('完成管理', 'Done') : text('批量管理', 'Manage')}</button>
  </div>;
}
export function CheckpointSelection({ browser, disabled = false, onDelete }: { browser: Browser; disabled?: boolean; onDelete: (items: JobCheckpoint[]) => void }) {
  const text = useWorkspaceText();
  const checkbox = React.useRef<HTMLInputElement>(null);
  React.useEffect(() => { if (checkbox.current) checkbox.current.indeterminate = browser.someSelected && !browser.allSelected; }, [browser.someSelected, browser.allSelected]);
  if (!browser.managing) return null;
  return <div className="checkpoint-selection">
    <label><input ref={checkbox} type="checkbox" checked={browser.allSelected} disabled={disabled || !browser.selectable.length} onChange={browser.togglePage}/>{text('全选本页', 'Select page')}</label>
    <span>{text(`已选 ${browser.selected.length} 项`, `${browser.selected.length} selected`)}</span>
    <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={disabled || !browser.selected.length} onClick={browser.clear}>{text('清除选择', 'Clear selection')}</button>
    <button type="button" className="ui-btn ui-btn-sm ui-btn-danger" disabled={disabled || !browser.selected.length} onClick={() => onDelete(browser.selected)}><Trash2 size={14}/>{text('删除所选', 'Delete selected')}</button>
  </div>;
}
