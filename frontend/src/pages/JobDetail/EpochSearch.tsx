import { Search } from 'lucide-react';
import { parseEpochQuery } from '../../utils/epochFilter';
import { useWorkspaceText } from '../../utils/workspaceText';

/** Search by epoch number: "3", "2-5", "3, 7" or "8+". */
export default function EpochSearch({ value, onChange, label }: { value: string; onChange: (value: string) => void; label: string }) {
  const text = useWorkspaceText();
  const invalid = parseEpochQuery(value) === 'invalid';
  return <div className="epoch-search-field">
    <label className="epoch-search" data-invalid={invalid || undefined}>
      <Search size={14} aria-hidden="true"/>
      <input type="search" aria-label={label} aria-invalid={invalid || undefined} aria-describedby={invalid ? 'epoch-search-error' : undefined}
        placeholder={text('按轮次搜索，如 3、2-5、8+', 'Search epochs: 3, 2-5, 8+')} value={value} onChange={event => onChange(event.target.value)}/>
    </label>
    {invalid && <span id="epoch-search-error" className="epoch-search-error" role="status">{text('请输入轮次数字，如 3、2-5 或 8+', 'Enter epoch numbers such as 3, 2-5 or 8+')}</span>}
  </div>;
}
