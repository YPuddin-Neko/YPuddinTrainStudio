import { X } from 'lucide-react';
import { PathInput } from './PathBrowser';
import { useWorkspaceText } from '../utils/workspaceText';

export default function ReferenceImageInput({ value = [], onChange, disabled = false, label }: {
  value?: string[]; onChange: (value: string[]) => void; disabled?: boolean; label?: string;
}) {
  const text = useWorkspaceText();
  const title = label || text('参考图片', 'Reference image');
  const change = (index: number, path: string) => {
    const next = value.length ? [...value] : [''];
    next[index] = path;
    onChange(next.length === 1 && !path ? [] : next);
  };
  return <fieldset className="m-0 min-w-0 border-0 p-0 space-y-2" disabled={disabled} aria-label={title}>
    {(value.length ? value : ['']).map((path, index) => <div className="flex min-w-0 items-start gap-2" key={index}>
      <div className="min-w-0 flex-1"><PathInput value={path} ariaLabel={value.length > 1 ? `${title} ${index + 1}` : title}
        placeholder={text('选择一张图片', 'Choose an image')} onChange={next => change(index, next)}/></div>
      {value.length > 1 && <button type="button" className="ui-btn ui-btn-quiet ui-btn-icon ui-btn-danger" aria-label={text(`移除参考图片 ${index + 1}`, `Remove reference image ${index + 1}`)}
        onClick={() => onChange(value.filter((_, other) => other !== index))}><X size={14}/></button>}
    </div>)}
  </fieldset>;
}
