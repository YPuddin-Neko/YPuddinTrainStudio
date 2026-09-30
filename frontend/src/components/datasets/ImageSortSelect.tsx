import StudioSelect from '../StudioSelect';
import { useWorkspaceText } from '../../utils/workspaceText';
import './image-sort-select.css';

export type ImageSort = 'filename' | 'folder' | 'modified';
export default function ImageSortSelect({ value, onChange, disabled = false }: { value: ImageSort; onChange: (value: ImageSort) => void; disabled?: boolean }) {
  const text = useWorkspaceText();
  return <StudioSelect className="image-sort-select" aria-label={text('图片排序', 'Image order')} value={value} disabled={disabled}
    onValueChange={value => onChange(value as ImageSort)} options={[
      { value: 'filename', label: text('文件名 · 自然排序', 'Filename · natural order') },
      { value: 'folder', label: text('文件夹', 'Folder') },
      { value: 'modified', label: text('修改时间 · 最新优先', 'Modified · newest first') },
    ]}/>;
}
