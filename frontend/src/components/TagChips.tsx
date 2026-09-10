import React from 'react';
import { useTranslation } from 'react-i18next';
import { X } from 'lucide-react';
import { addTag, moveTag, parseTags, removeTag, serializeTags, updateTag } from '../utils/tags';

interface TagChipsProps {
  caption: string;
  onChange: (caption: string) => void;
  readOnly?: boolean;
}

/**
 * Caption → tag chips 编辑器：增删改、HTML5 拖拽排序。
 * 所有变更通过 serializeTags 序列化回逗号分隔字符串。
 */
export const TagChips: React.FC<TagChipsProps> = ({ caption, onChange, readOnly = false }) => {
  const { t } = useTranslation();
  const tags = React.useMemo(() => parseTags(caption), [caption]);
  const [draft, setDraft] = React.useState('');
  const [editIndex, setEditIndex] = React.useState<number | null>(null);
  const [editValue, setEditValue] = React.useState('');
  const [dragIndex, setDragIndex] = React.useState<number | null>(null);
  const [dragOver, setDragOver] = React.useState<number | null>(null);

  const commit = (next: string[]) => onChange(serializeTags(next));

  const handleAdd = () => {
    const t = draft.trim();
    if (!t) return;
    commit(addTag(tags, t));
    setDraft('');
  };

  const startEdit = (idx: number) => {
    setEditIndex(idx);
    setEditValue(tags[idx]);
  };

  const commitEdit = () => {
    if (editIndex === null) return;
    const v = editValue.trim();
    commit(v ? updateTag(tags, editIndex, v) : removeTag(tags, editIndex));
    setEditIndex(null);
  };

  return (
    <div className="space-y-2" data-testid="tag-chips">
      <div className="flex flex-wrap gap-1.5">
        {tags.map((tag, idx) => (
          <span
            key={`${tag}-${idx}`}
            data-testid={`tag-chip-${idx}`}
            draggable={!readOnly}
            onDragStart={() => setDragIndex(idx)}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(idx);
            }}
            onDrop={(e) => {
              e.preventDefault();
              if (dragIndex !== null) commit(moveTag(tags, dragIndex, idx));
              setDragIndex(null);
              setDragOver(null);
            }}
            onDragEnd={() => {
              setDragIndex(null);
              setDragOver(null);
            }}
            className={`inline-flex items-center space-x-1 px-2 py-1 rounded-md text-xs border select-none ${
              dragOver === idx && dragIndex !== null
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/40'
                : 'border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-800'
            } ${readOnly ? '' : 'cursor-grab active:cursor-grabbing'}`}
            onDoubleClick={() => !readOnly && startEdit(idx)}
            title={readOnly ? tag : t('dataset.tagChipsHint', '双击编辑，拖拽排序')}
          >
            {editIndex === idx ? (
              <input
                autoFocus
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                onBlur={commitEdit}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commitEdit();
                  if (e.key === 'Escape') setEditIndex(null);
                }}
                className="w-24 px-1 py-0.5 text-xs border rounded dark:bg-slate-900"
                data-testid="tag-edit-input"
              />
            ) : (
              <span className="font-mono">{tag}</span>
            )}
            {!readOnly && (
              <button
                type="button"
                onClick={() => commit(removeTag(tags, idx))}
                className="text-slate-400 hover:text-red-500"
                data-testid={`tag-remove-${idx}`}
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </span>
        ))}
      </div>
      {!readOnly && (
        <div className="flex space-x-2">
          <input
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ',') {
                e.preventDefault();
                handleAdd();
              }
            }}
            placeholder={t('dataset.addTagPlaceholder')}
            className="flex-1 px-2 py-1.5 text-xs border rounded dark:bg-slate-900 dark:border-slate-600 font-mono"
            data-testid="tag-add-input"
          />
          <button
            type="button"
            onClick={handleAdd}
            className="px-2.5 py-1.5 text-xs bg-slate-200 dark:bg-slate-700 rounded hover:bg-slate-300 dark:hover:bg-slate-600"
          >
            {t('dataset.addTag')}
          </button>
        </div>
      )}
    </div>
  );
};
