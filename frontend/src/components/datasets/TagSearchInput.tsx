import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { useWorkspaceText } from '../../utils/workspaceText';

export type SiteName = 'danbooru' | 'gelbooru' | 'e621' | 'rule34';
type Suggestion = { tag: string; category: string; posts: number | null; alias: string | null };

/** The word the caret is in, and where it starts and ends. */
const wordAt = (value: string, caret: number) => {
  const start = value.slice(0, caret).search(/\S*$/);
  const end = caret + (value.slice(caret).match(/^\S*/)?.[0].length ?? 0);
  return { word: value.slice(start, end), start, end };
};

/**
 * A line of site tags separated by spaces. The word being typed is looked up on the site, and picking
 * a suggestion puts the site's own spelling of the tag in its place.
 */
export default function TagSearchInput({ value, onChange, source, disabled, placeholder, id, ...aria }: {
  value: string; onChange: (value: string) => void; source: SiteName; disabled?: boolean; placeholder?: string; id?: string;
  'aria-describedby'?: string;
}) {
  const text = useWorkspaceText();
  const generated = React.useId();
  const inputId = id || `tag-search-${generated}`;
  const listId = `${inputId}-suggestions`;
  const input = React.useRef<HTMLInputElement>(null);
  const [caret, setCaret] = React.useState(0);
  const [focused, setFocused] = React.useState(false);
  const [dismissed, setDismissed] = React.useState(false);
  const [active, setActive] = React.useState(-1);
  const current = wordAt(value, caret);
  const term = current.word.replace(/^-/, '');
  const [asked, setAsked] = React.useState('');
  React.useEffect(() => { const timer = setTimeout(() => setAsked(term), 250); return () => clearTimeout(timer); }, [term]);
  const suggestions = useQuery({
    queryKey: ['site-tag-suggestions', source, asked], enabled: focused && asked.length >= 2 && asked === term && !disabled,
    staleTime: 5 * 60_000, retry: false, refetchOnWindowFocus: false,
    queryFn: ({ signal }) => apiClient.get<Suggestion[]>('/site-downloads/suggestions', { params: { source, q: asked }, silent: true, signal }),
  });
  const options = (asked === term && suggestions.data) || [];
  const open = focused && !dismissed && !disabled && options.length > 0;
  React.useEffect(() => { setActive(-1); setDismissed(false); }, [asked, source]);
  const categories: Record<string, [string, string]> = {
    character: ['角色', 'Character'], copyright: ['作品', 'Series'], artist: ['画师', 'Artist'], general: ['一般', 'General'], meta: ['元数据', 'Meta'],
  };
  const pick = (index: number) => {
    const option = options[index];
    if (!option) return;
    const prefix = current.word.startsWith('-') ? '-' : '';
    const rest = value.slice(current.end).replace(/^\s*/, '');
    const next = `${value.slice(0, current.start)}${prefix}${option.tag} ${rest}`;
    onChange(next);
    const position = current.start + prefix.length + option.tag.length + 1;
    setCaret(position); setDismissed(true);
    requestAnimationFrame(() => input.current?.setSelectionRange(position, position));
  };
  const keyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (!open) return;
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      const step = event.key === 'ArrowDown' ? 1 : -1;
      setActive(index => (index + step + options.length + (index < 0 && step < 0 ? 1 : 0)) % options.length);
    } else if ((event.key === 'Enter' || event.key === 'Tab') && active >= 0) {
      event.preventDefault(); pick(active);
    } else if (event.key === 'Escape') {
      event.preventDefault(); event.stopPropagation(); setDismissed(true);
    }
  };
  const track = (event: React.SyntheticEvent<HTMLInputElement>) => setCaret(event.currentTarget.selectionStart ?? event.currentTarget.value.length);
  return <div className="tag-search">
    <input ref={input} id={inputId} value={value} disabled={disabled} placeholder={placeholder} autoComplete="off" spellCheck={false}
      role="combobox" aria-autocomplete="list" aria-expanded={open} aria-controls={open ? listId : undefined}
      aria-activedescendant={open && active >= 0 ? `${listId}-${active}` : undefined} aria-describedby={aria['aria-describedby']}
      onChange={event => { onChange(event.target.value); track(event); setDismissed(false); }} onSelect={track} onKeyDown={keyDown}
      onFocus={event => { setFocused(true); track(event); }} onBlur={() => setFocused(false)}/>
    {open && <div id={listId} role="listbox" className="tag-search-list" aria-label={text('站点标签建议', 'Site tag suggestions')}>
      {options.map((option, index) => <div key={`${option.tag}-${option.alias || ''}`} id={`${listId}-${index}`} role="option" aria-selected={index === active}
        className="tag-search-option" data-category={option.category} onPointerDown={event => { event.preventDefault(); pick(index); }} onPointerMove={() => setActive(index)}>
        <span className="tag-search-name">{option.alias ? <><small>{option.alias} →</small> {option.tag}</> : option.tag}</span>
        <span className="tag-search-category">{text(...(categories[option.category] || [option.category, option.category]))}</span>
        {option.posts != null && <span className="tag-search-posts">{option.posts.toLocaleString()}</span>}
      </div>)}
    </div>}
  </div>;
}
