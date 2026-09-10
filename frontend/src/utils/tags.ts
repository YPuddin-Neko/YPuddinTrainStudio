/**
 * Caption tag 解析/序列化/操作工具。
 * caption 以逗号分隔 tag，tag 内部允许空格（如 "1girl, long hair"）。
 * 序列化约定：", " 连接、去重（保序）、去空白、去空项。
 */

export function parseTags(caption: string | null | undefined): string[] {
  if (!caption) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of caption.split(',')) {
    const tag = raw.trim().replace(/\s+/g, ' ');
    if (!tag) continue;
    const key = tag.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      out.push(tag);
    }
  }
  return out;
}

export function serializeTags(tags: string[]): string {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of tags) {
    const tag = raw.trim().replace(/\s+/g, ' ');
    if (!tag) continue;
    const key = tag.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      out.push(tag);
    }
  }
  return out.join(', ');
}

export function addTag(tags: string[], tag: string): string[] {
  const t = tag.trim().replace(/\s+/g, ' ');
  if (!t) return tags;
  if (tags.some((x) => x.toLowerCase() === t.toLowerCase())) return tags;
  return [...tags, t];
}

export function removeTag(tags: string[], index: number): string[] {
  return tags.filter((_, i) => i !== index);
}

export function updateTag(tags: string[], index: number, value: string): string[] {
  const next = [...tags];
  next[index] = value;
  return next;
}

/** 把 fromIndex 的项移动到 toIndex（用于拖动排序） */
export function moveTag(tags: string[], fromIndex: number, toIndex: number): string[] {
  if (fromIndex === toIndex) return tags;
  if (fromIndex < 0 || fromIndex >= tags.length) return tags;
  const clamped = Math.max(0, Math.min(tags.length - 1, toIndex));
  const next = [...tags];
  const [item] = next.splice(fromIndex, 1);
  next.splice(clamped, 0, item);
  return next;
}

/** 批量合并：给一组 caption 添加/删除 tag，返回新的 caption */
export function mergeTagsAdd(caption: string, add: string[]): string {
  let tags = parseTags(caption);
  for (const a of add) {
    tags = addTag(tags, a);
  }
  return serializeTags(tags);
}

export function mergeTagsRemove(caption: string, remove: string[]): string {
  const lower = new Set(remove.map((r) => r.trim().toLowerCase()));
  return serializeTags(parseTags(caption).filter((t) => !lower.has(t.toLowerCase())));
}
