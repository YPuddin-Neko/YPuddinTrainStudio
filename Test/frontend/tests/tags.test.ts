import { describe, it, expect } from 'vitest';
import {
  parseTags,
  serializeTags,
  addTag,
  removeTag,
  updateTag,
  moveTag,
  mergeTagsAdd,
  mergeTagsRemove,
} from '../../../frontend/src/utils/tags';

describe('tags utils', () => {
  it('parseTags: 逗号分隔 + 去重 + 去空白', () => {
    expect(parseTags('1girl, smile, 1girl , ,cat_ears')).toEqual(['1girl', 'smile', 'cat_ears']);
    expect(parseTags('')).toEqual([]);
    expect(parseTags(null)).toEqual([]);
    expect(parseTags(undefined)).toEqual([]);
  });

  it('parseTags: 保留 tag 内部空格并压缩', () => {
    expect(parseTags('long  hair, blue  eyes')).toEqual(['long hair', 'blue eyes']);
  });

  it('serializeTags: 反序列化后格式规范', () => {
    expect(serializeTags(['1girl', 'smile'])).toBe('1girl, smile');
    expect(serializeTags([' a ', '', 'b'])).toBe('a, b');
  });

  it('parse + serialize 往返一致', () => {
    const caption = '1girl, solo, long hair, blue eyes';
    expect(serializeTags(parseTags(caption))).toBe(caption);
  });

  it('addTag: 大小写不敏感去重', () => {
    expect(addTag(['1girl'], '1GIRL')).toEqual(['1girl']);
    expect(addTag(['1girl'], 'smile')).toEqual(['1girl', 'smile']);
    expect(addTag([], '')).toEqual([]);
  });

  it('removeTag: 按下标删除', () => {
    expect(removeTag(['a', 'b', 'c'], 1)).toEqual(['a', 'c']);
    expect(removeTag(['a'], 0)).toEqual([]);
  });

  it('updateTag: 原位替换', () => {
    expect(updateTag(['a', 'b'], 0, 'x')).toEqual(['x', 'b']);
  });

  it('moveTag: 拖动排序（前移/后移/越界保护）', () => {
    expect(moveTag(['a', 'b', 'c', 'd'], 0, 2)).toEqual(['b', 'c', 'a', 'd']);
    expect(moveTag(['a', 'b', 'c', 'd'], 3, 0)).toEqual(['d', 'a', 'b', 'c']);
    expect(moveTag(['a', 'b'], 0, 0)).toEqual(['a', 'b']);
    expect(moveTag(['a', 'b'], 5, 0)).toEqual(['a', 'b']); // 非法下标不动
    expect(moveTag(['a', 'b'], 1, 99)).toEqual(['a', 'b']); // clamp 到末尾
  });

  it('mergeTagsAdd / mergeTagsRemove: 批量编辑 caption', () => {
    expect(mergeTagsAdd('a, b', ['c', 'a'])).toBe('a, b, c');
    expect(mergeTagsRemove('a, b, c', ['B', 'x'])).toBe('a, c');
    expect(mergeTagsRemove('', ['a'])).toBe('');
  });
});
