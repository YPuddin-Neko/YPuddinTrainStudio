import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../client';
import { FamilyInfo } from '../types';

/**
 * 模型族数据源：一次拉取 /api/families，全应用共享缓存。
 * 所有族相关 UI（Models 族下拉、adapter.preset 选项、text_modes 联动、weights 提示）
 * 都从这里取，不写死。
 */
export function useFamilies() {
  return useQuery<FamilyInfo[]>({
    queryKey: ['families'],
    queryFn: () => apiClient.get<FamilyInfo[]>('/families'),
    staleTime: 5 * 60 * 1000,
  });
}

/** 按 name 找族，找不到时返回 undefined（不做硬编码兜底） */
export function familyByName(families: FamilyInfo[] | undefined, name: string | undefined): FamilyInfo | undefined {
  if (!families || !name) return undefined;
  return families.find((f) => f.name === name);
}
