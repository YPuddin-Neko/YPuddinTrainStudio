/** One name per group: the workflow rail and the section heading read the same label. */
export const PARAMETER_FLOW: Array<{ group: string; label: [string, string] }> = [
  { group: 'model', label: ['基础模型', 'Base model'] },
  { group: 'dataset', label: ['数据与分桶', 'Data and buckets'] },
  { group: 'caption', label: ['标签处理', 'Captions'] },
  { group: 'loop', label: ['训练循环', 'Training loop'] },
  { group: 'adapter', label: ['适配器', 'Adapter'] },
  { group: 'optimizer', label: ['优化器', 'Optimizer'] },
  { group: 'scheduler', label: ['学习率调度', 'LR schedule'] },
  { group: 'memory', label: ['显存与计算', 'Memory and compute'] },
  { group: 'objective', label: ['噪声与损失', 'Noise and loss'] },
  { group: 'sampling', label: ['采样预览', 'Sample previews'] },
  { group: 'validation', label: ['验证', 'Validation'] },
  { group: 'checkpoint', label: ['保存与恢复', 'Save and resume'] },
  { group: 'logging', label: ['日志与监控', 'Logging and monitoring'] },
];

export function parameterGroupLabel(group: string, english: boolean): string | undefined {
  return PARAMETER_FLOW.find(item => item.group === group)?.label[english ? 1 : 0];
}

export const PARAMETER_GROUP_ORDER = PARAMETER_FLOW.map(item => item.group);

/** Preserve backend-added groups while putting the known workflow in one order. */
export function workflowSchema<T extends { [key: string]: any }>(schema: T): T {
  return {...schema, 'x-ui-groups': [...PARAMETER_GROUP_ORDER, ...(schema['x-ui-groups'] || []).filter((group: string) => group !== 'training' && !PARAMETER_GROUP_ORDER.includes(group))]};
}
