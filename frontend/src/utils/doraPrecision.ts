import type { PlanDora } from '../api/types';

export type DoraPrecisionReport = PlanDora;

const dtypeNames: Record<string, string> = {bf16: 'BF16', fp16: 'FP16', fp32: 'FP32'};

export function readDoraPrecisionReport(value: unknown): DoraPrecisionReport | null {
  if (!value || typeof value !== 'object') return null;
  const report = value as Partial<DoraPrecisionReport>;
  if (typeof report.active !== 'boolean' || !['standard', 'comfyui'].includes(report.compute_mode || '')
    || typeof report.confirmation_required !== 'boolean'
    || !['base_dtypes', 'merge_dtypes', 'auto_merge_dtypes', 'confirmation_reasons'].every(key => {
      const values = report[key as keyof DoraPrecisionReport];
      return Array.isArray(values) && values.every(item => typeof item === 'string');
    })) return null;
  return report as DoraPrecisionReport;
}

export function doraDtypeLabel(values: string[] | undefined): string | null {
  if (!values?.length || values.some(value => !dtypeNames[value])) return null;
  return [...new Set(values)].map(value => dtypeNames[value]).join(' / ');
}

/** Only a current server plan knows the dtype of the actual adapter targets. */
export function doraAutoLabel(report: DoraPrecisionReport | null | undefined, english: boolean): string {
  const dtype = report?.active ? doraDtypeLabel(report.auto_merge_dtypes) : null;
  return english ? `Auto (${dtype || 'Pending check'})` : `自动（${dtype || '待检测'}）`;
}

export function doraConfirmationMessages(report: DoraPrecisionReport, english: boolean): string[] {
  const text = (zh: string, en: string) => english ? en : zh;
  const merged = doraDtypeLabel(report.merge_dtypes), automatic = doraDtypeLabel(report.auto_merge_dtypes);
  const messages = (report.confirmation_reasons || []).flatMap(reason => {
    if (reason === 'standard_mode') return [text('标准模式使用 FP32 计算 DoRA；低精度加载导出权重时，出图效果可能与训练预览不同。', 'Standard mode computes DoRA in FP32. Loading the exported weights at lower precision may produce images that differ from the training preview.')];
    if (reason === 'manual_merge_dtype_mismatch') return [merged && automatic
      ? text(`已选择 ${merged} 融合，当前底模自动解析为 ${automatic}。使用不同的融合精度可能改变出图效果。`, `${merged} merging is selected; Auto resolves to ${automatic} for the current base model. Using a different merge precision may change the generated result.`)
      : text('手动选择的融合精度与自动检测结果不同，可能改变出图效果。', 'The selected merge precision differs from Auto and may change the generated result.')];
    return [];
  });
  return messages.length ? messages : [report.confirmation_message || text('请确认当前 DoRA 计算方式和融合精度。', 'Confirm the current DoRA computation and merge precision.')];
}
