/** A validation error as the service reports it: pydantic's kind (`less_than`) and bound (`{ lt: 1 }`) beside its wording. */
export interface ValidationIssue { loc?: unknown; msg?: unknown; type?: unknown; ctx?: unknown }

const number = (value: unknown, percent: boolean) => {
  if (typeof value !== 'number' && !(typeof value === 'string' && /^-?\d+(\.\d+)?(e-?\d+)?$/i.test(value))) return String(value);
  // A field shown as a percentage states its bound as one too: below 1 reads below 100%.
  return percent ? `${Number((Number(value) * 100).toFixed(4))}%` : String(Number(value));
};

/**
 * The message for an error of a kind Studio knows, in the field's unit; null for other kinds (a validator's own
 * reason) and when the bound the message needs is missing.
 */
export function describeValidation(issue: ValidationIssue, { label = '', percent = false, english = false }: { label?: string; percent?: boolean; english?: boolean } = {}): string | null {
  const ctx = (issue.ctx && typeof issue.ctx === 'object' ? issue.ctx : {}) as Record<string, unknown>;
  const text = (zh: string, en: string) => english ? en : zh;
  const bound = (key: string) => key in ctx ? number(ctx[key], percent) : null;
  switch (issue.type) {
    case 'missing': return label ? text(`请填写或选择${label}`, `Fill in or choose ${label}`) : text('请填写此项', 'This is required');
    case 'greater_than_equal': { const value = bound('ge'); return value && text(`输入值应大于或等于 ${value}`, `Enter ${value} or more`); }
    case 'greater_than': { const value = bound('gt'); return value && text(`输入值应大于 ${value}`, `Enter more than ${value}`); }
    case 'less_than_equal': { const value = bound('le'); return value && text(`输入值应小于或等于 ${value}`, `Enter ${value} or less`); }
    case 'less_than': { const value = bound('lt'); return value && text(`输入值应小于 ${value}`, `Enter less than ${value}`); }
    case 'multiple_of': { const value = bound('multiple_of'); return value && text(`输入值应为 ${value} 的倍数`, `Enter a multiple of ${value}`); }
    case 'int_parsing': case 'int_type': case 'int_from_float': return text('请输入整数', 'Enter a whole number');
    case 'float_parsing': case 'float_type': case 'finite_number': return text('请输入有效数字', 'Enter a valid number');
    case 'bool_parsing': case 'bool_type': return text('请选择开启或关闭', 'Choose on or off');
    case 'string_type': return text('请输入文字', 'Enter text');
    case 'string_too_short': return ctx.min_length === 1 ? (label ? text(`请填写${label}`, `Fill in ${label}`) : text('请填写此项', 'This is required'))
      : 'min_length' in ctx ? text(`至少输入 ${ctx.min_length} 个字符`, `Enter at least ${ctx.min_length} characters`) : null;
    case 'string_too_long': return 'max_length' in ctx ? text(`最多输入 ${ctx.max_length} 个字符`, `Enter at most ${ctx.max_length} characters`) : null;
    case 'too_short': return 'min_length' in ctx ? text(`至少需要 ${ctx.min_length} 项`, `Add at least ${ctx.min_length}`) : null;
    case 'too_long': return 'max_length' in ctx ? text(`最多 ${ctx.max_length} 项`, `Use at most ${ctx.max_length}`) : null;
    case 'literal_error': case 'enum': return text('请从可选项中选择', 'Choose one of the options');
    case 'string_pattern_mismatch': return text('格式不正确', 'The format is not valid');
    case 'extra_forbidden': return text('当前版本不支持此参数，请检查导入的配置', 'This version has no such setting; check the imported configuration');
    case 'list_type': case 'dict_type': case 'model_type': case 'model_attributes_type': return text('格式不正确', 'The format is not valid');
    default: return null;
  }
}
