// 增强型 show_when 表达式解释器，支持完整语法：
// - 标识符：a.b.c
// - 字符串字面量：'lokr'
// - 数字：1, 0.05
// - 布尔/空：true, false, null
// - 运算符：==, !=, <, <=, >, >=, &&, ||, !, 括号 ()
// - in 操作符：adapter.algo in ['lokr', 'lora']

export function evaluateShowWhen(expression: string, config: Record<string, any>): boolean {
  if (!expression || typeof expression !== 'string' || !expression.trim()) return true;

  try {
    let expr = expression.trim();

    // 1. 替换 in 语法：<left> in [<items>] -> [<items>].includes(<left>)
    // 匹配如: adapter.algo in ['lokr', 'lora'] 或 algo in [1, 2]
    expr = expr.replace(
      /([a-zA-Z0-9_$.]+)\s+in\s+\[(.*?)\]/g,
      (_, left, items) => `[${items}].includes(${left})`
    );

    // 2. 将字段引用 path 替换为安全性属性访问路径
    // 将 'string' 临时替换掉避免混淆
    const strings: string[] = [];
    expr = expr.replace(/'([^']*)'/g, (m) => {
      strings.push(m);
      return `__STR_${strings.length - 1}__`;
    });

    // 替换标识符 (支持包含 . 的路径)
    expr = expr.replace(/\b([a-zA-Z_$][a-zA-Z0-9_$]*(?:\.[a-zA-Z0-9_$]+)*)\b/g, (match) => {
      if (
        ['true', 'false', 'null', 'undefined', 'includes'].includes(match) ||
        match.startsWith('__STR_')
      ) {
        return match;
      }
      const parts = match.split('.');
      return `(config?.${parts.join('?.')})`;
    });

    // 恢复字符串字面量
    expr = expr.replace(/__STR_(\d+)__/g, (_, idx) => strings[Number(idx)]);

    const fn = new Function('config', `return Boolean(${expr});`);
    return fn(config);
  } catch (err) {
    console.warn(`[showWhen] Failed to evaluate "${expression}":`, err);
    return true;
  }
}
