// 用于解析和计算 show_when 表达式的工具函数

export function evaluateShowWhen(expression: string, config: Record<string, any>): boolean {
  try {
    if (!expression || typeof expression !== 'string') return true;

    // 1. 解析 token，保护字符串不被替换
    const stringLiterals: string[] = [];
    let normalized = expression.replace(/'([^']*)'/g, (match) => {
      stringLiterals.push(match);
      return `__STR_${stringLiterals.length - 1}__`;
    });

    // 2. 替换操作符为 JS 语法
    normalized = normalized
      .replace(/&&/g, '&&')
      .replace(/\|\|/g, '||')
      .replace(/\btrue\b/g, 'true')
      .replace(/\bfalse\b/g, 'false')
      .replace(/\bnull\b/g, 'null');

    // 3. 替换字段路径 (如 adapter.algo) 为 config 查找表达式
    normalized = normalized.replace(/\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\b/g, (match) => {
      // 跳过 JS 关键字和占位符
      if (
        ['true', 'false', 'null', 'undefined', 'in', 'function', 'return'].includes(match) ||
        match.startsWith('__STR_')
      ) {
        return match;
      }
      // 转换成 config 取值形式，并处理 undefined
      const keys = match.split('.');
      let access = 'config';
      for (const key of keys) {
        access += `?.["${key}"]`;
      }
      return `(${access} ?? undefined)`;
    });

    // 4. 恢复字符串字面量
    normalized = normalized.replace(/__STR_(\d+)__/g, (_, index) => stringLiterals[Number(index)]);

    // 5. 执行表达式
    // 为了安全起见，可以加入简单的过滤，避免执行恶意代码
    const evaluateFn = new Function('config', `return (${normalized});`);
    return !!evaluateFn(config);
  } catch (error) {
    console.error(`Error evaluating show_when expression: ${expression}`, error);
    // 表达式解析失败时，默认显示该字段，保证配置可见性
    return true;
  }
}
