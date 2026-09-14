import { evaluateShowWhen } from '../schema/showWhen';

type Config = Record<string, any>;
type Reason = { zh?: string; en?: string };
type Rules = { fixed?: Record<string, unknown>; reason?: Record<string, Reason> };
type Capability = Rules & { aliases?: string[]; typed_fields?: string[]; defaults?: Record<string, unknown>; conditional?: Array<Rules & { when: string }> };

export function configPathValue(config: Config, path: string) {
  return path.split('.').reduce((value, key) => value?.[key], config);
}

function assign(config: Config, path: string, value: unknown): Config {
  const keys = path.split('.');
  const result = {...config};
  let node = result;
  keys.forEach((key, index) => {
    if (index === keys.length - 1) node[key] = value;
    else node = node[key] = {...node[key]};
  });
  return result;
}

function optimizerSchema(schema: any) {
  const section = schema?.properties?.optimizer;
  return section?.$ref ? section.$ref.slice(2).split('/').reduce((node: any, key: string) => node?.[key], schema) : section;
}

function capability(schema: any, config: Config): Capability | undefined {
  const name = config.optimizer?.type || optimizerSchema(schema)?.properties?.type?.default;
  return schema?.['x-optimizer-capabilities']?.[name];
}

function optimizerName(schema: any, name: string): string {
  const normalized = name.toLowerCase();
  const options: string[] = optimizerSchema(schema)?.properties?.type?.['x-ui']?.options || [];
  if (options.includes(normalized)) return normalized;
  for (const [id, profile] of Object.entries(schema?.['x-optimizer-capabilities'] || {}) as Array<[string, Capability]>) {
    if (id === normalized || profile.aliases?.some(alias => alias.toLowerCase() === normalized)) return id;
  }
  return name;
}

/** Legacy extra arguments become the dedicated values before defaults are filled. */
function migrateArguments(schema: any, source: Config): Config {
  const args = source.optimizer?.args;
  if (!args || typeof args !== 'object') return source;
  const properties = optimizerSchema(schema)?.properties || {};
  const profile = capability(schema, source);
  const options: string[] = properties.type?.['x-ui']?.options || [];
  if (!profile && !options.includes(source.optimizer?.type)) return source;
  const available = new Set(['lr', 'weight_decay', ...(profile?.typed_fields || [])]);
  const betasCondition = properties.betas?.['x-ui']?.show_when;
  if (!betasCondition || evaluateShowWhen(betasCondition, source)) available.add('betas');
  const epsCondition = properties.eps?.['x-ui']?.show_when;
  if (!epsCondition || evaluateShowWhen(epsCondition, source)) available.add('eps');
  const fixed = optimizerRules(schema, source).fixed;
  let next = source;
  const remaining = {...args};
  for (const [name, value] of Object.entries(args)) {
    if (!properties[name] || !available.has(name)) continue;
    const original = source.optimizer?.[name];
    // Keep conflicting imports intact so validation can explain the conflict.
    if (original !== undefined && JSON.stringify(original) !== JSON.stringify(properties[name].default) && JSON.stringify(original) !== JSON.stringify(value)) continue;
    if (`optimizer.${name}` in fixed && JSON.stringify(value) !== JSON.stringify(fixed[`optimizer.${name}`])) continue;
    next = assign(next, `optimizer.${name}`, value);
    delete remaining[name];
  }
  return next === source ? source : assign(next, 'optimizer.args', remaining);
}

export function optimizerRules(schema: any, config: Config): Required<Rules> {
  const rule = capability(schema, config);
  const result = {fixed: {...rule?.fixed}, reason: {...rule?.reason}};
  for (const conditional of rule?.conditional || []) {
    if (!evaluateShowWhen(conditional.when, config)) continue;
    Object.assign(result.fixed, conditional.fixed);
    Object.assign(result.reason, conditional.reason);
  }
  return result;
}

export function normalizeOptimizerConfig(schema: any, source: Config): Config {
  const type = source.optimizer?.type;
  const name = typeof type === 'string' ? optimizerName(schema, type) : type;
  let next = migrateArguments(schema, name === type ? source : assign(source, 'optimizer.type', name));
  const profile = capability(schema, next);
  for (const [path, value] of Object.entries(profile?.defaults || {})) {
    if (configPathValue(next, path) === undefined) next = assign(next, path, value);
  }
  if (profile) {
    for (const [name, property] of Object.entries(optimizerSchema(schema)?.properties || {}) as Array<[string, any]>) {
      if (property.default !== undefined && next.optimizer?.[name] === undefined) next = assign(next, `optimizer.${name}`, property.default);
    }
  }
  const rules = optimizerRules(schema, next);
  for (const [path, value] of Object.entries(rules.fixed)) {
    if (JSON.stringify(configPathValue(next, path)) !== JSON.stringify(value)) next = assign(next, path, value);
  }
  if ('optimizer.group_lr' in rules.fixed && Array.isArray(next.adapter?.rules)) {
    const entries = next.adapter.rules;
    if (entries.some((rule: Config) => rule.lr != null)) {
      next = assign(next, 'adapter.rules', entries.map((rule: Config) => rule.lr == null ? rule : {...rule, lr: null}));
    }
  }
  return next;
}

/** Explicit algorithm changes start with that algorithm's defaults. Saved values are restored by the editor. */
export function selectOptimizer(schema: any, config: Config, type: string): Config {
  let next = assign(config, 'optimizer.type', type);
  const profile = capability(schema, next);
  const properties = optimizerSchema(schema)?.properties || {};
  // Optimizer-specific extras cannot safely carry over into another constructor.
  next = assign(next, 'optimizer.args', {});
  for (const [name, property] of Object.entries(properties) as Array<[string, any]>) {
    if (name === 'type' || name === 'args' || property.default === undefined) continue;
    next = assign(next, `optimizer.${name}`, property.default);
  }
  for (const [path, value] of Object.entries(profile?.defaults || {})) next = assign(next, path, value);
  return normalizeOptimizerConfig(schema, next);
}

export function restoreOptimizerSelection(schema: any, current: Config, previous: Config): Config {
  let next: Config = {...current, optimizer: previous.optimizer};
  if (previous.scheduler) next = {...next, scheduler: previous.scheduler};
  if (previous.adapter?.lr_scale !== undefined) next = assign(next, 'adapter.lr_scale', previous.adapter.lr_scale);
  if (Array.isArray(current.adapter?.rules) && Array.isArray(previous.adapter?.rules)) {
    const rates = new Map(previous.adapter.rules.map((rule: Config) => [rule.match, rule.lr]));
    next = assign(next, 'adapter.rules', current.adapter.rules.map((rule: Config) => rates.has(rule.match) ? {...rule, lr: rates.get(rule.match)} : rule));
  }
  return normalizeOptimizerConfig(schema, next);
}

export function optimizerManagedReason(schema: any, config: Config, path: string, english = false) {
  const rule = optimizerRules(schema, config);
  const arrayOverride = path === 'adapter.rules.lr' && 'optimizer.group_lr' in rule.fixed;
  if (!(path in rule.fixed) && !arrayOverride) return undefined;
  const reason = rule.reason[arrayOverride ? 'adapter.rules[].lr' : path] || rule.reason[arrayOverride ? 'optimizer.group_lr' : path];
  return reason?.[english ? 'en' : 'zh'] || (english ? 'Managed by the selected optimizer.' : '由当前优化器自动管理。');
}

export function managedValueLabel(value: unknown, english = false) {
  if (value && typeof value === 'object') return english ? 'No overrides' : '不单独覆盖';
  if (typeof value === 'boolean') return value ? (english ? 'Enabled' : '已开启') : (english ? 'Disabled' : '未开启');
  return String(value ?? '');
}
