export function mergeConfig(base: any, partial: any): any {
  if (!partial || typeof partial !== 'object' || Array.isArray(partial)) return partial;
  const result = { ...base };
  for (const [key, value] of Object.entries(partial)) {
    result[key] = value && typeof value === 'object' && !Array.isArray(value) ? mergeConfig(result[key], value) : value;
  }
  return result;
}

/** Used by the development mock; the live defaults come from the backend. */
export function schemaDefaults(schema: any, node: any = schema): any {
  if (node.default !== undefined) return structuredClone(node.default);
  if (node.$ref) return schemaDefaults(schema, node.$ref.slice(2).split('/').reduce((part: any, key: string) => part[key], schema));
  if (node.type === 'object' && node.properties) return Object.fromEntries(
    Object.entries(node.properties).map(([key, property]) => [key, schemaDefaults(schema, property)]).filter(([, value]) => value !== undefined),
  );
  return undefined;
}
