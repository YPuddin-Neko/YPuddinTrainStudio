import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import openapiTS, { astToString } from 'openapi-typescript';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.resolve(process.argv[2] || path.join(root, '../docs/api/openapi.json'));
const document = JSON.parse(await fs.readFile(source, 'utf8'));
const schema = document.components.schemas.TtsVersionConfig;
if (!schema?.properties || schema.additionalProperties !== false) throw new Error('Missing TtsVersionConfig schema');
const types = astToString(await openapiTS(structuredClone(document)));
await fs.writeFile(path.join(root, 'src/api/generated.ts'), types);
await fs.writeFile(path.join(root, 'src/pages/Tts/ttsVersionSchema.json'), JSON.stringify(schema, null, 2) + '\n');
const gptSovits = document.components.schemas.GptSovitsVersionConfig;
if (gptSovits) {
  const definitions = {};
  const resolve = value => {
    if (Array.isArray(value)) return value.map(resolve);
    if (!value || typeof value !== 'object') return value;
    if (typeof value.$ref === 'string' && value.$ref.startsWith('#/components/schemas/')) {
      const name = value.$ref.slice('#/components/schemas/'.length);
      if (!(name in definitions)) {
        definitions[name] = {};
        if (!document.components.schemas[name]) throw new Error(`Missing schema ${name}`);
        definitions[name] = resolve(document.components.schemas[name]);
      }
      return { ...value, $ref: `#/$defs/${name}` };
    }
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, resolve(item)]));
  };
  const generated = { ...resolve(gptSovits), $defs: definitions };
  await fs.writeFile(path.join(root, 'src/pages/Tts/gptSovitsVersionSchema.json'), JSON.stringify(generated, null, 2) + '\n');
}
process.stdout.write(`Generated API types and TTS configuration schema from ${source}\n`);
