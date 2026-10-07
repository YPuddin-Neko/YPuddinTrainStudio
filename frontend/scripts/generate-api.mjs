import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import openapiTS, { astToString } from 'openapi-typescript';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.resolve(process.argv[2] || path.join(root, '../docs/api/openapi.json'));
const document = JSON.parse(await fs.readFile(source, 'utf8'));
const schema = document.components.schemas.TtsVersionConfig;
if (!schema?.properties || schema.additionalProperties !== false) throw new Error('Missing TtsVersionConfig schema');
const types = astToString(await openapiTS(document));
await fs.writeFile(path.join(root, 'src/api/generated.ts'), types);
await fs.writeFile(path.join(root, 'src/pages/Tts/ttsVersionSchema.json'), JSON.stringify(schema, null, 2) + '\n');
process.stdout.write(`Generated API types and TTS configuration schema from ${source}\n`);
