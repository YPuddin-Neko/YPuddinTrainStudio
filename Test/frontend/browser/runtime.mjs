import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { mkdirSync } from 'node:fs';
const require = createRequire(new URL('../../../frontend/package.json', import.meta.url));
export const puppeteer = (await import(pathToFileURL(require.resolve('puppeteer-core')).href)).default;
const output = fileURLToPath(new URL('../../screenshots/frontend', import.meta.url));
mkdirSync(output, { recursive: true });
process.chdir(fileURLToPath(new URL('../..', import.meta.url)));
