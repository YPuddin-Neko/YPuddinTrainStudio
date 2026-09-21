import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
const require = createRequire(new URL('../../frontend/package.json', import.meta.url));
const { ESLint } = require('eslint');
const eslint = new ESLint({
  cwd: fileURLToPath(new URL('../..', import.meta.url)),
  overrideConfigFile: fileURLToPath(new URL('../../frontend/eslint.config.js', import.meta.url)),
});
const results = await eslint.lintFiles(['frontend/src', 'frontend/vite.config.ts', 'Test/frontend/tests', 'Test/frontend/mocks', 'Test/frontend/vitest.config.ts']);
const formatter = await eslint.loadFormatter('stylish');
process.stdout.write(formatter.format(results));
if (results.some(result => result.errorCount || result.warningCount)) process.exitCode = 1;
