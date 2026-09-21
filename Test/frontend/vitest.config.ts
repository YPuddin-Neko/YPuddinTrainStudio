import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(new URL('../../frontend/package.json', import.meta.url));
const frontend = fileURLToPath(new URL('../../frontend', import.meta.url));
const pkg = require('./package.json');

export default async () => {
  const { default: react } = await import(pathToFileURL(require.resolve('@vitejs/plugin-react')).href);
  return {
    root: frontend,
    server: { fs: { allow: [fileURLToPath(new URL("../..", import.meta.url))] } },
    plugins: [react()],
    resolve: { dedupe: [...Object.keys(pkg.dependencies), ...Object.keys(pkg.devDependencies)] },
    test: {
      globals: true,
      environment: 'jsdom',
      include: ['../Test/frontend/tests/**/*.test.{ts,tsx}'],
      setupFiles: ['../Test/frontend/tests/setup.ts'],
    },
  };
};
