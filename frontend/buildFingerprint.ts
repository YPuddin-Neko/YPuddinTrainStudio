import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import type { Plugin } from 'vite';

/** Content checks survive archive extraction and clocks differing across machines. */
export function buildFingerprint(): Plugin {
  const root = process.cwd();
  const files = (directory: string): string[] => existsSync(directory) ? readdirSync(directory).flatMap(name => {
    const path = join(directory, name);
    return statSync(path).isDirectory() ? files(path) : [path];
  }) : [];
  const digest = (paths: string[], base: string) => Object.fromEntries(paths.map(path => [
    relative(base, path).split('\\').join('/'), createHash('sha256').update(readFileSync(path)).digest('hex'),
  ]));
  return {
    name: 'studio-build-fingerprint',
    apply: 'build',
    closeBundle() {
      const inputs = [...files(join(root, 'src')), ...files(join(root, 'public')),
        ...readdirSync(root).filter(name => /^(package(?:-lock)?\.json|index\.html|buildFingerprint\.ts|(?:vite|tailwind|postcss)\.config\.[^/]+|tsconfig[^/]*\.json|\.env(?:\..*)?)$/.test(name)).map(name => join(root, name))];
      const dist = join(root, 'dist');
      writeFileSync(join(dist, '.source-manifest.json'), JSON.stringify({
        version: 1, inputs: digest(inputs, root),
        outputs: digest(files(dist).filter(path => !path.endsWith('.source-manifest.json')), dist),
      }, null, 2));
    },
  };
}
