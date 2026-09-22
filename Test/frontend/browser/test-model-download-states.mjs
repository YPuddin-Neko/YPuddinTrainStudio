// Run against a local Vite preview. All API requests are intercepted; no model
// download or mutation reaches a real training service.
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { puppeteer } from './runtime.mjs';

const origin = process.env.STUDIO_BROWSER_URL || 'http://127.0.0.1:3019';
const output = fileURLToPath(new URL('../../screenshots/frontend/model-download-states/', import.meta.url));
mkdirSync(output, { recursive: true });
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_BINARY || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true,
  args: ['--no-sandbox'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1000, deviceScaleFactor: 1 });
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const settings = {
    paths: { bootstrap_env_dir: '', data_root: '/studio', models_dir: '/studio/models', cache_dir: '/studio/cache', output_dir: '/studio/output', output_mode: 'project' },
    server: { host: '127.0.0.1', port: 8123 }, ui: { language: 'zh-CN', theme: 'light' },
  };
  let assets = [];
  const catalog = [
    { id: 'anima-dit', family: 'anima', kind: 'dit', name: 'Anima 2B', filename: 'anima.safetensors' },
    { id: 'anima-encoder', family: 'anima', kind: 'text_encoder', name: 'Qwen3 0.6B', filename: 'encoder.safetensors' },
    { id: 'anima-vae', family: 'anima', kind: 'vae', name: 'Anima VAE', filename: 'vae.safetensors' },
  ];
  const failed = {
    id: 'failed', recommendation_id: 'anima-encoder', provider: 'huggingface', mirror: 'official', family: 'anima', kind: 'text_encoder',
    source_url: 'https://huggingface.co/test/anima/resolve/main/encoder.safetensors', filename: 'encoder.safetensors',
    target_path: '/studio/models/text_encoders/anima/encoder.safetensors', status: 'failed', error: '连接超时，请重试。',
    downloaded_bytes: 0, total_bytes: 1024 * 1024 * 1024, bytes_per_second: 0, eta_seconds: null, progress_at: null,
    dtype: 'bf16', is_default: true, purpose: 'training', model_id: null, created_at: 2, finished_at: 3,
  };
  let downloads = [{ ...failed, id: 'old-failed', created_at: 1, error: '过期的错误，不应显示。' }, failed];
  let retries = 0;
  await page.setRequestInterception(true);
  page.on('request', async request => {
    const url = new URL(request.url());
    if (url.origin !== origin) { await request.abort(); return; }
    if (!url.pathname.startsWith('/api/')) { await request.continue(); return; }
    let data;
    if (url.pathname === '/api/settings') data = settings;
    else if (url.pathname === '/api/families') data = [{ name: 'anima', label: 'Anima 2B', weights: [{ field: 'dit_path' }, { field: 'text_encoder_path' }, { field: 'vae_path' }] }];
    else if (url.pathname === '/api/jobs') data = [];
    else if (url.pathname === '/api/system/stats') data = { cpu_pct: 0, ram: { used_mb: 1024, total_mb: 16384 }, disks: [], gpus: [] };
    else if (url.pathname === '/api/models') data = assets;
    else if (url.pathname === '/api/models/downloads') data = downloads;
    else if (url.pathname === '/api/models/recommendations') data = catalog.map(entry => {
      const model = assets.find(asset => asset.kind === entry.kind);
      return { ...entry, size: 1024 * 1024 * 1024, dtype: 'bf16', recommended: true, model_id: model?.id || null, available_path: model?.path || null, is_default: model?.is_default || false,
        sources: [{ provider: 'huggingface', repo_id: 'test/anima', filename: entry.filename, revision: 'main', url: `https://huggingface.co/test/anima/blob/main/${entry.filename}` }] };
    });
    else if (url.pathname === '/api/models/downloads/failed/retry' && request.method() === 'POST') {
      retries += 1;
      const next = { ...failed, id: 'retry', created_at: 4, status: 'queued', error: null, finished_at: null };
      downloads.push(next); data = next;
    } else if (url.pathname === '/api/events') {
      await request.respond({ status: 200, contentType: 'text/event-stream', body: ': browser fixture\n\n' }); return;
    } else {
      errors.push(`Unexpected API request: ${request.method()} ${url.pathname}`);
      await request.respond({ status: 404, contentType: 'application/json', body: '{}' }); return;
    }
    await request.respond({ status: 200, contentType: 'application/json', body: JSON.stringify(data) });
  });

  await page.goto(`${origin}/settings/environment?tab=models&family=anima&view=downloads`, { waitUntil: 'domcontentloaded' });
  const card = '[data-testid="model-component-text_encoder"]';
  await page.waitForSelector(`${card} .model-download-error`);
  const readState = () => page.evaluate(selector => {
    const component = document.querySelector(selector);
    const button = component.querySelector('.model-catalog-action button');
    const error = component.querySelector('.model-download-error');
    return { button: button?.textContent.trim(), disabled: button?.disabled, background: button && getComputedStyle(button).backgroundColor,
      error: error?.textContent, errorColor: error && getComputedStyle(error).color,
      extraTasks: document.querySelectorAll('.model-download-inline-row').length,
      forbidden: [...document.querySelectorAll('.models-view-tabs button, .models-workspace button')].some(element => /^(下载任务|下载记录|更换来源)$/.test(element.textContent.trim())) };
  }, card);
  const evidence = { failed: await readState() };
  assert.equal(evidence.failed.button, '重试');
  assert.equal(evidence.failed.errorColor, 'rgb(220, 38, 38)');
  assert.equal(evidence.failed.extraTasks, 0);
  assert.equal(evidence.failed.forbidden, false);
  await page.screenshot({ path: `${output}failed-light.png`, fullPage: true });
  await page.evaluate(() => document.documentElement.classList.add('dark'));
  evidence.failedDark = await readState();
  assert.equal(evidence.failedDark.errorColor, 'rgb(252, 165, 165)');
  await page.screenshot({ path: `${output}failed-dark.png`, fullPage: true });
  await page.evaluate(() => document.documentElement.classList.remove('dark'));

  await page.click(`${card} .model-catalog-action button`);
  await page.waitForSelector(`${card} .model-button-downloading`);
  await page.waitForFunction(selector => getComputedStyle(document.querySelector(`${selector} .model-button-downloading`)).backgroundColor === 'rgb(229, 231, 235)', {}, card);
  evidence.queued = await readState();
  assert.equal(evidence.queued.button, '下载中');
  assert.equal(evidence.queued.disabled, true);
  assert.equal(evidence.queued.background, 'rgb(229, 231, 235)');
  assert.equal(evidence.queued.extraTasks, 0);
  assert.equal(evidence.queued.error, undefined);
  await page.click(`${card} .model-button-downloading`);
  assert.equal(retries, 1, 'A disabled download button must not resubmit');
  downloads = downloads.map(task => task.id === 'retry' ? { ...task, status: 'downloading', downloaded_bytes: 536870912, bytes_per_second: 8388608, eta_seconds: 64, progress_at: Date.now() / 1000 } : task);
  await page.click('button[aria-label="刷新模型"]');
  await page.waitForFunction(selector => document.querySelector(`${selector} progress`)?.value === 536870912, {}, card);
  evidence.running = await readState();
  assert.equal(evidence.running.disabled, true);
  assert.equal(evidence.running.extraTasks, 0);
  await page.screenshot({ path: `${output}running-light.png`, fullPage: true });

  downloads = downloads.map(task => task.id === 'retry' ? { ...task, status: 'completed', downloaded_bytes: task.total_bytes } : task);
  assets = [{ id: 'encoder-ready', family: 'anima', kind: 'text_encoder', path: failed.target_path, dtype: 'bf16', size: failed.total_bytes, is_default: true, exists: true, purpose: 'training', created_at: 5 }];
  await page.click('button[aria-label="刷新模型"]');
  await page.waitForSelector(`${card} .model-ready`);
  evidence.completed = await readState();
  assert.equal(evidence.completed.error, undefined);
  assert.equal(evidence.completed.button, undefined);
  assert.equal(evidence.completed.extraTasks, 0);
  assert.equal(downloads.length, 3, 'Backend history remains intact');
  await page.screenshot({ path: `${output}completed-light.png`, fullPage: true });
  assert.deepEqual(errors, []);
  writeFileSync(`${output}evidence.json`, `${JSON.stringify(evidence, null, 2)}\n`);
  console.log(JSON.stringify({ status: 'passed', states: Object.keys(evidence), output }));
} finally {
  await browser.close();
}
