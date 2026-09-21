/**
 * Visual check for the parameter-form report: key-column alignment, trimmed
 * descriptions and the invalid-value styling. Run against the mock dev server.
 */
import { puppeteer } from './runtime.mjs';

const BASE = process.env.BASE || 'http://localhost:5199';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1500,1200'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1500, height: 1200 });
page.on('pageerror', (err) => console.error('BROWSER ERROR:', err.message));

await page.goto(`${BASE}/projects/p_9235ff0ba122/train`, { waitUntil: 'domcontentloaded', timeout: 90000 });
await sleep(4000);

// Reach the optimizer group and put it on Prodigy Plus Schedule-Free.
const opened = await page.evaluate(() => {
  const group = [...document.querySelectorAll('button')].find((b) => /优化器与学习率|Optimizer/.test(b.textContent || ''));
  if (!group) return 'group button not found';
  if (group.getAttribute('aria-expanded') === 'false') group.click();
  return 'ok';
});
await sleep(1200);
const picked = await page.evaluate(() => {
  const trigger = document.querySelector('[data-field-path="optimizer.type"] .studio-select');
  if (!trigger) return 'optimizer.type control missing';
  trigger.click();
  return 'opened';
});
await sleep(600);
await page.evaluate(() => {
  const option = [...document.querySelectorAll('[role="option"]')].find((o) => /Prodigy Plus/.test(o.textContent || ''));
  if (option) option.click();
});
await sleep(2500);

// Alignment: every configuration key in one column must share a right edge.
const alignment = await page.evaluate(() => {
  const columns = new Map();
  for (const field of document.querySelectorAll('.config-field[data-field-path^="optimizer."]')) {
    const code = field.querySelector('.config-field-key');
    if (!code) continue;
    const box = code.getBoundingClientRect();
    const fieldBox = field.getBoundingClientRect();
    if (!box.width) continue;
    const key = Math.round(fieldBox.left);
    if (!columns.has(key)) columns.set(key, []);
    columns.get(key).push({
      path: field.dataset.fieldPath,
      right: Math.round(box.right),
      help: !!field.querySelector('.config-help-trigger'),
      paragraphs: field.querySelectorAll(':scope > p').length,
    });
  }
  return [...columns.entries()].map(([left, rows]) => ({
    left,
    rights: [...new Set(rows.map((r) => r.right))],
    withoutHelp: rows.filter((r) => !r.help).map((r) => r.path),
    maxParagraphs: Math.max(...rows.map((r) => r.paragraphs)),
    rows: rows.length,
  }));
});

await page.evaluate(() => {
  const heading = [...document.querySelectorAll('.config-field-section h3, .config-group-title')].find((h) => /自动步长估计/.test(h.textContent || ''));
  (heading || document.querySelector('[data-field-path="optimizer.d0"]'))?.scrollIntoView({ block: 'center' });
});
await sleep(600);
await page.screenshot({ path: 'screenshots/frontend/verify-optimizer-alignment.png' });

// Invalid value: the mock backend rejects an adapter rank above 1024.
await page.evaluate(() => {
  const group = [...document.querySelectorAll('button')].find((b) => /适配器|Adapter/.test(b.textContent || ''));
  if (group && group.getAttribute('aria-expanded') === 'false') group.click();
});
await sleep(1000);
const typed = await page.evaluate(() => {
  const input = document.querySelector('[data-field-path="adapter.rank"] input[type="number"]');
  if (!input) return 'adapter.rank input missing';
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
  setter.call(input, '2000');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  return 'typed';
});
await sleep(4000);

const invalid = await page.evaluate(() => {
  const field = document.querySelector('.config-field[data-field-path="adapter.rank"]');
  if (!field) return { found: false };
  const input = field.querySelector('input[type="number"]');
  const error = field.querySelector('.config-field-error');
  const hint = field.querySelector('.config-field-hint');
  return {
    found: true,
    invalidClass: field.classList.contains('config-field-invalid'),
    ariaInvalid: input?.getAttribute('aria-invalid'),
    inputBorder: input && getComputedStyle(input).borderColor,
    errorText: error?.textContent,
    errorColor: error && getComputedStyle(error).color,
    hintColor: hint && getComputedStyle(hint).color,
  };
});
await page.evaluate(() => document.querySelector('[data-field-path="adapter.rank"]')?.scrollIntoView({ block: 'center' }));
await sleep(500);
await page.screenshot({ path: 'screenshots/frontend/verify-invalid-field.png' });

console.log(JSON.stringify({ opened, picked, typed, alignment, invalid }, null, 2));
await browser.close();
