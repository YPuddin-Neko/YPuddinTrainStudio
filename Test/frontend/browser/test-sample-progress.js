import { puppeteer } from './runtime.mjs';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1000'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 1000 });
page.on('pageerror', (err) => console.error('BROWSER ERROR:', err.message));
await page.goto('http://localhost:3001/jobs/job_01', { waitUntil: 'domcontentloaded' });
// 等 sample_progress 推送出现（800ms 一推，等 ~4s 让 done 涨起来）
await sleep(4000);
const text = await page.$eval('[data-testid="sample-progress"]', (el) => el.textContent).catch(() => null);
console.log('sample-progress bar text:', text);
await page.screenshot({ path: 'screenshots/frontend/24-mock-sample-progress.png' });
await browser.close();
console.log('screenshot 24 saved');
