import { puppeteer } from './runtime.mjs';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1200'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 1200 });
page.on('pageerror', (err) => console.error('BROWSER ERROR:', err.message));
await page.goto('http://localhost:3000/jobs/j_9ac6d1d6260f', { waitUntil: 'domcontentloaded' });
await sleep(3500);
await page.screenshot({ path: 'screenshots/frontend/22-jobdetail-charts.png', fullPage: false });
await browser.close();
console.log('screenshot 22 saved');
