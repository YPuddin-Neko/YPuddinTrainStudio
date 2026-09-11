import puppeteer from 'puppeteer-core';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,900'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 900 });
page.on('pageerror', (err) => console.error('BROWSER ERROR:', err.message));
await page.goto('http://localhost:3000/', { waitUntil: 'domcontentloaded', timeout: 90000 });
await sleep(3500);
await page.screenshot({ path: 'screenshots/38-apple-gpu.png' });
await browser.close();
console.log('saved 38');
