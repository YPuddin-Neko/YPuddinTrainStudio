import puppeteer from 'puppeteer-core';
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
await sleep(3000);
await page.screenshot({ path: 'screenshots/23-mock-sse-t0.png' });
await sleep(6000);
await page.screenshot({ path: 'screenshots/23-mock-sse-t6.png' });
await browser.close();
console.log('mock sse screenshots saved');
