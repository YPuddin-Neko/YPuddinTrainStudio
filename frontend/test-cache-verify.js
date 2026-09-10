import puppeteer from 'puppeteer-core';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1000'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 1000 });
await page.goto('http://localhost:3000/datasets/d_9e68ae980a1f', { waitUntil: 'domcontentloaded' });
await sleep(3000);
await page.screenshot({ path: 'screenshots/21-cache-coverage.png' });
await browser.close();
console.log('screenshot 21 saved');
