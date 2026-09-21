import { puppeteer } from './runtime.mjs';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1100'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 1100 });
page.on('pageerror', (err) => console.error('BROWSER ERROR:', err.message));

const shots = [
  ['/', '30-dashboard'],
  ['/projects', '31-projects'],
  ['/queue', '32-queue'],
  ['/jobs/j_9ac6d1d6260f', '33-jobdetail'],
  ['/projects/p_9235ff0ba122/train', '34-trainconfig'],
  ['/datasets/d_9e68ae980a1f', '35-dataset'],
  ['/models', '36-models'],
  ['/settings', '37-settings'],
];
for (const [path, name] of shots) {
  await page.goto(`http://localhost:3000${path}`, { waitUntil: 'domcontentloaded', timeout: 90000 });
  await sleep(2800);
  await page.screenshot({ path: `screenshots/frontend/${name}.png` });
  console.log(name, 'saved');
}
await browser.close();
