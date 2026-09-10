import puppeteer from 'puppeteer-core';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const browser = await puppeteer.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: 'new',
  args: ['--no-sandbox', '--window-size=1600,1100'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 1100 });
page.on('pageerror', (err) => console.error('BROWSER ERROR:', err.message));

// 25: krea2 训练配置页（应用 krea2-lokr-default 预设）
await page.goto('http://localhost:3000/projects/p_9235ff0ba122/train', { waitUntil: 'domcontentloaded' });
await sleep(3000);
// 选择 krea2-lokr-default 预设（第一个 select 是预设选择器）
await page.select('select', 'krea2-lokr-default');
await sleep(2500);
// 滚动到 model 分组顶部
await page.evaluate(() => {
  document.querySelector('.overflow-y-auto')?.scrollTo(0, 0);
});
await sleep(500);
await page.screenshot({ path: 'screenshots/25-krea2-train-config.png' });
console.log('25 saved');

// 26: 模型权重页（krea2 fp8 标签）
await page.goto('http://localhost:3000/models', { waitUntil: 'domcontentloaded' });
await sleep(2500);
await page.screenshot({ path: 'screenshots/26-krea2-models.png' });
console.log('26 saved');

await browser.close();
