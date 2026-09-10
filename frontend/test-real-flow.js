import puppeteer from 'puppeteer-core';
import fs from 'fs';

async function run() {
  const browser = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  const page = await browser.newPage();
  page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
  page.on('pageerror', err => console.error('BROWSER ERROR:', err.message));

  console.log('Opening dashboard...');
  await page.goto('http://localhost:3000/', { waitUntil: 'networkidle0' });
  await page.screenshot({ path: 'screenshots/dashboard.png' });

  console.log('Creating project...');
  await page.goto('http://localhost:3000/projects');
  await page.waitForSelector('button');
  const buttons = await page.$$('button');
  for (const btn of buttons) {
    const text = await btn.evaluate(el => el.textContent);
    if (text?.includes('New Project')) {
      await btn.click();
      break;
    }
  }
  await page.waitForSelector('input');
  await page.type('input', 'toy-test-project');
  await page.screenshot({ path: 'screenshots/create-project.png' });
  
  console.log('Configuring training...');
  await page.goto('http://localhost:3000/projects/proj_01/train');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'screenshots/train-config.png' });
  
  console.log('Opening job queue...');
  await page.goto('http://localhost:3000/queue');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: 'screenshots/queue.png' });
  
  await browser.close();
  console.log('All flows executed successfully.');
}

run().catch(console.error);
