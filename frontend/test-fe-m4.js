import puppeteer from 'puppeteer-core';

const BACKEND = 'http://127.0.0.1:8765';
const FRONTEND = 'http://localhost:3000';
const DATASET_PATH = '/tmp/ypuddin_demo';
const PROJECT_NAME = `m4-e2e-${Date.now()}`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function api(path, opts = {}) {
  const res = await fetch(`${BACKEND}/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) throw new Error(`API ${path} failed: ${res.status} ${await res.text()}`);
  return res.json();
}

async function setInputValue(input, page, value) {
  await input.evaluate((el, v) => {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, String(value));
  const actual = await input.evaluate((el) => el.value);
  if (actual !== String(value)) throw new Error(`setInputValue failed: expected ${value}, got ${actual}`);
}

async function clickButtonByText(page, text, exact = false) {
  const buttons = await page.$$('button');
  for (const btn of buttons) {
    const t = await btn.evaluate((el) => el.textContent?.trim());
    if (exact ? t === text : t?.includes(text)) {
      await btn.click();
      return true;
    }
  }
  return false;
}

async function run() {
  const browser = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1600,1000'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000 });
  page.on('pageerror', (err) => console.error('BROWSER ERROR:', err.message));
  page.on('dialog', async (dialog) => {
    console.log('  [dialog]', dialog.message().slice(0, 80));
    await dialog.accept();
  });

  // ============ 1. 新建项目 ============
  console.log('STEP 1: Create project');
  await page.goto(`${FRONTEND}/projects`, { waitUntil: 'domcontentloaded' });
  await sleep(2000);
  await clickButtonByText(page, 'New Project');
  await page.waitForSelector('[data-testid="project-name-input"]');
  await page.type('[data-testid="project-name-input"]', PROJECT_NAME);
  await clickButtonByText(page, 'Create', true);
  await sleep(2000);

  const projectsResp = await api('/projects');
  const project = (Array.isArray(projectsResp) ? projectsResp : projectsResp.items).find((p) => p.name === PROJECT_NAME);
  if (!project) throw new Error('project not created');
  const pid = project.id;
  console.log('  project:', pid);

  // ============ 2. 项目详情页：UI 注册数据集 ============
  console.log('STEP 2: Register dataset via UI (project detail)');
  await page.goto(`${FRONTEND}/projects/${pid}`, { waitUntil: 'domcontentloaded' });
  await sleep(2000);
  await clickButtonByText(page, 'Register Dataset');
  await page.waitForSelector('[data-testid="register-dataset-modal"]');
  const pathInput = await page.$('[data-testid="register-dataset-modal"] input[placeholder="/path/to/images"]');
  await pathInput.type(DATASET_PATH);
  await clickButtonByText(page, 'Register', true);
  await sleep(2000);
  await page.screenshot({ path: 'screenshots/12-project-datasets.png' });

  // 轮询索引完成
  let datasets = await api(`/projects/${pid}/datasets`);
  // 后端返回 DatasetInfo[]：{source, stats, index_status}
  const did = (datasets[0].source || datasets[0]).id;
  console.log('  dataset:', did, '(waiting for index ready…)');
  let info = null;
  for (let i = 0; i < 40; i++) {
    await sleep(1000);
    info = await api(`/datasets/${did}`);
    if (info.index_status === 'ready') break;
  }
  console.log('  index_status:', info.index_status, '| images:', info.stats.images);
  if (info.index_status !== 'ready') throw new Error('indexing did not finish in time');

  // ============ 3. 数据集页：概览 + 网格 ============
  console.log('STEP 3: Dataset page overview + grid');
  await page.goto(`${FRONTEND}/datasets/${did}`, { waitUntil: 'domcontentloaded' });
  await sleep(3000);
  await page.screenshot({ path: 'screenshots/13-dataset-overview.png' });

  // ============ 4. 编辑 caption（加 tag 保存） ============
  console.log('STEP 4: Edit caption of first image');
  const firstCard = await page.$('[data-testid^="image-card-"] img');
  if (!firstCard) throw new Error('no image card found');
  await firstCard.click();
  await page.waitForSelector('[data-testid="caption-editor"]');
  await sleep(500);
  const addInput = await page.$('[data-testid="tag-add-input"]');
  await addInput.type('e2e_added_tag');
  await page.keyboard.press('Enter');
  await sleep(300);
  await page.screenshot({ path: 'screenshots/14-caption-editor.png' });
  await page.waitForSelector('[data-testid="caption-save-btn"]');
  await clickButtonByText(page, 'Save caption');
  await sleep(1500);
  console.log('  caption saved');

  // ============ 5. 批量加 tag（选两张） ============
  console.log('STEP 5: Batch add tag to 2 images');
  const cards = await page.$$('[data-testid^="image-card-"]');
  for (let i = 0; i < 2 && i < cards.length; i++) {
    const selectBtn = await cards[i].$('button');
    await selectBtn.click();
  }
  await sleep(500);
  const batchInput = await page.$('[data-testid="batch-add-input"]');
  await batchInput.type('batch_m4_tag');
  await page.screenshot({ path: 'screenshots/15-batch-tags.png' });
  await page.waitForSelector('[data-testid="batch-apply-btn"]');
  await clickButtonByText(page, 'Apply to selection');
  await sleep(2000);
  console.log('  batch tags applied');

  // 验证：q 搜索过滤
  const searchInput = await page.$('[data-testid="dataset-search"]');
  await setInputValue(searchInput, page, 'batch_m4_tag');
  await sleep(1500);
  await page.screenshot({ path: 'screenshots/16-search-filter.png' });
  const afterFilter = await api(`/datasets/${did}/images?q=batch_m4_tag`);
  console.log('  images with batch_m4_tag:', afterFilter.total);

  // ============ 6. 预缓存任务（先把项目草稿设为 toy-smoke + 数据源） ============
  console.log('STEP 6a: Apply toy-smoke preset + dataset source, save project draft');
  await page.goto(`${FRONTEND}/projects/${pid}/train`, { waitUntil: 'domcontentloaded' });
  await sleep(2500);
  await page.select('select', 'toy-smoke');
  await sleep(800);
  await clickButtonByText(page, 'Add Dataset Source');
  await sleep(500);
  const srcPathInput = await page.$('[data-testid="sources-editor"] input[placeholder="/path/to/dataset"]');
  if (!srcPathInput) throw new Error('sources path input not found');
  await srcPathInput.type(DATASET_PATH);
  await sleep(2500); // 等草稿自动保存 (debounce 1s) + plan
  const draftCheck = await api(`/projects/${pid}/config`);
  console.log('  draft family:', draftCheck?.model?.family, '| sources:', JSON.stringify(draftCheck?.dataset?.sources));

  console.log('STEP 6b: Pre-cache job');
  await page.goto(`${FRONTEND}/datasets/${did}`, { waitUntil: 'domcontentloaded' });
  await sleep(2000);
  await clickButtonByText(page, 'Pre-cache');
  await sleep(2500);
  // 等待 cache job 完成
  let cacheJob = null;
  for (let i = 0; i < 40; i++) {
    await sleep(1500);
    const jobsResp = await api(`/jobs?project_id=${pid}`);
    const items = Array.isArray(jobsResp) ? jobsResp : jobsResp.items;
    cacheJob = items.find((j) => j.type === 'cache');
    if (cacheJob && ['completed', 'failed', 'cancelled'].includes(cacheJob.status)) break;
  }
  console.log('  cache job:', cacheJob?.id, cacheJob?.status, cacheJob?.error || '');
  await page.screenshot({ path: 'screenshots/17-precache.png' });

  // ============ 7. 项目详情：jobs / artifacts tabs ============
  console.log('STEP 7: Project detail tabs');
  await page.goto(`${FRONTEND}/projects/${pid}`, { waitUntil: 'domcontentloaded' });
  await sleep(2000);
  await clickButtonByText(page, 'Jobs');
  await sleep(1000);
  await page.screenshot({ path: 'screenshots/18-project-jobs.png' });

  // ============ 8. 模型权重页：添加模型 ============
  console.log('STEP 8: Models page add');
  await page.goto(`${FRONTEND}/models`, { waitUntil: 'domcontentloaded' });
  await sleep(2000);
  await clickButtonByText(page, 'Add Model');
  await page.waitForSelector('[data-testid="add-model-modal"]');
  const modelPathInput = await page.$('[data-testid="add-model-modal"] input[placeholder="/models/xxx.safetensors"]');
  await modelPathInput.type('/private/tmp/ypuddin_data/runs/j_0dfc9491bd8d/lora-final.safetensors');
  await clickButtonByText(page, 'Add', true);
  await sleep(1500);
  await page.screenshot({ path: 'screenshots/19-models.png' });

  // ============ 9. 设置页：改语言/主题 ============
  console.log('STEP 9: Settings page');
  await page.goto(`${FRONTEND}/settings`, { waitUntil: 'domcontentloaded' });
  await sleep(2000);
  await page.select('[data-testid="settings-language"]', 'en');
  await clickButtonByText(page, 'Save Settings');
  await sleep(1500);
  await page.screenshot({ path: 'screenshots/20-settings.png' });
  // 恢复中文
  await page.select('[data-testid="settings-language"]', 'zh-CN');
  await clickButtonByText(page, 'Save Settings');
  await sleep(1000);

  await browser.close();
  console.log('FE-M4 E2E COMPLETE');
}

run().catch((e) => {
  console.error('E2E FAILED:', e.message);
  process.exit(1);
});
