import { puppeteer } from './runtime.mjs';

const BACKEND = 'http://127.0.0.1:8765';
const FRONTEND = 'http://localhost:3000';
const DATASET_PATH = '/tmp/ypuddin_demo';
const PROJECT_NAME = `toy-e2e-${Date.now()}`;

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

// Helper: set a React-controlled input value reliably via native setter + input event
async function setInputValue(input, page, value) {
  await input.evaluate((el, v) => {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, String(value));
  const actual = await input.evaluate((el) => el.value);
  if (actual !== String(value)) {
    throw new Error(`setInputValue failed: expected ${value}, got ${actual}`);
  }
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

  // ============ 1. Dashboard ============
  console.log('STEP 1: Dashboard');
  await page.goto(`${FRONTEND}/`, { waitUntil: 'domcontentloaded' });
  await sleep(2500);
  await page.screenshot({ path: 'screenshots/frontend/01-dashboard.png' });

  // ============ 2. Create project via UI ============
  console.log('STEP 2: Create project via UI');
  await page.goto(`${FRONTEND}/projects`, { waitUntil: 'domcontentloaded' });
  await sleep(2000);
  await clickButtonByText(page, 'New Project');
  await page.waitForSelector('[data-testid="project-name-input"]');
  await page.type('[data-testid="project-name-input"]', PROJECT_NAME);
  await clickButtonByText(page, 'Create', true);
  await sleep(2000);
  await page.screenshot({ path: 'screenshots/frontend/02-project-created.png' });

  // Get the project id from backend
  const projectsResp = await api('/projects');
  const projectItems = Array.isArray(projectsResp) ? projectsResp : projectsResp.items;
  const project = projectItems.find((p) => p.name === PROJECT_NAME);
  if (!project) throw new Error('Project not found after creation!');
  const pid = project.id;
  console.log('  project created:', pid);

  // ============ 3. Register dataset via API (UI is FE-M4 scope) ============
  console.log('STEP 3: Register dataset via API');
  const ds = await api(`/projects/${pid}/datasets`, {
    method: 'POST',
    body: { path: DATASET_PATH, repeats: 1, caption_ext: '.txt' },
  });
  console.log('  dataset registered:', ds.id || ds);

  // ============ 4. Train config page ============
  console.log('STEP 4: Configure training (toy-smoke preset + sources + sampling)');
  await page.goto(`${FRONTEND}/projects/${pid}/train`, { waitUntil: 'domcontentloaded' });
  await sleep(2500);

  // 4a. Select preset toy-smoke
  await page.select('select', 'toy-smoke');
  await sleep(1000);

  // 4b. Add dataset source path
  await clickButtonByText(page, 'Add Dataset Source');
  await sleep(500);
  const pathInput = await page.$('[data-testid="sources-editor"] input[placeholder="/path/to/dataset"]');
  if (!pathInput) throw new Error('sources path input not found');
  await pathInput.type(DATASET_PATH);

  // 4c. Enable sampling.enabled switch
  const samplingSwitch = await page.$('[data-testid="field-sampling.enabled"] input[type="checkbox"]');
  if (samplingSwitch) {
    const checked = await samplingSwitch.evaluate((el) => el.checked);
    if (!checked) await samplingSwitch.click();
  }
  await sleep(500);

  // 4d. Add sample prompt
  await clickButtonByText(page, 'Add Sample Prompt');
  await sleep(500);
  const promptArea = await page.$('[data-testid="prompts-editor"] textarea[placeholder="Prompt text"]');
  if (!promptArea) throw new Error('prompt textarea not found');
  await promptArea.type('a red square on white background');
  // set width/height to 64
  const promptsEditor = await page.$('[data-testid="prompts-editor"]');
  const numInputs = await promptsEditor.$$('input[type="number"]');
  // inputs order: seed, width, height
  if (numInputs.length >= 3) {
    await setInputValue(numInputs[1], page, '64');
    await setInputValue(numInputs[2], page, '64');
  }

  // 4e. Set sampling.steps = 2
  const stepsInput = await page.$('[data-testid="field-sampling.steps"] input[type="number"]');
  if (stepsInput) {
    await setInputValue(stepsInput, page, '2');
  }

  // Wait for plan debounce + response
  await sleep(2000);
  await page.screenshot({ path: 'screenshots/frontend/03-train-config.png' });

  // ============ 5. Enqueue ============
  console.log('STEP 5: Enqueue job');
  const enqueued = await clickButtonByText(page, 'Enqueue Training Job');
  if (!enqueued) throw new Error('Enqueue button not found');
  await sleep(2500);
  await page.screenshot({ path: 'screenshots/frontend/04-enqueued.png' });

  // Find the job id
  const jobsResp = await api('/jobs');
  const jobItems = Array.isArray(jobsResp) ? jobsResp : jobsResp.items;
  const job = jobItems.find((j) => j.project_id === pid) || jobItems[jobItems.length - 1];
  console.log('  job:', job.id, job.status);
  const jid = job.id;

  // ============ 6. Queue page ============
  console.log('STEP 6: Queue page');
  await page.goto(`${FRONTEND}/queue`, { waitUntil: 'domcontentloaded' });
  await sleep(2500);
  await page.screenshot({ path: 'screenshots/frontend/05-queue.png' });

  // ============ 7. Job detail: wait for completion ============
  console.log('STEP 7: Job detail (wait for completion)');
  await page.goto(`${FRONTEND}/jobs/${jid}`, { waitUntil: 'domcontentloaded' });
  // poll backend until job completed (toy train is fast, allow up to 60s)
  let finalJob = null;
  for (let i = 0; i < 30; i++) {
    await sleep(2000);
    finalJob = await api(`/jobs/${jid}`);
    if (['completed', 'failed', 'cancelled'].includes(finalJob.status)) break;
  }
  console.log('  final status:', finalJob.status);
  await sleep(1500);
  await page.screenshot({ path: 'screenshots/frontend/06-jobdetail-metrics.png' });

  // samples tab
  await clickButtonByText(page, 'Samples');
  await sleep(1500);
  await page.screenshot({ path: 'screenshots/frontend/07-jobdetail-samples.png' });

  // checkpoints tab
  await clickButtonByText(page, 'Checkpoints');
  await sleep(1500);
  await page.screenshot({ path: 'screenshots/frontend/08-jobdetail-checkpoints.png' });

  // logs tab
  await clickButtonByText(page, 'Logs');
  await sleep(1500);
  await page.screenshot({ path: 'screenshots/frontend/09-jobdetail-logs.png' });

  // ============ 8. Artifacts ============
  console.log('STEP 8: Artifacts page');
  await page.goto(`${FRONTEND}/artifacts`, { waitUntil: 'domcontentloaded' });
  await sleep(2000);
  await page.screenshot({ path: 'screenshots/frontend/10-artifacts.png' });

  // ============ 9. Convert artifact via API (button also exists in UI) ============
  console.log('STEP 9: Convert artifact via UI');
  const selectEl = await page.$('select[title="Convert format"]');
  if (selectEl) {
    await selectEl.select('kohya');
    await sleep(2500);
    await page.screenshot({ path: 'screenshots/frontend/11-artifact-converted.png' });
    console.log('  convert done via UI');
  } else {
    console.log('  WARN: convert select not found, skipping UI convert');
  }

  await browser.close();
  console.log('E2E FLOW COMPLETE');
}

run().catch((e) => {
  console.error('E2E FAILED:', e.message);
  process.exit(1);
});
