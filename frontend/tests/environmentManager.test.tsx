import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { File as NodeFile } from 'node:buffer';
import { EnvironmentManagerPanel } from '../src/components/EnvironmentManagerPanel';
import i18n from '../src/i18n';

const server = setupServer();
beforeAll(async () => {
  const form = await new Request('http://localhost', { method: 'POST', headers: { 'content-type': 'application/x-www-form-urlencoded' }, body: '' }).formData();
  vi.stubGlobal('FormData', form.constructor);
  vi.stubGlobal('File', NodeFile);
  server.listen({ onUnhandledRequest: 'error' });
});
afterEach(() => { cleanup(); server.resetHandlers(); vi.restoreAllMocks(); });
afterAll(() => { server.close(); vi.unstubAllGlobals(); });
let runtime: ReturnType<typeof environment>;
let operations: ReturnType<typeof operation>[];
let create = vi.fn<(body: unknown) => void>();
let apply = vi.fn<(id: unknown) => void>();

function environment() {
  const pkg = (name: string, backend: string | null, version: string | null) => ({ name, backend, version, supported: true, reason: 'supported', available: !!version, importable: !!version, kernel_tested: !!backend && !!version, wheel_required: false, error: null, docs_url: 'https://example.com/docs' });
  return {
    runtime: { python: '3.12.1', python_executable: 'C:\\Studio\\venv\\Scripts\\python.exe', platform: 'Windows', machine: 'AMD64', torch: '2.5.1+cu128', cuda_runtime: '12.8', cuda_available: true, mps_available: false, gpu_capability: [8, 9], gpus: [{ name: 'RTX test', telemetry_source: 'nvml' }], virtual_environment: true, cuda_device_count: 2, distributed_available: true, nccl_available: true, multi_gpu_training: false, training_device_policy: 'single_device' },
    packages: [
      { ...pkg('torch', null, '2.5.1'), supported: false, reason: 'protected_runtime' },
      pkg('xformers', 'xformers', null),
      { ...pkg('flash-attn', 'flash_attn', null), wheel_required: true },
      pkg('tensorboard', null, null),
      pkg('wandb', null, null),
      pkg('nvidia-ml-py', null, null),
      pkg('schedulefree', null, null),
      pkg('sageattention', 'sage', '2.2.0'),
      pkg('onnxruntime', null, null),
    ],
    attention_default: 'auto', restart_required: false, maintenance: false, running_jobs: false, probe_deferred: false,
  };
}
function operation(packageName = 'xformers', status = 'ready') {
  return { id: 'env_test', package: packageName, action: 'install', status, created_at: 1, dismissed_at: null as number | null, plan: [{ name: packageName, from_version: null, version: '1.2.3' }], logs: ['Resolved compatible wheel; Torch unchanged.'], error: null as string | null, restart_required: false };
}
beforeEach(async () => {
  await i18n.changeLanguage('zh-CN');
  runtime = environment(); operations = []; create = vi.fn(); apply = vi.fn();
  server.use(
    http.get('/api/environment', () => HttpResponse.json(runtime)),
    http.get('/api/environment/torch', () => HttpResponse.json({ builds: [], operations: [], current_python: runtime.runtime.python_executable, selected_environment: null, environments: [], disk_free_bytes: 100 * 1024 ** 3, minimum_free_bytes: 8 * 1024 ** 3, optional_extensions: [] })),
    http.get('/api/environment/operations', () => HttpResponse.json(operations)),
    http.post('/api/environment/operations', async ({ request }) => {
      const body = await request.json() as { package: string; action: string };
      create(body); operations = [operation(body.package)]; return HttpResponse.json(operations[0], { status: 202 });
    }),
    http.post('/api/environment/operations/:id/apply', ({ params }) => {
      apply(params.id); operations = [{ ...operations[0], status: 'completed', restart_required: true }]; runtime.restart_required = true; runtime.maintenance = true;
      return HttpResponse.json(operations[0], { status: 202 });
    }),
    http.post('/api/environment/operations/:id/cancel', () => { operations = [{ ...operations[0], status: 'cancelled' }]; return HttpResponse.json(operations[0]); }),
    http.post('/api/environment/operations/:id/dismiss', ({ params }) => { const op = operations.find(item => item.id === params.id)!; op.dismissed_at = Date.now() / 1000; return HttpResponse.json(op); }),
    http.put('/api/environment/settings', async ({ request }) => { const body = await request.json() as { attention_default: string }; runtime.attention_default = body.attention_default; return HttpResponse.json(body); }),
  );
});

describe('real environment management UI contracts', () => {
  it.each([
    ['windows-cuda', 'Windows CUDA'], ['linux-cuda', 'Linux CUDA'], ['macos-mps', 'macOS MPS'],
    ['windows-cpu', 'Windows CPU'], ['linux-cpu', 'Linux CPU'], ['macos-cpu', 'macOS CPU'], ['legacy', '旧版环境'],
  ])('shows the actual deployment profile %s', async (profile, label) => {
    Object.assign(runtime.runtime, { environment_profile: profile });
    render(<EnvironmentManagerPanel />);
    expect(await screen.findByText(label)).toBeInTheDocument();
    expect(screen.getByText('部署环境')).toBeInTheDocument();
    if (profile.endsWith('-cpu')) {
      expect(screen.getByText('计算后端').parentElement).toHaveTextContent('CPU');
    }
  });
  it('shows only optional CUDA attention extensions and ignores removed package deep links', async () => {
    render(<EnvironmentManagerPanel focusPackage="onnxruntime" />);
    await screen.findByTestId('environment-package-xformers');
    expect(screen.getByTestId('environment-package-flash-attn')).toBeInTheDocument();
    for (const name of ['torch', 'tensorboard', 'wandb', 'nvidia-ml-py', 'schedulefree', 'onnxruntime', 'sageattention']) {
      expect(screen.queryByTestId(`environment-package-${name}`)).not.toBeInTheDocument();
    }
    expect(screen.queryByRole('button', { name: '检查安装计划' })).not.toBeInTheDocument();
    expect(screen.queryByTestId('environment-operations')).not.toBeInTheDocument();
    expect(screen.queryByText('环境操作记录')).not.toBeInTheDocument();
    expect(within(screen.getByRole('navigation')).getAllByRole('button')).toHaveLength(3);
  });
  it('shows the target runtime and requires plan review before mutation', async () => {
    render(<EnvironmentManagerPanel />);
    expect(await screen.findByText('2.5.1+cu128')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '检查安装计划' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('解释器与显卡诊断'));
    expect(screen.getByText('C:\\Studio\\venv\\Scripts\\python.exe')).toBeVisible();
    const row = screen.getByTestId('environment-package-xformers');
    fireEvent.click(within(row).getByRole('button', { name: '安装' }));
    fireEvent.click(screen.getByText('手动版本与 wheel'));
    fireEvent.change(screen.getByLabelText('xformers 版本'), { target: { value: '1.2.3' } });
    fireEvent.click(screen.getByRole('button', { name: '检查安装计划' }));
    await waitFor(() => expect(create).toHaveBeenCalledWith({ package: 'xformers', action: 'install', version: '1.2.3' }));
    const confirm = await screen.findByRole('button', { name: '确认并执行此计划' });
    expect(apply).not.toHaveBeenCalled();
    expect(screen.getByLabelText('安装日志')).toHaveTextContent('Torch unchanged');
    fireEvent.click(confirm);
    await waitFor(() => expect(apply).toHaveBeenCalledWith('env_test'));
    expect(await screen.findByText(/重启前队列不会启动新任务/)).toBeInTheDocument();
  });

  it('blocks environment modifications while training and leaves attention selection to the training form', async () => {
    runtime.running_jobs = true; runtime.probe_deferred = true;
    render(<EnvironmentManagerPanel />);
    expect(await screen.findByText(/训练或缓存任务正在运行/)).toBeInTheDocument();
    expect(within(screen.getByTestId('environment-package-xformers')).getByRole('button', { name: '安装' })).toBeDisabled();
    expect(screen.queryByRole('combobox', { name: '新任务默认注意力' })).not.toBeInTheDocument();
    expect(create).not.toHaveBeenCalled();
  });

  it('Windows flash uses an explicitly uploaded wheel instead of a source build', async () => {
    server.use(http.post('/api/environment/wheels', async ({ request }) => {
      expect(request.headers.get('content-type')).toContain('multipart/form-data; boundary=');
      expect(await request.text()).toContain('flash_attn.whl');
      return HttpResponse.json({ wheel_id: 'wheel_ok', package: 'flash-attn', filename: 'flash_attn-1.2.3-cp312-cp312-win_amd64.whl', version: '1.2.3', sha256: 'abc' }, { status: 201 });
    }));
    render(<EnvironmentManagerPanel />);
    const row = await screen.findByTestId('environment-package-flash-attn');
    fireEvent.click(within(row).getByRole('button', { name: '安装' }));
    expect(screen.getByRole('button', { name: '检查安装计划' })).toBeDisabled();
    expect(screen.getByText(/此平台需要预编译 wheel/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('flash-attn wheel'), { target: { files: [new File(['wheel'], 'flash_attn.whl')] } });
    await waitFor(() => expect(screen.getByRole('button', { name: '检查安装计划' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: '检查安装计划' }));
    await waitFor(() => expect(create).toHaveBeenCalledWith({ package: 'flash-attn', action: 'install', version: '1.2.3', wheel_id: 'wheel_ok' }));
    expect(apply).not.toHaveBeenCalled();
  });

  it('shows concrete validation errors and preserves the editable form', async () => {
    server.use(http.post('/api/environment/operations', () => HttpResponse.json({ error: { code: 'request.validation', message: 'Request validation failed', details: { errors: [{ loc: ['body', 'version'], msg: 'Exact version required' }] } } }, { status: 422 })));
    render(<EnvironmentManagerPanel />);
    const row = await screen.findByTestId('environment-package-xformers');
    fireEvent.click(within(row).getByRole('button', { name: '安装' }));
    fireEvent.click(screen.getByText('手动版本与 wheel'));
    fireEvent.change(screen.getByLabelText('xformers 版本'), { target: { value: 'invalid' } });
    fireEvent.click(screen.getByRole('button', { name: '检查安装计划' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Exact version required');
    expect(screen.getByLabelText('xformers 版本')).toHaveValue('invalid');
  });

  it('persists a dismissed failure across drawer remounts and preserves explicit history', async () => {
    operations = [{ ...operation('xformers', 'failed'), error: 'Protected Torch dependency conflict' }];
    const first = render(<EnvironmentManagerPanel />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Protected Torch dependency conflict');
    expect(screen.getByLabelText('安装日志')).toHaveTextContent('Torch unchanged');
    expect(screen.queryByRole('button', { name: '确认并执行此计划' })).not.toBeInTheDocument();
    expect(apply).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: '关闭结果' }));
    await waitFor(() => expect(screen.queryByTestId('environment-operations')).not.toBeInTheDocument());
    expect(operations[0].dismissed_at).not.toBeNull();
    first.unmount();
    render(<EnvironmentManagerPanel />);
    await screen.findByRole('button', { name: '历史记录（1）' });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '历史记录（1）' }));
    fireEvent.click(within(screen.getByTestId('environment-operations')).getByRole('button', { expanded: false }));
    expect(await screen.findByLabelText('安装日志')).toHaveTextContent('Torch unchanged');
    expect(screen.queryByRole('button', { name: '关闭结果' })).not.toBeInTheDocument();
    expect(operations[0].status).toBe('failed');
  });

  it('omits completed history and failures superseded by a completed operation', async () => {
    operations = [{ ...operation('xformers', 'completed'), id: 'new' }, { ...operation('flash-attn', 'failed'), id: 'old', error: 'Old error' }];
    render(<EnvironmentManagerPanel />);
    await screen.findByTestId('environment-package-xformers');
    expect(screen.queryByTestId('environment-operations')).not.toBeInTheDocument();
    expect(screen.queryByText('Old error')).not.toBeInTheDocument();
  });

  it('restores an in-progress install and logs without exposing an interrupt action', async () => {
    operations = [operation('flash-attn', 'installing')];
    render(<EnvironmentManagerPanel />);
    expect(await screen.findByLabelText('安装日志')).toHaveTextContent('Torch unchanged');
    expect(screen.getByText('下载并安装')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '取消计划' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '关闭结果' })).not.toBeInTheDocument();
    expect(within(screen.getByTestId('environment-package-xformers')).getByRole('button', { name: '安装' })).toBeDisabled();
  });

  it('reopens concrete failure details when a collapsed active install fails', async () => {
    operations = [operation('flash-attn', 'installing')];
    render(<EnvironmentManagerPanel />);
    await screen.findByLabelText('安装日志');
    fireEvent.click(within(screen.getByTestId('environment-operations')).getByRole('button', { expanded: true }));
    expect(screen.queryByLabelText('安装日志')).not.toBeInTheDocument();
    operations = [{ ...operations[0], status: 'failed', error: 'Wheel verification failed' }];
    fireEvent.click(screen.getByRole('button', { name: '重新检测' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Wheel verification failed');
    expect(screen.getByLabelText('安装日志')).toHaveTextContent('Torch unchanged');
  });

  it('does not rewrite a saved attention choice when opening environment settings', async () => {
    runtime.attention_default = 'sage';
    render(<EnvironmentManagerPanel />);
    await screen.findByTestId('environment-package-xformers');
    expect(runtime.attention_default).toBe('sage');
    expect(screen.queryByRole('combobox', { name: '新任务默认注意力' })).not.toBeInTheDocument();
  });

  it('shows the actual Apple backend instead of reporting missing CUDA as a fault', async () => {
    runtime.runtime.cuda_available = false;
    runtime.runtime.mps_available = true;
    runtime.runtime.gpus = [];
    render(<EnvironmentManagerPanel />);
    expect(await screen.findByText('Apple MPS')).toBeInTheDocument();
    expect(screen.getByText('Apple GPU')).toBeInTheDocument();
    expect(screen.queryByText(/但无法使用 CUDA/)).not.toBeInTheDocument();
  });

  it('renders the English controls without Chinese fallbacks', async () => {
    await i18n.changeLanguage('en');
    render(<EnvironmentManagerPanel />);
    expect(await screen.findByText('Runtime and compute backends')).toBeInTheDocument();
    expect(screen.queryByLabelText('Default attention for new jobs')).not.toBeInTheDocument();
    expect(screen.getByTestId('environment-manager').textContent).not.toMatch(/[\u4e00-\u9fff]/);
  });
});

it('reports Torch multi-device capability separately from single-device trainer support', async () => {
  render(<EnvironmentManagerPanel/>);
  const info=await screen.findByTestId('environment-training-devices');
  expect(info).toHaveTextContent('PyTorch 可用 CUDA 显卡：2 张');
  expect(info).toHaveTextContent('单任务多卡训练尚未接入');
  expect(info).toHaveTextContent('当前平台不适用，不影响单卡训练');
});
