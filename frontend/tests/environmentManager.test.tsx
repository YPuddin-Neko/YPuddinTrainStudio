import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { File as NodeFile } from 'node:buffer';
import { EnvironmentManagerPanel } from '../src/components/EnvironmentManagerPanel';
import type { DtkCatalog, DtkWheel } from '../src/components/DtkWheelPicker';
import type { WindowsAttentionCatalog } from '../src/components/WindowsAttentionWheelPicker';
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
    http.get('/api/environment/dtk/wheels', () => HttpResponse.json(dtkCatalog())),
    http.get('/api/environment/windows/wheels', () => HttpResponse.json(windowsCatalog())),
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
    expect(screen.queryByRole('button', { name: '检查安装条件' })).not.toBeInTheDocument();
    expect(screen.queryByTestId('environment-operations')).not.toBeInTheDocument();
    expect(screen.queryByText('环境操作记录')).not.toBeInTheDocument();
    expect(within(screen.getByRole('navigation')).getAllByRole('button')).toHaveLength(3);
  });
  it('shows the target runtime and requires plan review before mutation', async () => {
    render(<EnvironmentManagerPanel />);
    expect(await screen.findByText('2.5.1+cu128')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '检查安装条件' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('解释器与显卡诊断'));
    expect(screen.getByText('C:\\Studio\\venv\\Scripts\\python.exe')).toBeVisible();
    const row = screen.getByTestId('environment-package-xformers');
    fireEvent.click(within(row).getByRole('button', { name: '安装' }));
    fireEvent.click(screen.getByText('手动版本与 wheel'));
    fireEvent.change(screen.getByLabelText('xformers 版本'), { target: { value: '1.2.3' } });
    fireEvent.click(screen.getByRole('button', { name: '检查安装条件' }));
    await waitFor(() => expect(create).toHaveBeenCalledWith({ package: 'xformers', action: 'install', version: '1.2.3' }));
    const confirm = await screen.findByRole('button', { name: '确认安装' });
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
    expect(screen.getByRole('button', { name: '检查安装条件' })).toBeDisabled();
    expect(screen.getByText(/先选择兼容构建或上传 wheel/)).toBeInTheDocument();
    expect(screen.queryByLabelText('flash-attn 版本')).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('flash-attn wheel'), { target: { files: [new File(['wheel'], 'flash_attn.whl')] } });
    await waitFor(() => expect(screen.getByRole('button', { name: '检查安装条件' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: '检查安装条件' }));
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
    fireEvent.click(screen.getByRole('button', { name: '检查安装条件' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Exact version required');
    expect(screen.getByLabelText('xformers 版本')).toHaveValue('invalid');
  });

  it('keeps finished logs only for the current visit without a dismiss or history button', async () => {
    operations = [operation('xformers', 'planning')];
    const first = render(<EnvironmentManagerPanel />);
    await screen.findByLabelText('安装日志');
    operations = [{ ...operations[0], status: 'failed', error: 'Protected Torch dependency conflict' }];
    fireEvent.click(screen.getByRole('button', { name: '重新检测' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Protected Torch dependency conflict');
    expect(screen.getByRole('heading', {name:'安装日志'})).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '关闭结果' })).not.toBeInTheDocument();
    expect(apply).not.toHaveBeenCalled();
    first.unmount();
    render(<EnvironmentManagerPanel />);
    await screen.findByTestId('environment-package-xformers');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByTestId('environment-operations')).not.toBeInTheDocument();
    expect(screen.queryByText('安装记录')).not.toBeInTheDocument();
    expect(screen.queryByText(/历史记录/)).not.toBeInTheDocument();
    expect(operations[0].status).toBe('failed');
  });

  it('omits completed history and failures superseded by a completed operation', async () => {
    operations = [{ ...operation('xformers', 'completed'), id: 'new' }, { ...operation('flash-attn', 'failed'), id: 'old', error: 'Old error' }];
    render(<EnvironmentManagerPanel />);
    await screen.findByTestId('environment-package-xformers');
    expect(screen.queryByTestId('environment-operations')).not.toBeInTheDocument();
    expect(screen.queryByText('Old error')).not.toBeInTheDocument();
  });

  it('renders PyTorch operations in the shared installation log below the version controls', async () => {
    server.use(http.get('/api/environment/torch',()=>HttpResponse.json({builds:[{id:'torch-cu128',label:'PyTorch 2.11.0 · CUDA 12.8',supported:true,recommended:true}],operations:[{id:'torch_install',build_id:'torch-cu128',status:'installing',phase:'installing_pytorch',logs:['Installing PyTorch in an isolated environment'],error:null,environment_id:null,dismissed_at:null}],selected_environment:null,disk_free_bytes:100*1024**3})));
    render(<EnvironmentManagerPanel/>);
    const log=await screen.findByLabelText('PyTorch 安装日志');
    const section=screen.getByTestId('environment-operations');
    expect(section).toContainElement(log);
    expect(section).toContainElement(screen.getByRole('progressbar',{name:'PyTorch 安装进度'}));
    expect(section).toContainElement(screen.getByRole('button',{name:'取消安装'}));
    expect(document.getElementById('environment-torch')).not.toContainElement(log);
    expect(screen.getByRole('heading',{name:'安装日志'})).toBeInTheDocument();
    expect(screen.getByRole('navigation',{name:'当前页章节'})).toHaveTextContent('安装日志');
  });
  it.each([['uninstall','正在卸载','确认卸载'],['repair','正在重装','确认重装']])('describes %s using its actual action', async (action,progress,confirm) => {
    operations=[{...operation('xformers','ready'),action}];
    render(<EnvironmentManagerPanel/>);
    expect(await screen.findByRole('button',{name:confirm})).toBeInTheDocument();
    operations=[{...operations[0],status:'installing'}];
    fireEvent.click(screen.getByRole('button',{name:'重新检测'}));
    expect(await screen.findByText(progress)).toBeInTheDocument();
    expect(screen.queryByText('下载并安装')).not.toBeInTheDocument();
  });

  it('restores an in-progress install and logs without exposing an interrupt action', async () => {
    operations = [operation('flash-attn', 'installing')];
    render(<EnvironmentManagerPanel />);
    expect(await screen.findByLabelText('安装日志')).toHaveTextContent('Torch unchanged');
    expect(screen.getByText('正在安装')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '取消安装' })).not.toBeInTheDocument();
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
  expect(info).toHaveTextContent('PyTorch 可用显卡：2 张');
  expect(info).toHaveTextContent('当前环境使用单设备训练');
  expect(info).toHaveTextContent('当前平台不适用，不影响单设备训练');
});

it('identifies DTK and describes multi-GPU training without NVIDIA requirements', async () => {
  Object.assign(runtime.runtime, { environment_profile: 'linux-dtk', compute_backend: 'hip', hip_runtime: '6.3.42134', cuda_runtime: null, platform: 'Linux', multi_gpu_training: true, gpus: [{name: 'HYGON BW', device: 'cuda:0', mem_total_mb: 65520}] });
  render(<EnvironmentManagerPanel/>);
  expect(await screen.findByText('Linux DTK')).toBeInTheDocument();
  expect(screen.getByText('HIP 运行时').parentElement).toHaveTextContent('6.3.42134');
  expect(screen.getByText('计算后端').parentElement).toHaveTextContent('DTK / HIP');
  const info = screen.getByTestId('environment-training-devices');
  expect(info).toHaveTextContent('可在训练参数中选择显卡数量');
  expect(info).toHaveTextContent('数据并行（DDP）分担训练数据，每张卡保留完整模型');
  expect(info).toHaveTextContent('主模型全参训练可选择显存分片（FSDP），分担参数、梯度和优化器状态');
  expect(info).toHaveTextContent('NCCL 兼容接口');
  expect(screen.queryByText('NVIDIA 显卡计算')).not.toBeInTheDocument();
  expect(screen.queryByText('CUDA 版本')).not.toBeInTheDocument();
});

it('explains Windows DDP startup verification without claiming FSDP or verified CUDA support', async () => {
  Object.assign(runtime.runtime, {multi_gpu_training:true, gloo_available:true, multi_gpu_backend:'gloo', multi_gpu_probe_required:true});
  render(<EnvironmentManagerPanel/>);
  const info = await screen.findByTestId('environment-training-devices');
  expect(info).toHaveTextContent('每次启动时检查所选显卡的 Gloo 通信，通过后才加载训练模型');
  expect(info).toHaveTextContent('当前不支持 Windows 原生显存分片（FSDP）');
  expect(info).toHaveTextContent('实际可用性会在任务启动时检查');
  expect(info).not.toHaveTextContent('主模型全参训练可选择显存分片');
});

function dtkCatalog(): DtkCatalog {
  return {source_url:'https://download.sourcefind.cn:65024/4/main/', runtime:{environment_profile:'linux-dtk',torch:'2.4.1+das.opt1.dtk25041',python:'3.10.14',dtk:'25.04.1',machine:'x86_64'},reason:null,
    wheels:[{id:'vendor-flash',package:'flash-attn',version:'2.6.3+dtk25041',filename:'flash_attn_vendor.whl',url:'https://download.sourcefind.cn:65024/file/example.whl',size_bytes:12000000,sha256:'a'.repeat(64),dtk:'25.04.1',torch:'2.4.1',python_tag:'cp310',platform_tag:'linux_x86_64',compatible:true,reason:null},
      {id:'wrong-torch',package:'flash-attn',version:'2.7.4+dtk2604',filename:'flash_attn_newer.whl',url:'https://download.sourcefind.cn:65024/file/other.whl',size_bytes:18000000,sha256:'b'.repeat(64),dtk:'26.04',torch:'2.7.1',python_tag:'cp310',platform_tag:'linux_x86_64',compatible:false,reason:'Torch 版本不匹配'}] as DtkWheel[]};
}
function useDtkRuntime() {
  Object.assign(runtime.runtime, {environment_profile:'linux-dtk',compute_backend:'hip',hip_runtime:'6.3',cuda_runtime:null,platform:'Linux'});
  const flash = runtime.packages.find(item=>item.name==='flash-attn')!;
  Object.assign(flash,{supported:false,reason:'requires_dtk_wheel',wheel_required:true});
}

it('offers the matching vendor build for an actually failed DTK SDPA probe', async () => {
  useDtkRuntime();
  Object.assign(runtime, { sdpa: { status: 'failed', reason: 'hip_sdpa_flash_library_missing', error: 'No matching libraries found for flash_attn_2_cuda*.so', detail: 'native diagnostic', device: 'cuda:0', checked_at: 1 } });
  server.use(http.get('/api/environment/dtk/wheels', () => HttpResponse.json(dtkCatalog())));
  render(<EnvironmentManagerPanel/>);
  const sdpa = await screen.findByTestId('environment-sdpa');
  expect(sdpa).toHaveTextContent('当前计算路径不可用');
  expect(sdpa).toHaveTextContent('重启后重新检测');
  fireEvent.click(within(sdpa).getByRole('button', { name: '查看匹配的 FlashAttention 包' }));
  expect(await screen.findByTestId('dtk-wheels-flash-attn')).toBeInTheDocument();
  expect(create).not.toHaveBeenCalled();
});

it('does not claim untested DTK SDPA has passed while another job owns the GPU', async () => {
  useDtkRuntime();
  Object.assign(runtime, { probe_deferred: true, running_jobs: true, sdpa: { status: 'not_tested', reason: null } });
  render(<EnvironmentManagerPanel/>);
  const sdpa = await screen.findByTestId('environment-sdpa');
  expect(sdpa).toHaveTextContent('任务运行中，检测已延后');
  expect(sdpa).not.toHaveTextContent('已通过前向与反向检测');
});

it('keeps complete long DTK versions and kernel status when opening extension management', async () => {
  useDtkRuntime();
  const versions = {
    xformers: '0.0.33+das.opt1.dtk2604.torch251',
    'flash-attn': '2.8.3+das.opt1.dtk2604.torch271',
  };
  for (const [name, version] of Object.entries(versions)) {
    Object.assign(runtime.packages.find(item => item.name === name)!, {
      version, supported: true, reason: 'supported', available: true, importable: true, kernel_tested: true,
    });
  }
  render(<EnvironmentManagerPanel/>);
  const flash = await screen.findByTestId('environment-package-flash-attn');
  fireEvent.click(within(flash).getByRole('button', { name: '管理' }));
  await screen.findByTestId('dtk-wheels-flash-attn');
  for (const [name, version] of Object.entries(versions)) {
    const row = screen.getByTestId(`environment-package-${name}`);
    expect(within(row).getByText(version, { exact: true })).toBeVisible();
    expect(within(row).getByText('已通过 DTK / HIP 内核检测', { exact: true })).toBeVisible();
    expect(within(row).getByRole('button', { name: '管理' })).toBeEnabled();
  }
  expect(create).not.toHaveBeenCalled();
  expect(apply).not.toHaveBeenCalled();
});

it('selects an official DTK wheel by server-issued ID and leaves incompatible builds unselectable', async () => {
  useDtkRuntime();
  server.use(http.get('/api/environment/dtk/wheels',()=>HttpResponse.json(dtkCatalog())));
  render(<EnvironmentManagerPanel/>);
  const row=await screen.findByTestId('environment-package-flash-attn');
  fireEvent.click(within(row).getByRole('button',{name:'安装'}));
  const choice=await screen.findByRole('combobox',{name:'DTK 适配版本'});
  expect(screen.queryByLabelText('flash-attn 版本')).not.toBeInTheDocument();
  expect(screen.getByRole('button',{name:'检查安装条件'})).toBeDisabled();
  fireEvent.click(choice);
  fireEvent.click(screen.getByRole('option',{name:/2.6.3\+dtk25041/}));
  expect(screen.queryByRole('option',{name:/2.7.4/})).not.toBeInTheDocument();
  fireEvent.click(screen.getByText('包信息与运行要求'));
  expect(screen.getByText(/显卡检测只验证常规 FlashAttention 运算/)).toBeVisible();
  fireEvent.click(screen.getByText('查看其他版本与不匹配原因'));
  expect(screen.getByText('Torch 版本不匹配')).toBeVisible();
  fireEvent.click(screen.getByRole('button',{name:'下载并检查安装包'}));
  await waitFor(()=>expect(create).toHaveBeenCalledWith({package:'flash-attn',action:'install',vendor_wheel_id:'vendor-flash'}));
  expect(apply).not.toHaveBeenCalled();
});

it('keeps offline upload available when no official DTK build matches', async () => {
  useDtkRuntime();
  const catalog=dtkCatalog();catalog.wheels=catalog.wheels.map(item=>({...item,compatible:false,reason:'Torch 版本不匹配'}));
  server.use(http.get('/api/environment/dtk/wheels',()=>HttpResponse.json({...catalog,reason:'no_matching_vendor_build'})));
  render(<EnvironmentManagerPanel/>);
  const row=await screen.findByTestId('environment-package-flash-attn');
  fireEvent.click(within(row).getByRole('button',{name:'安装'}));
  expect(await screen.findByText(/官方目录中暂未找到与当前环境匹配的版本/)).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'检查安装条件'})).toBeDisabled();
  expect(screen.getByText('上传 wheel')).toBeVisible();
  const input = screen.getByLabelText('flash-attn wheel');
  expect(input).not.toBeDisabled();
  expect(input).not.toBeVisible();
  const choose = vi.spyOn(input, 'click');
  fireEvent.click(screen.getByRole('button', {name:'上传 wheel'}));
  expect(choose).toHaveBeenCalledOnce();
});

it('shows measured DTK download progress and stops showing stale speed after failure', async () => {
  operations=[{...operation('flash-attn','planning'),phase:'download',downloaded_bytes:4000000,total_bytes:10000000,bytes_per_second:2000000,eta_seconds:3} as ReturnType<typeof operation>];
  const view=render(<EnvironmentManagerPanel/>);
  const progress=await screen.findByRole('progressbar',{name:'适配包下载进度'});
  expect(progress).toHaveAttribute('aria-valuenow','40');
  expect(screen.getByText('2.0 MB/s')).toBeInTheDocument();
  expect(screen.getByText('预计剩余 3s')).toBeInTheDocument();
  expect(screen.getByRole('button',{name:'取消下载'})).toBeEnabled();
  operations[0].status='failed';operations[0].error='Network disconnected';
  fireEvent.click(screen.getByRole('button',{name:'重新检测'}));
  expect(await screen.findByText('Network disconnected')).toBeInTheDocument();
  expect(screen.getByRole('progressbar',{name:'适配包下载进度'})).toHaveAttribute('aria-valuenow','40');
  expect(screen.queryByText('2.0 MB/s')).not.toBeInTheDocument();
  view.unmount();render(<EnvironmentManagerPanel/>);
  await screen.findByTestId('environment-package-xformers');
  expect(screen.queryByText('Network disconnected')).not.toBeInTheDocument();
});

it('describes Python-only vendor builds as pending device validation rather than verified compatibility', async () => {
  useDtkRuntime();
  const catalog=dtkCatalog();
  catalog.runtime.torch='2.5.1+dtk25041';
  catalog.wheels=[{...catalog.wheels[0],id:'vendor-xfs-python',package:'xformers',version:'0.0.33+dtk2604',dtk:'26.04 (vendor label; Python-only)',torch:'>=2.5',declared_torch:'>=2.1.0',python_tag:'py3',platform_tag:'any',binary:false,validation:'kernel_probe_required',requires_packages:['flash-attn>=2.6.1']}];
  server.use(http.get('/api/environment/dtk/wheels',()=>HttpResponse.json(catalog)));
  render(<EnvironmentManagerPanel/>);
  const row=await screen.findByTestId('environment-package-xformers');
  fireEvent.click(within(row).getByRole('button',{name:'安装'}));
  fireEvent.click(await screen.findByRole('combobox',{name:'DTK 适配版本'}));
  expect(screen.queryByRole('option',{name:'选择适配版本'})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('option',{name:/0.0.33/}));
  expect(screen.getByText(/版本要求已满足/)).toHaveTextContent('安装完成后会测试扩展能否在当前显卡上运行');
  fireEvent.click(screen.getByText('包信息与运行要求'));
  expect(screen.getByText(/此包提供纯 Python 接口/)).toHaveTextContent('仍需检查依赖并通过显卡检测');
  expect(screen.getByText('发布目录 DTK 标签').parentElement).toHaveTextContent('26.04');
  expect(screen.getByText('PyTorch 要求').parentElement).toHaveTextContent('>=2.5');
  expect(screen.getByText('安装包声明的依赖').parentElement).toHaveTextContent('>=2.1.0');
  expect(screen.getByText('需要已安装').parentElement).toHaveTextContent('flash-attn>=2.6.1');
  expect(screen.getByText(/DTK 25.04.1 · PyTorch/)).toBeInTheDocument();
});

it('replaces empty CUDA runtime controls with matched DTK guidance and direct manual downloads', async () => {
  useDtkRuntime();
  const catalog = dtkCatalog();
  Object.assign(catalog.runtime, {distribution:'Ubuntu 22.04.5 LTS',distribution_version:'22.04',installed_dtk:'25.04.1'});
  catalog.guidance = {toolkit_source_url:'https://download.sourcefind.cn:65024/1/main',driver_source_url:'https://download.sourcefind.cn:65024/6/main',compatibility_source_url:'https://download.sourcefind.cn:65024/file/1/compatibility.md',driver_version:'6.3.31-V1.5.3.beta',driver_verification:'manual_confirmation_required',current_stack_reason:'torch24_transformers5_diffusers040_conflict',
    recommendation:{dtk:'26.04',toolkit_url:'https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/runtime.tar.gz',toolkit_checksum_url:'https://download.sourcefind.cn:65024/file/1/DTK-26.04/Ubuntu22.04/runtime.tar.gz.md5',python_tag:'cp311',minimum_driver:'6.3.30-V1.4.1a',status:'candidate_requires_validation',reason:'matched_vendor_metadata_requires_driver_and_training_validation',wheels:[{package:'torch',version:'2.7.1+dtk2604',url:'https://download.sourcefind.cn:65024/file/4/pytorch/torch.whl'}]}};
  const torchRequest = vi.fn();
  server.use(http.get('/api/environment/dtk/wheels',()=>HttpResponse.json(catalog)),http.get('/api/environment/torch',()=>{torchRequest();return HttpResponse.json({});}));
  render(<EnvironmentManagerPanel/>);
  const panel = await screen.findByTestId('dtk-runtime-guidance');
  expect(await within(panel).findByText(/Ubuntu 22.04.5 LTS/)).toBeInTheDocument();
  expect(within(panel).getByText('6.3.31-V1.5.3.beta')).toBeInTheDocument();
  expect(within(panel).getByText(/DTK 26.04 要求驱动/)).toHaveTextContent('安装前请确认当前驱动是否兼容');
  expect(within(panel).getByRole('link',{name:'下载 DTK 26.04'})).toHaveAttribute('href',catalog.guidance.recommendation!.toolkit_url);
  expect(within(panel).getByRole('link',{name:'手动下载 torch 2.7.1+dtk2604'})).toHaveAttribute('href',catalog.guidance.recommendation!.wheels[0].url);
  expect(within(panel).getByRole('link',{name:'DTK 版本目录'})).toHaveAttribute('href','https://download.sourcefind.cn:65024/1/main');
  expect(within(panel).getByRole('link',{name:'驱动下载目录'})).toHaveAttribute('href','https://download.sourcefind.cn:65024/6/main');
  expect(screen.queryByRole('combobox',{name:'选择 PyTorch 版本'})).not.toBeInTheDocument();
  expect(screen.queryByText(/12 GiB/)).not.toBeInTheDocument();
  expect(torchRequest).not.toHaveBeenCalled();
  expect(create).not.toHaveBeenCalled();
});

it('allows a DTK wheel to be manually downloaded and uploaded without sending the automatic selection', async () => {
  useDtkRuntime();
  server.use(http.post('/api/environment/wheels',()=>HttpResponse.json({wheel_id:'manual-dtk',package:'flash-attn',filename:'flash_attn_vendor.whl',version:'2.6.3+dtk25041',sha256:'a'.repeat(64)},{status:201})));
  render(<EnvironmentManagerPanel focusPackage="flash-attn"/>);
  fireEvent.click(await screen.findByRole('combobox',{name:'DTK 适配版本'}));
  expect(screen.queryByRole('option',{name:'选择适配版本'})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('option',{name:/2.6.3\+dtk25041/}));
  expect(screen.getByRole('link',{name:'手动下载此 wheel'})).toHaveAttribute('href',dtkCatalog().wheels[0].url);
  expect(screen.getByText('上传 wheel')).toBeVisible();
  fireEvent.change(screen.getByLabelText('flash-attn wheel'),{target:{files:[new File(['wheel'],'flash_attn_vendor.whl')]}});
  await waitFor(()=>expect(screen.getByRole('button',{name:'检查安装条件'})).toBeEnabled());
  expect(screen.queryByRole('link',{name:'手动下载此 wheel'})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'检查安装条件'}));
  await waitFor(()=>expect(create).toHaveBeenCalledWith({package:'flash-attn',action:'install',version:'2.6.3+dtk25041',wheel_id:'manual-dtk'}));
});

it('retains official manual sources when DTK matching fails', async () => {
  useDtkRuntime();
  server.use(http.get('/api/environment/dtk/wheels',()=>HttpResponse.json({error:{code:'catalog.unavailable',message:'目录暂时不可用'}},{status:503})));
  render(<EnvironmentManagerPanel/>);
  const panel=await screen.findByTestId('dtk-runtime-guidance');
  expect(await within(panel).findByRole('alert')).toHaveTextContent('目录暂时不可用');
  expect(within(panel).getByRole('link',{name:'DTK 版本目录'})).toHaveAttribute('href','https://download.sourcefind.cn:65024/1/main');
  expect(within(panel).getByRole('link',{name:'驱动下载目录'})).toHaveAttribute('href','https://download.sourcefind.cn:65024/6/main');
  expect(within(panel).getByRole('button',{name:'刷新推荐'})).toBeEnabled();
});

function windowsCatalog(compatible = false): WindowsAttentionCatalog {
  return {
    source_url: 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases', release: 'v0.9.6',
    provider: 'mjun0812-community-windows', origin: 'live', checked_at: 1, error: null, reason: compatible ? null : 'no_matching_build',
    runtime: {python:'3.12.10',torch:compatible ? '2.11.0+cu128' : '2.5.1+cu128',cuda_runtime:'12.8',platform:'Windows',machine:'AMD64'},
    wheels: [{id:'mjun0812-cp312',package:'flash-attn',version:'2.8.3+cu128torch2.11',filename:'flash_attn-2.8.3+cu128torch2.11-cp312-cp312-win_amd64.whl',url:'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.6/flash_attn-2.8.3%2Bcu128torch2.11-cp312-cp312-win_amd64.whl',source_url:'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.9.6',provider:'mjun0812-community-windows',size_bytes:250730469,sha256:'a'.repeat(64),torch:'2.11',cuda:'12.8',python_tag:'cp312',platform_tag:'win_amd64',validation:'kernel_probe_required',compatible,reason:compatible ? null : 'torch_version_mismatch'}],
  };
}

it('offers a matching Windows community build and stages a plan before any installation', async () => {
  runtime.runtime.torch = '2.11.0+cu128';
  server.use(http.get('/api/environment/windows/wheels', () => HttpResponse.json(windowsCatalog(true))));
  render(<EnvironmentManagerPanel focusPackage="flash-attn"/>);
  const select = await screen.findByRole('combobox',{name:'Windows FlashAttention 版本'});
  expect(screen.getByText(/并非 FlashAttention 官方提供/)).toBeInTheDocument();
  expect(screen.queryByLabelText('flash-attn 版本')).not.toBeInTheDocument();
  expect(screen.getByRole('button',{name:'检查安装条件'})).toBeDisabled();
  expect(select).toHaveTextContent('选择兼容版本');
  fireEvent.click(select);
  expect(screen.queryByRole('option',{name:'选择兼容版本'})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('option',{name:/2.8.3\+cu128torch2.11/}));
  fireEvent.click(select);
  expect(screen.getAllByRole('option')).toHaveLength(1);
  expect(screen.queryByRole('option',{name:'选择兼容版本'})).not.toBeInTheDocument();
  fireEvent.click(select);
  expect(screen.getByRole('link',{name:'手动下载此 wheel'})).toHaveAttribute('href',windowsCatalog(true).wheels[0].url);
  expect(create).not.toHaveBeenCalled(); expect(apply).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button',{name:'下载并检查安装包'}));
  await waitFor(()=>expect(create).toHaveBeenCalledWith({package:'flash-attn',action:'install',vendor_wheel_id:'mjun0812-cp312'}));
  expect(apply).not.toHaveBeenCalled();
  expect(await screen.findByRole('button',{name:'确认安装'})).toBeEnabled();
});

it('makes catalog failure explicit while allowing bundled matching and manual upload', async () => {
  const catalog = {...windowsCatalog(true),origin:'bundled',error:'无法连接发布页，使用内置目录。'};
  server.use(http.get('/api/environment/windows/wheels',()=>HttpResponse.json(catalog)),http.post('/api/environment/wheels',()=>HttpResponse.json({wheel_id:'offline-win',package:'flash-attn',filename:'offline.whl',version:'2.8.3',sha256:'a'.repeat(64)},{status:201})));
  render(<EnvironmentManagerPanel focusPackage="flash-attn"/>);
  expect(await screen.findByRole('alert')).toHaveTextContent('无法连接发布页');
  expect(screen.getByText(/^Python .*使用内置版本列表/, {selector:'p.settings-note.break-words'})).toBeInTheDocument();
  fireEvent.click(screen.getByRole('combobox',{name:'Windows FlashAttention 版本'}));
  fireEvent.click(screen.getByRole('option',{name:/2.8.3\+cu128torch2.11/}));
  expect(screen.getByRole('button',{name:'上传 wheel'})).toBeEnabled();
  fireEvent.change(screen.getByLabelText('flash-attn wheel'),{target:{files:[new File(['fake'], 'offline.whl')]}});
  await screen.findByText('offline.whl');
  expect(screen.queryByRole('link',{name:'手动下载此 wheel'})).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button',{name:'检查安装条件'}));
  await waitFor(()=>expect(create).toHaveBeenCalledWith({package:'flash-attn',action:'install',version:'2.8.3',wheel_id:'offline-win'}));
});

it('distinguishes no matching build from request failure and keeps the publisher and upload controls', async () => {
  server.use(http.get('/api/environment/windows/wheels',()=>HttpResponse.json(windowsCatalog(false))));
  const view = render(<EnvironmentManagerPanel focusPackage="flash-attn"/>);
  expect(await screen.findByText(/已查询的版本列表中没有适合当前环境的安装包/)).toBeInTheDocument();
  expect(screen.getByText('PyTorch 版本不匹配')).toBeInTheDocument();
  expect(screen.queryByRole('combobox',{name:'Windows FlashAttention 版本'})).not.toBeInTheDocument();
  expect(screen.getByRole('button',{name:'上传 wheel'})).toBeEnabled();
  view.unmount();
  server.use(http.get('/api/environment/windows/wheels',()=>HttpResponse.json({error:{code:'catalog.unavailable',message:'无法读取目录'}},{status:503})));
  render(<EnvironmentManagerPanel focusPackage="flash-attn"/>);
  expect(await screen.findByRole('alert')).toHaveTextContent('无法读取目录');
  expect(screen.getByRole('link',{name:'维护者发布页'})).toHaveAttribute('href',windowsCatalog().source_url);
  expect(screen.getByRole('button',{name:'上传 wheel'})).toBeEnabled();
});
