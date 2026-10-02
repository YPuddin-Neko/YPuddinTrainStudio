import React from 'react';
import { Check, ChevronRight, ExternalLink, RefreshCw } from 'lucide-react';
import { apiClient } from '../../api/client';
import { ApiError, type TrainerUpdateStatus, type TrainerInstallStatus } from '../../api/types';
import Dialog from '../../components/Dialog';
import { LoadingNote } from '../../components/Loading';
import { formatApiError } from '../../utils/errors';
import { formatTime } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import './trainer-updates.css';

type UpdateTarget = { id: string; target_commit: string; before_instance_id: string };
const PENDING_UPDATE_KEY = 'studio.trainer-update.pending';
const RELOADED_UPDATE_KEY = 'studio.trainer-update.reloaded';
function createUpdateRequestId(): string | null {
  try { if (typeof globalThis.crypto?.randomUUID === 'function') return globalThis.crypto.randomUUID(); } catch { /* Embedded browsers may require a secure context. */ }
  try {
    const bytes = globalThis.crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40; bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  } catch { return null; }
}
function readPendingUpdate(): UpdateTarget | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(PENDING_UPDATE_KEY) || 'null') as UpdateTarget | null;
    if (value?.id === sessionStorage.getItem(RELOADED_UPDATE_KEY)) { rememberPendingUpdate(null); return null; }
    return value && typeof value.id === 'string' && /^[a-f0-9]{40}$/i.test(value.target_commit) && typeof value.before_instance_id === 'string' ? value : null;
  } catch { return null; }
}
function rememberPendingUpdate(target: UpdateTarget | null) {
  try { if (target) sessionStorage.setItem(PENDING_UPDATE_KEY, JSON.stringify(target)); else sessionStorage.removeItem(PENDING_UPDATE_KEY); } catch { /* Storage may be disabled. */ }
}
function completedInstall(status: TrainerInstallStatus, expected: UpdateTarget | null): boolean {
  const op = status.operation;
  return !!op && !!expected && op.state === 'succeeded' && op.id === expected.id && op.target_commit === expected.target_commit && op.before_instance_id === expected.before_instance_id
    && status.running_commit === op.target_commit && !!op.before_instance_id && !!op.result_instance_id
    && status.instance_id === op.result_instance_id && status.instance_id !== op.before_instance_id;
}
async function installRequest<T>(request: (signal: AbortSignal) => Promise<T>, signal: AbortSignal, timeout = 5000): Promise<T> {
  const controller = new AbortController();
  let abort: () => void = () => {};
  let timer: number | undefined;
  const interrupted = new Promise<never>((_resolve, reject) => {
    abort = () => { controller.abort(); reject(new DOMException('Aborted', 'AbortError')); };
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
    timer = window.setTimeout(() => { controller.abort(); reject(new TypeError('Failed to fetch')); }, timeout);
  });
  try { return await Promise.race([request(controller.signal), interrupted]); }
  finally { window.clearTimeout(timer); signal.removeEventListener('abort', abort); }
}
function waitForInstallPoll(signal: AbortSignal, delay: number) {
  return new Promise<void>(resolve => {
    const finish = () => { window.clearTimeout(timer); signal.removeEventListener('abort', finish); resolve(); };
    const timer = window.setTimeout(finish, delay);
    signal.addEventListener('abort', finish, { once: true });
    if (signal.aborted) finish();
  });
}
function useTrainerInstall(onReload: () => void) {
  const text = useWorkspaceText();
  const [status, setStatus] = React.useState<TrainerInstallStatus | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [submitting, setSubmitting] = React.useState(false);
  const [watching, setWatching] = React.useState(false);
  const [reconnecting, setReconnecting] = React.useState(false);
  const [unconfirmed, setUnconfirmed] = React.useState(false);
  const [error, setError] = React.useState('');
  const [startFailed, setStartFailed] = React.useState(false);
  const [revision, setRevision] = React.useState(0);
  const expected = React.useRef<UpdateTarget | null>(readPendingUpdate());
  const observed = React.useRef<UpdateTarget | null>(expected.current);
  const posting = React.useRef<AbortController | null>(null);
  const reload = React.useRef(onReload); reload.current = onReload;
  const reloadScheduled = React.useRef<string | null>(null);
  const verified = status ? completedInstall(status, expected.current) : false;

  React.useEffect(() => {
    const controller = new AbortController();
    const signal = controller.signal;
    let disconnectedAt: number | null = null;
    let unconfirmedAt: number | null = null;
    let slowPoll = false;
    setWatching(true); setError(''); setStartFailed(false); setUnconfirmed(false);
    void (async () => {
      while (!signal.aborted) {
        try {
          const next = await installRequest(requestSignal => apiClient.get<TrainerInstallStatus>('/updates/install', { silent: true, signal: requestSignal }), signal);
          if (signal.aborted) return;
          setStatus(next); setLoading(false); setReconnecting(false); disconnectedAt = null;
          const op = next.operation;
          if (op && expected.current?.id === op.id && expected.current.target_commit === op.target_commit && op.before_instance_id) {
            expected.current = { id: op.id, target_commit: op.target_commit, before_instance_id: op.before_instance_id };
            rememberPendingUpdate(expected.current);
          }
          if (op && op.state !== 'succeeded' && !expected.current) {
            observed.current = { id: op.id, target_commit: op.target_commit, before_instance_id: op.before_instance_id };
            if (op.state !== 'failed') {
              expected.current = observed.current;
              rememberPendingUpdate(expected.current);
            }
          }
          const matches = !expected.current || op?.id === expected.current.id && op.target_commit === expected.current.target_commit;
          if (matches && op?.state === 'failed') { expected.current = null; rememberPendingUpdate(null); setUnconfirmed(false); setWatching(false); return; }
          if (matches && completedInstall(next, expected.current)) { setWatching(false); return; }
          if ((!op || op.state === 'succeeded') && !expected.current) { setWatching(false); return; }
          if (!matches || op?.state === 'succeeded') {
            unconfirmedAt ??= Date.now();
            if (Date.now() - unconfirmedAt >= 120000) { setUnconfirmed(true); slowPoll = true; }
          } else { unconfirmedAt = null; slowPoll = false; setUnconfirmed(false); }
        } catch (failure) {
          if (signal.aborted) return;
          setLoading(false);
          if (!expected.current) { setError(formatApiError(failure)); setWatching(false); return; }
          setReconnecting(true);
          disconnectedAt ??= Date.now();
          if (Date.now() - disconnectedAt >= 120000) { setUnconfirmed(true); slowPoll = true; }
        }
        await waitForInstallPoll(signal, slowPoll ? 10000 : 1500);
      }
    })();
    return () => controller.abort();
  }, [revision]);
  React.useEffect(() => () => posting.current?.abort(), []);
  React.useEffect(() => {
    const op = status?.operation;
    if (!verified || !op || expected.current?.id !== op.id || reloadScheduled.current === op.id) return;
    try { if (sessionStorage.getItem(RELOADED_UPDATE_KEY) === op.id) { rememberPendingUpdate(null); return; } } catch { /* Storage may be disabled. */ }
    reloadScheduled.current = op.id;
    const timer = window.setTimeout(() => {
      try { sessionStorage.setItem(RELOADED_UPDATE_KEY, op.id); } catch { /* Storage may be disabled. */ }
      rememberPendingUpdate(null);
      reload.current();
    }, 1500);
    return () => { window.clearTimeout(timer); reloadScheduled.current = null; };
  }, [verified, status]);
  const start = async (commit: string) => {
    if (posting.current || !status?.can_apply || !status.instance_id) return;
    const id = createUpdateRequestId();
    if (!id) { setStartFailed(true); setError(text('当前浏览器无法创建更新请求，请换用其他浏览器。', 'This browser cannot create an update request. Use another browser.')); return; }
    const target = { id, target_commit: commit, before_instance_id: status.instance_id };
    expected.current = target; observed.current = target; rememberPendingUpdate(target);
    const controller = new AbortController(); posting.current = controller;
    setSubmitting(true); setError(''); setStartFailed(false); setReconnecting(false); setUnconfirmed(false);
    try {
      const next = await installRequest(signal => apiClient.post<TrainerInstallStatus>('/updates/install', { target_commit: commit, request_id: target.id }, { silent: true, signal }), controller.signal, 15000);
      if (controller.signal.aborted) return;
      setStatus(next); setRevision(value => value + 1);
    } catch (failure) {
      if (controller.signal.aborted) return;
      if (failure instanceof ApiError && (failure.status < 500 || failure.code === 'updates.start_failed')) {
        expected.current = null; rememberPendingUpdate(null); setStartFailed(true); setReconnecting(false);
        const reason = failure.details?.reason;
        const message = formatApiError(failure);
        setError(typeof reason === 'string' && reason.trim() && !message.includes(reason) ? `${message}\n${reason}` : message);
      } else {
        setReconnecting(true); setRevision(value => value + 1);
      }
    } finally { if (!controller.signal.aborted) setSubmitting(false); if (posting.current === controller) posting.current = null; }
  };
  const operation = status?.operation && observed.current?.id === status.operation.id && observed.current.target_commit === status.operation.target_commit ? status.operation : null;
  const refresh = React.useCallback(() => setRevision(value => value + 1), []);
  const busy = submitting || watching && !!expected.current || !!status?.operation && !['failed', 'succeeded'].includes(status.operation.state)
    || !!expected.current && !verified;
  return { status, operation, loading, submitting, watching, reconnecting, unconfirmed, error, startFailed, verified, busy, start, refresh };
}

function externalUrl(value: string | null | undefined): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value);
    return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : undefined;
  } catch { return undefined; }
}

export default function TrainerUpdates({ onReload = () => window.location.reload() }: { onReload?: () => void } = {}) {
  const text = useWorkspaceText();
  const install = useTrainerInstall(onReload);
  const refreshInstall = install.refresh;
  const [confirmCommit, setConfirmCommit] = React.useState<string | null>(null);
  const [data, setData] = React.useState<TrainerUpdateStatus | null>(null);
  const [loading, setLoading] = React.useState<'read' | 'check' | null>('read');
  const [error, setError] = React.useState('');
  const [now, setNow] = React.useState(Date.now);
  const request = React.useRef<AbortController | null>(null);
  const load = React.useCallback(async (check: boolean) => {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(check ? 'check' : 'read');
    setError('');
    try {
      const options = { silent: true, signal: controller.signal };
      const result = check
        ? await apiClient.post<TrainerUpdateStatus>('/updates/check', undefined, options)
        : await apiClient.get<TrainerUpdateStatus>('/updates', options);
      if (!controller.signal.aborted) { setData(result); setNow(Date.now()); if (check) refreshInstall(); }
    } catch (failure) {
      if (!controller.signal.aborted) setError(formatApiError(failure));
    } finally {
      if (!controller.signal.aborted) setLoading(null);
      if (request.current === controller) request.current = null;
    }
  }, [refreshInstall]);
  React.useEffect(() => { void load(false); return () => request.current?.abort(); }, [load]);

  const retryAt = data?.retry_at;
  const coolingDown = !!retryAt && retryAt * 1000 > now;
  React.useEffect(() => {
    if (!retryAt || retryAt * 1000 <= now) return;
    const timer = window.setTimeout(() => setNow(Date.now()), Math.min(2_147_483_647, Math.max(0, retryAt * 1000 - Date.now() + 50)));
    return () => window.clearTimeout(timer);
  }, [retryAt, now]);

  const state = loading === 'check' ? 'checking' : error ? 'error' : data?.state ?? 'unchecked';
  const stateLabel = {
    unchecked: text('尚未检查更新', 'Not checked yet'),
    checking: text('正在检查更新…', 'Checking for updates…'),
    current: text('当前提交已是最新', 'Current commit is up to date'),
    available: text('有新的源码版本', 'New source version available'),
    ahead: text('当前源码领先于远端', 'Local source is ahead of the remote'),
    diverged: text('当前源码与远端已分叉', 'Local and remote source have diverged'),
    unknown: text('无法比较当前源码版本', 'Cannot compare the current source version'),
    error: data ? text('检查更新失败', 'Update check failed') : text('无法读取版本信息', 'Could not read version information'),
  }[state];
  const latest = data?.latest;
  const repositoryUrl = externalUrl(data?.repository_url);
  const checkErrors: Record<string, string> = {
    network: text('无法连接 GitHub，请检查网络或代理后重试。', 'Cannot connect to GitHub. Check your network or proxy, then retry.'),
    rate_limit: coolingDown ? text(`GitHub 请求次数已达上限，可在 ${formatTime(retryAt!)} 后重试。`, `GitHub request limit reached. Retry after ${formatTime(retryAt!)}.`) : text('GitHub 请求次数已达上限，请稍后重试。', 'GitHub request limit reached. Try again later.'),
    invalid_response: text('GitHub 返回的版本信息无效，请稍后重试。', 'GitHub returned invalid version information. Try again later.'),
    unavailable: text('暂时无法取得远端版本，请稍后重试。', 'Remote version information is unavailable. Try again later.'),
  };
  const failure = error || (data?.error_code && checkErrors[data.error_code]) || data?.error;
  const sourceLabel = data ? { git: text('Git 源码', 'Git checkout'), package: text('源码包', 'Source package'), unknown: text('未知', 'Unknown') }[data.current.source] : '';

  const op = install.operation;
  const operationLabels: Record<string, string> = {
    preparing: text('正在准备更新…', 'Preparing update…'), downloading: text('正在下载源码…', 'Downloading source…'),
    building: text('正在构建前端…', 'Building frontend…'), applying: text('正在应用更新…', 'Applying update…'),
    installing: text('正在安装依赖…', 'Installing dependencies…'), restarting: text('正在重启服务…', 'Restarting service…'),
    succeeded: install.verified ? text('更新完成', 'Update complete') : text('正在确认新版本…', 'Verifying the new version…'),
    failed: text('更新失败', 'Update failed'),
  };
  const blockedReasons: Record<string, string> = {
    start_with_studio_launcher: text('请用项目启动脚本运行服务后更新。', 'Start the service with the project launcher to update.'),
    restart_in_progress: text('服务正在重启。', 'The service is restarting.'),
    training_or_data_worker_running: text('训练或数据任务运行中，完成后可更新。', 'Wait for training or data tasks to finish before updating.'),
    extension_operation_running: text('等待扩展安装完成。', 'Wait for extension installation to finish.'),
    torch_operation_running: text('等待 PyTorch 安装完成。', 'Wait for PyTorch installation to finish.'),
    model_download_running: text('等待模型下载完成或取消下载。', 'Finish or cancel model downloads first.'),
    data_operation_running: text('等待数据处理完成。', 'Wait for data processing to finish.'),
    version_operation_running: text('等待版本操作完成。', 'Wait for the version operation to finish.'),
    update_in_progress: text('更新正在进行。', 'An update is in progress.'),
    current_version_unknown: text('无法识别当前源码版本，请先更换为带版本记录的源码包。', 'The current source version is unknown. Use a source package with version information.'),
    local_changes: text('当前源码有本地改动，请处理后再更新。', 'Resolve local source changes before updating.'),
    version_changed: text('源码版本已变化，请重新检查更新。', 'The source version changed. Check for updates again.'),
    update_check_required: text('请先检查更新。', 'Check for updates first.'),
  };
  const reason = install.status?.reason;
  const updateComplete = install.verified && op?.target_commit === latest?.commit;
  const mayUpdate = data?.state === 'available' && !!latest && !!install.status?.can_apply && !install.busy && !updateComplete && !install.loading && !install.watching && !install.error && !loading && !error;
  const showOperation = !!op || install.busy;
  const restartExpected = !!op && ['applying', 'installing', 'restarting'].includes(op.state);
  const operationLabel = install.unconfirmed ? install.reconnecting && restartExpected ? text('仍在等待训练器重启…', 'Still waiting for the trainer to restart…') : text('暂未确认更新结果', 'Update result not yet confirmed')
    : install.reconnecting ? text('正在重新连接服务…', 'Reconnecting to the service…')
      : op ? operationLabels[op.state] : text('正在准备更新…', 'Preparing update…');

  const installPanel = <>
        {install.error && <div role="alert" className="trainer-update-error trainer-install-reason">{install.startFailed && <strong className="trainer-install-failure-title">{text('更新失败', 'Update failed')}</strong>}{install.error}<button type="button" className="ui-btn ui-btn-sm" onClick={install.refresh}>{text('刷新状态', 'Refresh status')}</button></div>}
        {showOperation && <div className="trainer-install" data-state={op?.state}>
          <div className="trainer-install-heading"><span role="status">{operationLabel}</span>{op && <code title={op.target_commit}>{op.target_commit.slice(0, 8)}</code>}</div>
          {op?.state === 'failed' && <p role="alert" className="trainer-update-error">{op.error?.trim() || op.message?.trim() || text('请查看更新日志或启动窗口中的错误信息。', 'Check the update log or launcher window for error details.')}{op.rolled_back && <span>{text(' 已恢复原版源码。', ' The previous source files were restored.')}</span>}</p>}
          {install.unconfirmed && <p className="settings-note">{text('请查看启动窗口，页面会继续尝试连接。', 'Check the launcher window. This page will keep trying to connect.')}</p>}
          {!!op?.log?.length && <details className="trainer-install-log"><summary><ChevronRight size={14} className="disclosure-chevron"/>{text('更新日志', 'Update log')}</summary><pre>{op.log.join('\n')}</pre></details>}
          {(install.unconfirmed || op?.state === 'failed') && <button type="button" className="ui-btn ui-btn-sm" onClick={install.refresh}>{text('刷新状态', 'Refresh status')}</button>}
          {install.verified && <button type="button" className="ui-btn ui-btn-sm" onClick={onReload}>{text('刷新页面', 'Reload page')}</button>}
        </div>}
  </>;

  return <section className="settings-section trainer-updates" aria-labelledby="trainer-updates-title">
    <div className="settings-section-heading">
      <div><h2 id="trainer-updates-title">{text('训练器更新', 'Trainer updates')}</h2></div>
      <button type="button" className="ui-btn" disabled={loading !== null || install.busy || coolingDown} onClick={() => void load(data !== null)}>
        <RefreshCw size={14} className={loading ? 'animate-spin' : undefined} aria-hidden="true"/>
        {loading === 'check' ? text('正在检查…', 'Checking…') : !data && error ? text('重试', 'Retry') : text('检查更新', 'Check for updates')}
      </button>
    </div>
    {!data && loading === 'read' ? <LoadingNote block label={text('正在读取版本信息…', 'Reading version information…')}/> : <>
      <div className="trainer-update-state" data-state={state} role="status"><span>{stateLabel}</span>{data?.checked_at != null && <time dateTime={new Date(data.checked_at * 1000).toISOString()}>{text('上次检查', 'Last checked')} {formatTime(data.checked_at)}</time>}</div>
      {failure && !loading && <p role="alert" className="trainer-update-error">{failure}</p>}
      {data && <>
        {state === 'unknown' && <p className="settings-note trainer-update-note">{data.current.source === 'package' && !data.current.commit
          ? text('此源码包未记录提交版本，无法判断是否最新。', 'This source package has no recorded commit, so its update status is unknown.')
          : text('未能确认当前源码与远端的版本关系。', 'The relationship between the local and remote source could not be determined.')}</p>}
        <div className="trainer-version-grid" aria-busy={loading === 'check'}>
          <section className="trainer-version-card" aria-labelledby="trainer-current-version">
            <h3 id="trainer-current-version">{text('当前版本', 'Current version')}</h3>
            <strong className="trainer-version-number">{data.current.version}</strong>
            <dl><div><dt>{text('提交', 'Commit')}</dt><dd><code title={data.current.commit || undefined}>{data.current.commit?.slice(0, 8) || text('未记录', 'Not recorded')}</code></dd></div>
              {data.current.branch && <div><dt>{text('分支', 'Branch')}</dt><dd>{data.current.branch}</dd></div>}
              <div><dt>{text('来源', 'Source')}</dt><dd>{sourceLabel}</dd></div></dl>
            {data.current.dirty === true && <p className="trainer-local-changes">{text('有本地改动', 'Has local changes')}</p>}
          </section>
          <section className="trainer-version-card" aria-labelledby="trainer-latest-version">
            <h3 id="trainer-latest-version">{state === 'error' && latest ? text('上次获取的源码版本', 'Last known source version') : text('最新源码版本', 'Latest source version')}</h3>
            {latest ? <><strong className="trainer-version-number"><code title={latest.commit}>{latest.commit.slice(0, 8)}</code></strong><p className="trainer-latest-subject">{latest.subject}</p><dl><div><dt>{text('分支', 'Branch')}</dt><dd>{latest.branch}</dd></div><div><dt>{text('提交时间', 'Committed')}</dt><dd><time dateTime={latest.date}>{formatTime(latest.date)}</time></dd></div>{state === 'error' && data.last_success_at != null && <div><dt>{text('获取时间', 'Retrieved')}</dt><dd><time dateTime={new Date(data.last_success_at * 1000).toISOString()}>{formatTime(data.last_success_at)}</time></dd></div>}</dl></>
              : <p className="settings-note trainer-version-empty">{data.state === 'unchecked' ? text('检查后显示。', 'Shown after checking.') : text('暂未取得远端版本。', 'No remote version is available yet.')}</p>}
          </section>
        </div>
        {latest && <div className="trainer-update-actions">
          <button type="button" className="ui-btn ui-btn-primary trainer-update-apply" data-state={updateComplete ? 'succeeded' : install.busy ? 'busy' : undefined} aria-busy={install.busy} disabled={!mayUpdate} onClick={() => setConfirmCommit(latest.commit)}>
            {updateComplete ? <Check size={14} aria-hidden="true"/> : <RefreshCw size={14} className={install.busy ? 'animate-spin' : undefined} aria-hidden="true"/>}
            {updateComplete ? text('更新完成', 'Update complete') : install.busy ? text('正在更新…', 'Updating…') : text('更新并重启', 'Update and restart')}
          </button>
          {repositoryUrl && <a className="ui-btn" href={repositoryUrl} target="_blank" rel="noreferrer"><ExternalLink size={14}/>{text('打开 GitHub', 'Open GitHub')}</a>}
        </div>}
        {latest && reason && !install.busy && <p className="settings-note trainer-install-reason">{blockedReasons[reason] || text('当前无法更新，请稍后重试。', 'Updating is unavailable. Try again later.')}</p>}
        {installPanel}
      </>}
      {!data && installPanel}
    </>}
    {confirmCommit && <Dialog title={text('更新并重启', 'Update and restart')} onClose={() => setConfirmCommit(null)}>
      <p className="trainer-install-confirm">{text('将更新到提交', 'Update to commit')} <code>{confirmCommit.slice(0, 8)}</code>{text('，完成后会重启服务。', ' and restart the service when ready.')}</p>
      <div className="trainer-install-confirm-actions"><button type="button" className="ui-btn" onClick={() => setConfirmCommit(null)}>{text('取消', 'Cancel')}</button><button type="button" className="ui-btn ui-btn-primary" disabled={!mayUpdate || confirmCommit !== latest?.commit} onClick={() => { const commit = confirmCommit; setConfirmCommit(null); void install.start(commit); }}>{text('更新并重启', 'Update and restart')}</button></div>
    </Dialog>}
  </section>;
}
