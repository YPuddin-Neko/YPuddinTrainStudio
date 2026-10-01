import React from 'react';
import { ChevronRight, Download, ExternalLink, RefreshCw } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { TrainerUpdateStatus, TrainerCommit as UpdateCommit } from '../../api/types';
import { LoadingNote } from '../../components/Loading';
import { formatApiError } from '../../utils/errors';
import { formatTime } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import './trainer-updates.css';

function externalUrl(value: string | null | undefined): string | undefined {
  if (!value) return undefined;
  try {
    const url = new URL(value);
    return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : undefined;
  } catch { return undefined; }
}

function CommitEntry({ entry }: { entry: UpdateCommit }) {
  const text = useWorkspaceText();
  const url = externalUrl(entry.url);
  const meta = <div className="trainer-commit-meta"><code title={entry.commit}>{entry.commit.slice(0, 8)}</code><span>{entry.author}</span><time dateTime={entry.date}>{formatTime(entry.date)}</time>{url && <a href={url} target="_blank" rel="noreferrer" className="ui-link" aria-label={text(`查看提交 ${entry.commit.slice(0, 8)}`, `View commit ${entry.commit.slice(0, 8)}`)}><ExternalLink size={12}/></a>}</div>;
  return <li className="trainer-commit">
    {entry.body.trim() ? <details>
      <summary><ChevronRight size={14} className="disclosure-chevron" aria-hidden="true"/><span>{entry.subject}</span></summary>
      <p className="trainer-commit-body">{entry.body}</p>
    </details> : <p className="trainer-commit-subject">{entry.subject}</p>}
    {meta}
  </li>;
}

export default function TrainerUpdates() {
  const text = useWorkspaceText();
  const [data, setData] = React.useState<TrainerUpdateStatus | null>(null);
  const [loading, setLoading] = React.useState<'read' | 'check' | null>('read');
  const [error, setError] = React.useState('');
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
      if (!controller.signal.aborted) setData(result);
    } catch (failure) {
      if (!controller.signal.aborted) setError(formatApiError(failure));
    } finally {
      if (!controller.signal.aborted) setLoading(null);
      if (request.current === controller) request.current = null;
    }
  }, []);
  React.useEffect(() => { void load(false); return () => request.current?.abort(); }, [load]);

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
  const downloadUrl = externalUrl(latest?.download_url);
  const repositoryUrl = externalUrl(data?.repository_url);
  const commits = data?.commits ?? [];
  const historyUrl = data?.history_kind === 'updates'
    ? externalUrl(data.compare_url) || repositoryUrl
    : repositoryUrl && latest ? `${repositoryUrl.replace(/\/$/, '')}/commits/${encodeURIComponent(latest.commit)}` : repositoryUrl;
  const checkErrors: Record<string, string> = {
    network: text('无法连接 GitHub，请检查网络或代理后重试。', 'Cannot connect to GitHub. Check your network or proxy, then retry.'),
    rate_limit: text('GitHub 请求次数已达上限，请稍后重试。', 'GitHub request limit reached. Try again later.'),
    invalid_response: text('GitHub 返回的版本信息无效，请稍后重试。', 'GitHub returned invalid version information. Try again later.'),
    unavailable: text('暂时无法取得远端版本，请稍后重试。', 'Remote version information is unavailable. Try again later.'),
  };
  const failure = error || (data?.error_code && checkErrors[data.error_code]) || data?.error;
  const sourceLabel = data ? { git: text('Git 源码', 'Git checkout'), package: text('源码包', 'Source package'), unknown: text('未知', 'Unknown') }[data.current.source] : '';

  return <section className="settings-section trainer-updates" aria-labelledby="trainer-updates-title">
    <div className="settings-section-heading">
      <div><h2 id="trainer-updates-title">{text('训练器更新', 'Trainer updates')}</h2></div>
      <button type="button" className="ui-btn" disabled={loading !== null} onClick={() => void load(data !== null)}>
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
            <h3 id="trainer-latest-version">{text('最新源码版本', 'Latest source version')}</h3>
            {latest ? <><strong className="trainer-version-number"><code title={latest.commit}>{latest.commit.slice(0, 8)}</code></strong><p className="trainer-latest-subject">{latest.subject}</p><dl><div><dt>{text('分支', 'Branch')}</dt><dd>{latest.branch}</dd></div><div><dt>{text('提交时间', 'Committed')}</dt><dd><time dateTime={latest.date}>{formatTime(latest.date)}</time></dd></div></dl></>
              : <p className="settings-note trainer-version-empty">{data.state === 'unchecked' ? text('检查后显示。', 'Shown after checking.') : text('暂未取得远端版本。', 'No remote version is available yet.')}</p>}
          </section>
        </div>
        {latest && <div className="trainer-update-actions">
          {downloadUrl && <a className="ui-btn" href={downloadUrl} target="_blank" rel="noreferrer"><Download size={14}/>{text('下载源码', 'Download source')}</a>}
          {repositoryUrl && <a className="ui-btn" href={repositoryUrl} target="_blank" rel="noreferrer"><ExternalLink size={14}/>{text('打开 GitHub', 'Open GitHub')}</a>}
        </div>}
        <section className="trainer-update-history" aria-labelledby="trainer-history-title">
          <div className="trainer-history-heading"><h3 id="trainer-history-title">{data.history_kind === 'updates' ? text('更新记录', 'Changes since your version') : text('最近提交', 'Recent commits')}</h3>{data.total_commits != null && <span className="settings-note">{text(`共 ${data.total_commits} 条`, `${data.total_commits} commits`)}</span>}</div>
          {commits.length ? <ol className="trainer-commit-list">{commits.map(entry => <CommitEntry key={entry.commit} entry={entry}/>)}</ol>
            : <p className="settings-note trainer-history-empty">{data.state === 'unchecked' ? text('检查后显示提交记录。', 'Check for updates to see commit history.') : data.history_kind === 'updates' && data.state === 'current' ? text('没有新的提交。', 'No new commits.') : text('暂无提交记录。', 'No commit history is available.')}</p>}
          {data.has_more && historyUrl && <a className="ui-link trainer-history-more" href={historyUrl} target="_blank" rel="noreferrer">{text('在 GitHub 查看全部', 'View all on GitHub')}<ExternalLink size={12}/></a>}
        </section>
      </>}
    </>}
  </section>;
}
