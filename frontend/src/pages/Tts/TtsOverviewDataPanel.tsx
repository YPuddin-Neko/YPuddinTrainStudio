import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { ArrowRight, AudioLines, CheckCircle2, CircleAlert, Loader2, Play } from 'lucide-react';
import { ttsApi, ttsAudioUrl, type TtsEngine, type TtsSource, type TtsSourceRow, type TtsSplit } from '../../api/tts';
import AudioPlayer from '../../components/AudioPlayer';
import { SlidingIndicator } from '../../components/motion';
import { formatApiError } from '../../utils/errors';
import { formatEta, formatTime } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import { gptSovitsLanguages } from './gptSovitsResults';
import { sourceComplete, sourceName, sourceState } from './ttsOverviewData';

function AudioPreview({ source, dataUrl, refresh }: { source: TtsSource; dataUrl: string; refresh: () => void }) {
  const text = useWorkspaceText();
  const [selection, setSelection] = React.useState<{ id: string; reference: boolean; request: number } | null>(null);
  const { project_id: pid, version_id: vid } = source.scope;
  const rows = useQuery({
    queryKey: ['tts-overview-rows', pid, vid, source.id, source.snapshot_id],
    queryFn: ({ signal }) => ttsApi.rows(pid, vid, source.id, { snapshot_id: source.snapshot_id!, page: 1, page_size: 5 }, signal),
    retry: false,
    refetchOnWindowFocus: false,
  });
  const selected = rows.data?.items.find(row => row.id === selection?.id);
  const selectedUrl = selected && ttsAudioUrl(selection?.reference ? selected.reference_audio_url : selected.audio_url);
  const play = (row: TtsSourceRow, reference = false) => setSelection(value => ({ id: row.id, reference, request: (value?.request || 0) + 1 }));
  const retry = () => { setSelection(null); refresh(); void rows.refetch(); };
  return <>
    <div className="overview-panel-heading"><h3>{source.split === 'train' ? text('训练集预览', 'Training set preview') : text('验证集预览', 'Validation set preview')}</h3><Link className="ui-link" to={dataUrl}>{text('查看全部', 'View all')}<ArrowRight size={13}/></Link></div>
    {rows.isPending ? <p className="overview-muted" role="status"><Loader2 size={14} className="animate-spin"/>{text('正在读取音频与转写…', 'Loading audio and transcripts…')}</p>
      : rows.error ? <div className="overview-inline-error" role="alert"><span>{formatApiError(rows.error)}</span><button type="button" className="ui-btn ui-btn-sm" onClick={retry}>{text('重新读取', 'Reload')}</button></div>
        : rows.data && <>
          <p className="overview-section-detail">{text(`共 ${rows.data.total.toLocaleString()} 条 · 预览 ${rows.data.items.length} 条`, `${rows.data.total.toLocaleString()} rows · ${rows.data.items.length} previewed`)}</p>
          <ul className="tts-overview-clips">{rows.data.items.map(row => <li key={row.id} data-selected={selection?.id === row.id || undefined}>
            <div className="tts-overview-clip-heading"><strong title={row.audio_name || undefined}>{row.audio_name || text(`第 ${row.line} 行`, `Line ${row.line}`)}</strong><button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={!ttsAudioUrl(row.audio_url)} aria-pressed={selection?.id === row.id && !selection.reference} aria-label={text(`试听第 ${row.line} 行音频`, `Preview line ${row.line} audio`)} onClick={() => play(row)}><Play size={13}/>{text('试听', 'Preview')}</button></div>
            <div className="tts-overview-clip-meta"><span>{row.duration_seconds === null ? text('时长未知', 'Duration unknown') : `${row.duration_seconds.toFixed(2)} s`}</span>{row.language && <span>{gptSovitsLanguages.find(([code]) => code === row.language)?.[text('zh', 'en') === 'zh' ? 1 : 2] || row.language}</span>}{row.speaker && <span title={row.speaker}>{row.speaker}</span>}{row.issues.some(issue => issue.severity === 'error') && <span className="tts-overview-invalid">{text('存在问题', 'Has issues')}</span>}</div>
            <p className="tts-overview-transcript" title={row.text || undefined}>{row.text || text('暂无转写', 'No transcript')}</p>
            {row.reference_audio_name && <button type="button" className="ui-link tts-overview-reference" disabled={!ttsAudioUrl(row.reference_audio_url)} onClick={() => play(row, true)} aria-label={text(`试听第 ${row.line} 行参考音频`, `Preview line ${row.line} reference audio`)}>{text('试听参考音频', 'Preview reference audio')}</button>}
          </li>)}</ul>
          {!rows.data.items.length && <p className="overview-section-detail">{text('清单中没有音频条目。', 'The manifest has no audio rows.')}</p>}
        </>}
    {selected && selectedUrl && !rows.error && <div className="tts-overview-player"><strong>{selection?.reference ? selected.reference_audio_name : selected.audio_name}</strong><AudioPlayer src={selectedUrl} label={text(`试听第 ${selected.line} 行${selection?.reference ? '参考音频' : '音频'}`, `Preview line ${selected.line} ${selection?.reference ? 'reference audio' : 'audio'}`)} autoPlay playRequest={selection?.request}
      loadError={text('音频无法读取，请在训练数据中重新检查来源。', 'Audio could not be read. Check the source again in Training data.')}
      playError={text('播放未能开始，请点击播放器重试。', 'Playback could not start. Try the audio player again.')}/></div>}
  </>;
}

export default function TtsOverviewDataPanel({ sources, engine, dataUrl, readOnly, refresh }: { sources: TtsSource[]; engine: TtsEngine; dataUrl: string; readOnly: boolean; refresh: () => void }) {
  const text = useWorkspaceText();
  const [split, setSplit] = React.useState<TtsSplit>('train');
  const showValidation = engine !== 'gpt-sovits-v5' || sources.some(source => source.split === 'validation');
  const selectedSplit = showValidation ? split : 'train';
  const source = sources.find(item => item.split === selectedSplit);
  const complete = sourceComplete(source);
  const summary = complete ? source?.summary : null;
  const count = (value: number | undefined) => value === undefined ? '—' : value.toLocaleString();
  const issuePreview = source?.issues.slice(0, 3) || [];
  const rowsUrl = `${dataUrl}&tts_data=rows&split=${selectedSplit}`;
  const issuesUrl = `${dataUrl}&tts_data=issues&split=${selectedSplit}`;
  const sourceUrl = `${dataUrl}&tts_data=sources&split=${selectedSplit}`;
  return <section className="overview-data-panel tts-overview-data" aria-label={text('版本音频数据', 'Version audio data')}>
    {showValidation && <div className="overview-role-toolbar"><div className="overview-role-switch ui-segmented" role="group" aria-label={text('数据用途', 'Dataset role')}>
      {(['train', 'validation'] as const).map(value => { const item = sources.find(entry => entry.split === value); return <button type="button" key={value} aria-pressed={selectedSplit === value} onClick={() => setSplit(value)}>{value === 'train' ? text('训练集', 'Training set') : text('验证集', 'Validation set')} <strong>{sourceComplete(item) ? item?.summary?.clips_count.toLocaleString() ?? '—' : '—'}</strong></button>; })}
      <SlidingIndicator className="ui-segmented-thumb"/>
    </div></div>}
    {!source ? <div className="overview-data-empty"><span className="overview-data-empty-icon" aria-hidden="true"><AudioLines size={22}/></span><strong>{selectedSplit === 'train' ? text('这个版本还没有训练数据', 'This version has no training data') : text('这个版本还没有验证数据', 'This version has no validation data')}</strong><p>{selectedSplit === 'validation' ? text('登记验证清单后，可查看验证音频与转写。', 'Register a validation manifest to preview its audio and transcripts.') : text('登记音频清单后，这里会显示音频预览、转写与数据检查结果。', 'Register an audio manifest to see previews, transcripts and check results here.')}</p><Link className={`ui-btn${readOnly ? '' : ' ui-btn-primary'}`} to={dataUrl}>{readOnly ? text('查看训练数据', 'View training data') : text('登记音频清单', 'Register audio manifest')}</Link></div>
      : <div className="overview-data-grid">
        <section className="overview-panel overview-gallery tts-overview-gallery" aria-label={text('音频预览', 'Audio preview')}>
          {complete ? <AudioPreview key={`${source.id}:${source.snapshot_id}`} source={source} dataUrl={rowsUrl} refresh={refresh}/> : <><div className="overview-panel-heading"><h3>{selectedSplit === 'train' ? text('训练集预览', 'Training set preview') : text('验证集预览', 'Validation set preview')}</h3></div><div className="tts-overview-preview-unavailable"><AudioLines size={24} aria-hidden="true"/><strong>{sourceState(source, text)}</strong><p>{source.state === 'checking' ? text('检查完成后显示音频与转写。', 'Audio and transcripts appear when the check finishes.') : text('在训练数据中完成检查后，可查看音频与转写。', 'Check the source in Training data to preview audio and transcripts.')}</p><Link className="ui-link" to={sourceUrl}>{text('查看训练数据', 'View training data')}<ArrowRight size={13}/></Link></div></>}
          <div className="overview-source-summary"><Link to={sourceUrl}><strong title={source.path}>{sourceName(source.path)}</strong><span>{source.split === 'train' ? text('训练清单', 'Training manifest') : text('验证清单', 'Validation manifest')}</span><ArrowRight size={13}/></Link></div>
        </section>
        <section className="overview-panel tts-overview-checks" aria-label={text('数据检查', 'Data checks')}><header className="overview-panel-heading"><h3>{text('数据检查', 'Data checks')}</h3><Link className="ui-link" to={issuesUrl}>{text('查看详情', 'View details')}<ArrowRight size={13}/></Link></header>
          <p className="tts-overview-check-state" data-state={source.state}>{source.state === 'checking' ? <Loader2 size={15} className="animate-spin"/> : source.state === 'valid' ? <CheckCircle2 size={15}/> : <CircleAlert size={15}/>}<strong>{sourceState(source, text)}</strong></p>
          <dl className="overview-parameters tts-overview-source-stats"><div><dt>{text('音频总数', 'Total clips')}</dt><dd>{count(summary?.clips_count)}</dd></div><div><dt>{text('音频时长', 'Audio duration')}</dt><dd>{summary ? formatEta(summary.duration_seconds) : '—'}</dd></div><div><dt>{text('有效条目', 'Valid rows')}</dt><dd>{count(summary?.valid_clips_count)}</dd></div><div><dt>{text('无效条目', 'Invalid rows')}</dt><dd>{count(summary?.invalid_count)}</dd></div></dl>
          {source.checked_at !== null && <p className="overview-section-detail tts-overview-check-time">{text('检查时间', 'Checked')} · {formatTime(source.checked_at)}</p>}
          {issuePreview.length > 0 && <ul className="tts-overview-issues">{issuePreview.map((issue, index) => <li key={`${issue.code}:${index}`} data-severity={issue.severity}><span>{issue.severity === 'warning' ? text('警告', 'Warning') : text('错误', 'Error')}</span><p>{issue.message}</p></li>)}</ul>}
          {(source.issues_truncated || (source.issues_total ?? source.issues.length) > issuePreview.length) && <Link className="ui-link tts-overview-issue-link" to={issuesUrl}>{source.issues_total === null ? text('查看全部问题', 'View all issues') : text(`查看全部 ${source.issues_total} 个问题`, `View all ${source.issues_total} issues`)}<ArrowRight size={13}/></Link>}
        </section>
      </div>}
  </section>;
}
