import React from 'react';
import { useTranslation } from 'react-i18next';
import { History, Trash2 } from 'lucide-react';
import type { JobCheckpoint } from '../../api/types';
import CopyButton from '../../components/CopyButton';
import { epochAt, epochText, inEpochs, parseEpochQuery } from '../../utils/epochFilter';
import { formatBytes, formatTime } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import EpochSearch from './EpochSearch';
import { CheckpointPagination, CheckpointSelection } from './CheckpointBrowserControls';
import { useCheckpointBrowser } from './useCheckpointBrowser';
import './job-samples.css';

type Kind = 'paused' | 'stopped' | 'manual' | 'epoch' | 'step';
/** A job in these states still reads the resume point it started from, so that point cannot be deleted. */
const HOLDING = ['queued', 'scheduled', 'running', 'pausing', 'cancelling', 'paused'];

/** What saved a resume point, read from its folder name (state-<time>-step<n>[-tag] or the older state-<tag>). */
function pointKind(path: string): Kind {
  const name = path.replace(/\\/g, '/').split('/').pop() || '';
  if (/-paused$/.test(name)) return 'paused';
  if (/-stopped$/.test(name)) return 'stopped';
  if (/-manual$/.test(name)) return 'manual';
  if (/-epoch\d+$/.test(name)) return 'epoch';
  return 'step';
}

/** Full training states that can continue a run, newest first, separate from exported outputs. */
export default function ResumePointList({ points, stepsPerEpoch, loaded, resumeFrom, jobStatus, resuming, canResume, onResume, onDelete, onDeleteMany, deleting = false }: {
  points: JobCheckpoint[]; stepsPerEpoch?: number | null; loaded: boolean; resumeFrom?: string | null; jobStatus?: string; resuming: boolean; canResume: boolean;
  onResume: (point: JobCheckpoint) => void; onDelete: (point: JobCheckpoint) => void; onDeleteMany: (points: JobCheckpoint[]) => void; deleting?: boolean;
}) {
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const [query, setQuery] = React.useState('');
  const ranges = React.useMemo(() => parseEpochQuery(query), [query]);
  const shown = React.useMemo(() => [...points].reverse().filter(point => inEpochs(epochAt(point, stepsPerEpoch), ranges)), [points, stepsPerEpoch, ranges]);
  const isHeld = (point: JobCheckpoint) => !!resumeFrom && resumeFrom === point.path && HOLDING.includes(jobStatus || '');
  const browser = useCheckpointBrowser(shown, 'studio.job.resume.pageSize', point => !isHeld(point));
  const changeQuery = (value: string) => { setQuery(value); browser.resetFilter(); };
  const labels: Record<Kind, string> = {
    paused: text('暂停点', 'Pause point'), stopped: text('停止点', 'Stop point'), manual: text('手动保存', 'Saved manually'),
    epoch: text('按轮保存', 'Per-epoch save'), step: text('按步保存', 'Per-step save'),
  };

  if (!loaded) return <div className="resume-list" aria-busy="true">{Array.from({ length: 3 }, (_, index) => <div key={index} className="resume-point resume-point-skeleton" aria-hidden="true"><span className="ui-skeleton"/><span className="ui-skeleton"/></div>)}</div>;
  if (!points.length) return <div className="sample-empty"><History size={30} aria-hidden="true"/><p>{text('暂无恢复点', 'No resume points yet')}</p><span>{text('开启定期保存恢复点，或暂停训练后，可以从这里继续训练。', 'Turn on periodic resume points, or pause training, to continue from here later.')}</span></div>;

  return <section className="artifact-browser" aria-label={text('恢复点', 'Resume points')}>
    <div className="sample-toolbar">
      <EpochSearch value={query} onChange={changeQuery} label={text('按轮次搜索恢复点', 'Search resume points by epoch')}/>
      <div className="checkpoint-toolbar-controls"><span className="sample-toolbar-count">{ranges && ranges !== 'invalid' ? text(`找到 ${shown.length} 个，共 ${points.length} 个`, `${shown.length} of ${points.length}`) : text(`共 ${points.length} 个`, `${points.length} resume points`)}</span><CheckpointPagination browser={browser} kind="resume" disabled={deleting}/></div>
    </div>
    <CheckpointSelection browser={browser} disabled={deleting} onDelete={onDeleteMany}/>
    {!shown.length ? <div className="sample-empty"><History size={26} aria-hidden="true"/><p>{text('没有符合轮次的恢复点', 'No resume points in these epochs')}</p><button type="button" className="ui-link" onClick={() => changeQuery('')}>{text('清除搜索', 'Clear search')}</button></div>
      : <ul className="resume-list">{browser.visible.map(point => {
        const name = point.path.replace(/\\/g, '/').split('/').pop() || point.path;
        const kind = pointKind(point.path);
        const loss = typeof point.loss === 'number' && Number.isFinite(point.loss) ? String(Number(point.loss.toPrecision(5))) : '—';
        const used = !!resumeFrom && resumeFrom === point.path;
        const held = isHeld(point);
        return <li key={point.path} className="resume-point" aria-label={name}>
          <div className="resume-point-head">
            {browser.managing && <input type="checkbox" className="resume-point-select" aria-label={text(`选择 ${name}`, `Select ${name}`)} checked={browser.isSelected(point)} disabled={deleting || held} onChange={() => browser.toggle(point)}/>}
            <strong title={name}>{name}</strong>
            <span className="resume-point-kind" data-kind={kind}>{labels[kind]}</span>
            {used && <span className="resume-point-kind" data-kind="next">{jobStatus === 'paused' ? text('继续训练会从这里开始', 'Resuming starts here') : text('本次训练从这里恢复', 'This run resumed from here')}</span>}
          </div>
          <dl className="resume-point-facts">
            <div><dt>{text('保存时间', 'Saved')}</dt><dd>{formatTime(point.created_at)}</dd></div>
            <div><dt>{text('步数', 'Step')}</dt><dd>{point.step}</dd></div>
            <div><dt>{text('轮次', 'Epoch')}</dt><dd>{epochText(epochAt(point, stepsPerEpoch))}</dd></div>
            <div><dt>Loss</dt><dd>{loss}</dd></div>
            <div><dt>{text('大小', 'Size')}</dt><dd>{formatBytes(point.size)}</dd></div>
          </dl>
          <div className="resume-point-path"><code title={point.path}>{point.path}</code><CopyButton value={point.path} label={text('复制本机路径', 'Copy local path')}/></div>
          <div className="resume-point-actions">
            <button type="button" className="ui-btn ui-btn-sm" disabled={deleting || resuming || !canResume} onClick={() => onResume(point)}>{t('job.continueTraining')}</button>
            <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet ui-btn-danger" disabled={deleting || held} onClick={() => onDelete(point)}
              title={held ? (jobStatus === 'paused' ? text('暂停中的训练会从这个恢复点继续，不能删除', 'The paused run continues from this point; it cannot be deleted')
                : text('这次训练从这个恢复点开始，结束或取消后才能删除', 'This run started from this point; delete it after the run ends or is cancelled')) : undefined}><Trash2 size={14}/>{text('删除', 'Delete')}</button>
          </div>
        </li>;
      })}</ul>}
  </section>;
}
