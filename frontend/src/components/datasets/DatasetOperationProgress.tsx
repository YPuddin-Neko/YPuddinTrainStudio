import type { ReactNode } from 'react';
import { formatEta } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import './dataset-operation-progress.css';

export interface DatasetOperationProgressProps {
  label: string;
  phaseText: string;
  done?: number | null;
  total?: number | null;
  detail?: string;
  speed?: string | null;
  elapsed?: number | null;
  remaining?: number | null;
  state: 'active' | 'completed' | 'failed';
  actions?: ReactNode;
}

/** Progress is scoped to the caller's current phase, never an invented overall percentage. */
export default function DatasetOperationProgress({ label, phaseText, done, total, detail, speed, elapsed, remaining, state, actions }: DatasetOperationProgressProps) {
  const text = useWorkspaceText();
  const known = typeof total === 'number' && Number.isFinite(total) && total > 0 && typeof done === 'number' && Number.isFinite(done);
  const percent = state === 'completed' ? 100 : known ? Math.min(100, Math.max(0, Math.floor(done / total * 100))) : null;
  const finiteTime = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value > 0 && value < 1 ? text('不足 1 秒', '<1s') : formatEta(value) : '—';
  return <section className="dataset-operation-progress" data-state={state} aria-label={label}>
    <div className="dataset-operation-progress-heading"><strong>{label}</strong><span className="dataset-operation-progress-percent" aria-hidden="true">{percent === null ? '—' : `${percent}%`}</span></div>
    <div className="dataset-operation-progress-phase" role="status" aria-live="polite">{phaseText}</div>
    <div className="dataset-operation-progress-track" role="progressbar" aria-label={`${label} · ${phaseText}`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined} aria-valuetext={percent === null ? text('当前阶段总量尚未确定', 'The total for this phase is not yet known') : `${percent}%`}>
      <span className={percent === null ? 'is-indeterminate' : ''} style={percent === null ? undefined : { width: `${percent}%` }}/>
    </div>
    {detail && <p className="dataset-operation-progress-detail">{detail}</p>}
    {(speed !== undefined || elapsed != null || remaining !== undefined) && <div className="dataset-operation-progress-meta">
      {speed !== undefined && <span>{text('速度', 'Speed')} <strong>{speed || '—'}</strong></span>}
      {elapsed != null && <span>{text('已用', 'Elapsed')} <strong>{finiteTime(elapsed)}</strong></span>}
      {state === 'active' && remaining !== undefined && <span>{text('本阶段预计还需', 'Remaining in this phase')} <strong>{finiteTime(remaining)}</strong></span>}
    </div>}
    {actions && <div className="dataset-operation-progress-actions">{actions}</div>}
  </section>;
}
