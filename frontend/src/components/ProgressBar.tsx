import './progress-bar.css';

export default function ProgressBar({ label, value, max = 100, className = '' }: {
  label: string; value?: number; max?: number; className?: string;
}) {
  const percent = value !== undefined && Number.isFinite(value) && Number.isFinite(max) && max > 0
    ? Math.min(100, Math.max(0, value / max * 100)) : undefined;
  return <div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}
    className={`studio-progress ${className}`} data-indeterminate={percent === undefined || undefined}>
    <span className="studio-progress-fill" style={{ width: `${percent ?? 32}%` }}/>
  </div>;
}
