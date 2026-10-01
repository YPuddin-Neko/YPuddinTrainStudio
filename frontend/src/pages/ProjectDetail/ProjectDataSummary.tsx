import { Loader2 } from 'lucide-react';
import type { WorkspaceDataset } from '../../components/datasets/ProjectDatasetCards';
import { useWorkspaceText } from '../../utils/workspaceText';

function count(rows: WorkspaceDataset[], key: 'images' | 'captioned' | 'masks'): number | null {
  let total = 0;
  for (const row of rows) {
    const value = row.stats?.[key];
    if (row.index_status !== 'ready' || row.stats?.error || typeof value !== 'number' || !Number.isFinite(value) || value < 0) return null;
    total += value;
  }
  return total;
}

export default function ProjectDataSummary({ datasets }: { datasets: WorkspaceDataset[] }) {
  const text = useWorkspaceText();
  const training = datasets.filter(row => !row.source.is_reg);
  const regularization = datasets.filter(row => row.source.is_reg);
  const indexing = datasets.some(row => row.index_status === 'indexing');
  const items = [
    { key: 'images', label: text('图片', 'Images'), accessibleLabel: text('训练图片', 'Training images'), value: count(training, 'images') },
    { key: 'captions', label: text('标签', 'Captions'), accessibleLabel: text('训练图片标签', 'Training captions'), value: count(training, 'captioned') },
    { key: 'masks', label: text('遮罩', 'Masks'), accessibleLabel: text('训练图片遮罩', 'Training masks'), value: count(training, 'masks') },
    { key: 'regularization', label: text('正则', 'Regularization'), accessibleLabel: text('正则图片', 'Regularization images'), value: count(regularization, 'images') },
  ];
  return <div className="project-data-summary">
    <dl className="project-data-summary-counts" aria-label={text('本版本数据统计', 'Version data summary')}>
      {items.map(({ key, label, accessibleLabel, value }) => <div key={key}>
        <dt aria-label={accessibleLabel} title={accessibleLabel}>{label}</dt><dd>{value === null ? '—' : value.toLocaleString()}</dd>
      </div>)}
    </dl>
    {indexing && <Loader2 size={14} className="animate-spin" aria-label={text('索引中', 'Indexing')}/>}
  </div>;
}
