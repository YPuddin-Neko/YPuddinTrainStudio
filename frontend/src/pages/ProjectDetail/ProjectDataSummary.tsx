import { Loader2 } from 'lucide-react';
import OverflowStrip from '../../components/OverflowStrip';
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
  // One line of counts that pages sideways when the heading is narrow.
  return <div className="project-data-summary">
    <OverflowStrip snap role="group" label={text('本版本数据统计', 'Version data summary')} containerClassName="project-data-summary-strip" className="project-data-summary-counts"
      pageLabels={{ previous: text('上一组统计', 'Previous counts'), next: text('下一组统计', 'Next counts') }}>
      {items.map(({ key, label, accessibleLabel, value }) => <dl key={key}>
        <dt aria-label={accessibleLabel} title={accessibleLabel}>{label}</dt><dd>{value === null ? '—' : value.toLocaleString()}</dd>
      </dl>)}
    </OverflowStrip>
    {indexing && <Loader2 size={14} className="animate-spin" aria-label={text('索引中', 'Indexing')}/>}
  </div>;
}
