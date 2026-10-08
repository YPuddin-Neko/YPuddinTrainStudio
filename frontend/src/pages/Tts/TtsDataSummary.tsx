import type { ProjectVersion } from '../../utils/projectVersions';
import OverflowStrip from '../../components/OverflowStrip';
import { useWorkspaceText } from '../../utils/workspaceText';
import '../ProjectDetail/project-data.css';

export default function TtsDataSummary({ stats }: { stats: ProjectVersion['audio_stats'] }) {
  const text = useWorkspaceText();
  const values = [
    { label: text('训练音频', 'Training clips'), value: stats?.train.clips_count?.toLocaleString() ?? '—' },
    ...(stats?.validation ? [{ label: text('验证音频', 'Validation clips'), value: stats.validation.clips_count?.toLocaleString() ?? '—' }] : []),
    { label: text('训练时长', 'Training duration'), value: stats?.train.duration_seconds == null ? '—' : `${stats.train.duration_seconds.toFixed(1)} s` },
  ];
  return <div className="project-data-summary"><OverflowStrip snap role="group" label={text('本版本数据统计', 'Version data summary')} containerClassName="project-data-summary-strip" className="project-data-summary-counts" pageLabels={{ previous: text('上一组统计', 'Previous counts'), next: text('下一组统计', 'Next counts') }}>
    {values.map(item => <dl key={item.label}><dt>{item.label}</dt><dd>{item.value}</dd></dl>)}
  </OverflowStrip></div>;
}
