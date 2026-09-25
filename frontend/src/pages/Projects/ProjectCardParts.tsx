import React from 'react';
import type { Project } from '../../api/types';
import ProgressBar from '../../components/ProgressBar';
import { JobStatus } from '../Queue/jobPresentation';
import { shortTime } from '../../utils/jobs';
import { useWorkspaceText } from '../../utils/workspaceText';
import { ProjectCover } from './ProjectEditor';
import { coverSource } from './projectGallery';
import './projects.css';

/** The cover, or the name's first character on a tint derived from the name. */
export function ProjectArtwork({ name, coverUrl }: { name: string; coverUrl?: string | null }) {
  if (coverUrl) return <ProjectCover source={coverSource(coverUrl)} name={name}/>;
  let hash = 0;
  for (const character of name) hash = (hash * 31 + character.codePointAt(0)!) % 360;
  return <div className="project-monogram" style={{ '--hue': hash } as React.CSSProperties} aria-hidden="true">
    <span>{Array.from(name.trim())[0] || '·'}</span>
  </div>;
}

/** What the project's training is doing: the active run first, otherwise the latest one. */
export function ProjectActivityLine({ job }: { job?: Project['latest_job'] }) {
  const text = useWorkspaceText();
  if (!job) return <div className="project-activity" data-idle><span>{text('尚未训练', 'No training yet')}</span></div>;
  const progress = job.total_steps ? Math.min(100, Math.floor((job.step ?? 0) / job.total_steps * 100)) : null;
  const moving = ['running', 'pausing', 'cancelling', 'paused'].includes(job.status);
  const detail = moving && progress !== null ? `${progress}%`
    : job.status === 'failed' ? (job.error || '').trim().split('\n')[0]
      : job.finished_at != null ? shortTime(job.finished_at) : '';
  return <div className="project-activity" data-status={job.status} title={job.status === 'failed' && job.error ? job.error : job.name}>
    <JobStatus status={job.status}/>
    {moving && progress !== null && <ProgressBar className="project-activity-bar" label={text('训练进度', 'Training progress')} value={job.step ?? 0} max={job.total_steps ?? 0}/>}
    {detail && <span className="project-activity-detail">{detail}</span>}
  </div>;
}
