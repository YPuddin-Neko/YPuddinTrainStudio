import React from 'react';
import { useTranslation } from 'react-i18next';
import { CheckCircle2, CircleSlash, Loader2, PauseCircle, PlayCircle, XCircle } from 'lucide-react';
import type { JobProgress } from '../../api/types';
import { formatTime } from '../../utils/format';
import { useWorkspaceText } from '../../utils/workspaceText';
import OverflowStrip from '../../components/OverflowStrip';

type StepState = 'done' | 'current' | 'pending' | 'pausing' | 'paused' | 'failed' | 'stopping' | 'stopped' | 'resumed' | 'idle';

const PREPARING = ['starting', 'checking_communication', 'loading', 'indexing'];
// Stepper positions of the worker phases; position 2 reports a pause or an error.
// Adapter and optimizer setup follows caching and belongs to the training stage.
const PHASE_POSITION: Record<string, number> = { preparing: 0, caching: 1, injecting: 3, prepared: 3, training: 3, finalizing: 4 };

function phaseGroup(phase: string): string {
  return PREPARING.includes(phase) ? 'preparing' : phase.startsWith('caching_') ? 'caching' : phase;
}

/**
 * 准备 · 缓存 · 暂停/错误 · 训练 · 完成. The third step names what interrupted the run:
 * 暂停中 while a pause saves its resume point and 已暂停 once saved, 取消中 / 已取消 the same way for a
 * cancel, 错误 when it failed. The step it interrupted goes back to waiting.
 */
export default function JobStepper({ status, phase, progress }: { status: string; phase: string; progress?: JobProgress | null }) {
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const group = phaseGroup(phase);
  const interrupted: StepState | null = status === 'pausing' ? 'pausing' : status === 'paused' ? 'paused' : status === 'failed' ? 'failed'
    : status === 'cancelling' ? 'stopping' : status === 'cancelled' ? 'stopped' : null;
  // A pause, stop or error saves its state in the finalizing phase; training itself did not finish.
  const trainingDone = progress?.step != null && progress.total_steps != null && progress.step >= progress.total_steps;
  const reached = interrupted && group === 'finalizing' && !trainingDone ? PHASE_POSITION.training : PHASE_POSITION[group];
  const known = reached !== undefined;
  const pauses = progress?.pause_count ?? 0;
  const resumed = !interrupted && pauses > 0;

  const state = (position: number): StepState => {
    if (position === 2) return interrupted ?? (resumed ? 'resumed' : 'idle');
    if (status === 'completed') return 'done';
    if (!known) return 'pending';
    if (interrupted) return position < reached ? 'done' : 'pending';
    if (position < reached) return 'done';
    return position === reached && status === 'running' ? 'current' : 'pending';
  };
  const interruption = interrupted === 'pausing' ? text('暂停中', 'Pausing') : interrupted === 'paused' ? text('已暂停', 'Paused')
    : interrupted === 'failed' ? text('错误', 'Error') : interrupted === 'stopping' ? text('取消中', 'Cancelling') : interrupted === 'stopped' ? text('已取消', 'Cancelled')
      : resumed ? (pauses > 1 ? text(`已恢复 · ${pauses} 次`, `Resumed · ${pauses}×`) : text('已恢复', 'Resumed')) : text('暂停/错误', 'Pause/error');
  const pauseNote = resumed || interrupted === 'paused'
    ? [progress?.paused_step != null && progress.paused_at != null ? text(`第 ${progress.paused_step} 步暂停于 ${formatTime(progress.paused_at)}`, `Paused at step ${progress.paused_step}, ${formatTime(progress.paused_at)}`) : '',
      resumed && progress?.resumed_at != null ? text(`${formatTime(progress.resumed_at)} 从第 ${progress.resumed_step ?? progress.paused_step} 步恢复`, `resumed at step ${progress.resumed_step ?? progress.paused_step}, ${formatTime(progress.resumed_at)}`) : '',
      pauses > 1 ? text(`共暂停 ${pauses} 次`, `${pauses} pauses in total`) : ''].filter(Boolean).join(text('，', '; '))
    : '';
  const steps = [
    { key: 'preparing', label: t('job.phasePreparing', '准备') },
    { key: 'caching', label: t('job.phaseCaching', '缓存') },
    { key: 'interrupted', label: interruption },
    { key: 'training', label: t('job.phaseTraining', '训练') },
    { key: 'finished', label: t('job.phaseFinished', '完成') },
  ];
  // Steps that have not happened show their name alone.
  const icon = (value: StepState) => value === 'done' ? <CheckCircle2 size={14} aria-hidden="true"/>
    : value === 'current' || value === 'pausing' || value === 'stopping' ? <Loader2 size={14} className="animate-spin" aria-hidden="true"/>
      : value === 'paused' ? <PauseCircle size={14} aria-hidden="true"/>
      : value === 'resumed' ? <PlayCircle size={14} aria-hidden="true"/>
        : value === 'failed' ? <XCircle size={14} aria-hidden="true"/>
          : value === 'stopped' ? <CircleSlash size={14} aria-hidden="true"/>
            : null;
  const stateNames: Record<StepState, string> = {
    done: text('已完成', 'done'), current: text('进行中', 'in progress'), pending: text('未开始', 'not started'),
    pausing: text('正在保存恢复点', 'saving the resume point'), paused: text('恢复点已保存', 'resume point saved'), failed: text('出错', 'failed'),
    stopping: text('正在保存恢复点', 'saving the resume point'), stopped: text('已取消', 'cancelled'), resumed: text('暂停后已恢复', 'resumed after a pause'), idle: text('未发生', 'did not occur'),
  };

  return <OverflowStrip className="job-stage-scroll" containerClassName="job-stage-strip" role="group" label={text('任务阶段', 'Job stages')} activeKey={`${status}:${phase}`}><ol className="job-stepper" aria-label={text('任务阶段', 'Job stages')}>
    {!known && status === 'running' && <li className="job-step" data-state="current" title={phase || undefined}>
      <Loader2 size={14} className="animate-spin" aria-hidden="true"/><span>{t(`phase.${phase}`, t('job.phaseInProgress', '进行中'))}</span>
    </li>}
    {steps.map((step, position) => {
      const value = state(position);
      return <React.Fragment key={step.key}>
        <li className="job-step" data-state={value} data-step={step.key} aria-current={value === 'current' || value === 'pausing' || value === 'stopping' ? 'step' : undefined} title={position === 2 && pauseNote ? pauseNote : undefined}>
          {icon(value)}<span>{step.label}</span><span className="sr-only">{text(`（${stateNames[value]}）`, ` (${stateNames[value]})`)}</span>
        </li>
        {position < steps.length - 1 && <li className="job-step-line" aria-hidden="true"/>}
      </React.Fragment>;
    })}
  </ol></OverflowStrip>;
}
