import React from 'react';
import { useTranslation } from 'react-i18next';
import { CheckCircle2, CircleSlash, Loader2, PauseCircle, XCircle } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';

type StepState = 'done' | 'current' | 'pending' | 'paused' | 'failed' | 'stopped' | 'idle';

const PREPARING = ['starting', 'checking_communication', 'loading', 'indexing', 'injecting', 'prepared'];
// Stepper positions of the worker phases; position 2 reports a pause or an error.
const PHASE_POSITION: Record<string, number> = { preparing: 0, caching: 1, training: 3, finalizing: 4 };

function phaseGroup(phase: string): string {
  return PREPARING.includes(phase) ? 'preparing' : phase.startsWith('caching_') ? 'caching' : phase;
}

/**
 * 1 准备 · 2 缓存 · 3 暂停/错误 · 4 训练 · 5 完成. The third step names what interrupted the run:
 * 暂停 in yellow when the user paused it, 错误 in red when it failed.
 */
export default function JobStepper({ status, phase }: { status: string; phase: string }) {
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const group = phaseGroup(phase);
  const reached = PHASE_POSITION[group];
  const known = reached !== undefined;
  const interrupted = status === 'paused' ? 'paused' : status === 'failed' ? 'failed' : status === 'cancelled' ? 'stopped' : null;

  const state = (position: number): StepState => {
    if (position === 2) return interrupted ?? 'idle';
    if (status === 'completed') return 'done';
    if (!known) return 'pending';
    if (interrupted) return position < reached ? 'done' : 'pending';
    if (position < reached) return 'done';
    return position === reached && ['running', 'pausing', 'cancelling'].includes(status) ? 'current' : 'pending';
  };
  const interruption = interrupted === 'paused' ? text('暂停', 'Paused') : interrupted === 'failed' ? text('错误', 'Error') : interrupted === 'stopped' ? text('已停止', 'Stopped') : text('暂停/错误', 'Pause/error');
  const steps = [
    { key: 'preparing', label: t('job.phasePreparing', '准备') },
    { key: 'caching', label: t('job.phaseCaching', '缓存') },
    { key: 'interrupted', label: interruption },
    { key: 'training', label: t('job.phaseTraining', '训练') },
    { key: 'finished', label: t('job.phaseFinished', '完成') },
  ];
  const icon = (value: StepState, position: number) => value === 'done' ? <CheckCircle2 size={14} aria-hidden="true"/>
    : value === 'current' ? <Loader2 size={14} className="animate-spin" aria-hidden="true"/>
      : value === 'paused' ? <PauseCircle size={14} aria-hidden="true"/>
        : value === 'failed' ? <XCircle size={14} aria-hidden="true"/>
          : value === 'stopped' ? <CircleSlash size={14} aria-hidden="true"/>
            : <span aria-hidden="true">{position + 1}.</span>;
  const stateNames: Record<StepState, string> = {
    done: text('已完成', 'done'), current: text('进行中', 'in progress'), pending: text('未开始', 'not started'),
    paused: text('已暂停', 'paused'), failed: text('出错', 'failed'), stopped: text('已停止', 'stopped'), idle: text('未发生', 'did not occur'),
  };

  return <ol className="job-stepper" aria-label={text('任务阶段', 'Job stages')}>
    {!known && status === 'running' && <li className="job-step" data-state="current" title={phase || undefined}>
      <Loader2 size={14} className="animate-spin" aria-hidden="true"/><span>{t(`phase.${phase}`, t('job.phaseInProgress', '进行中'))}</span>
    </li>}
    {steps.map((step, position) => {
      const value = state(position);
      return <React.Fragment key={step.key}>
        <li className="job-step" data-state={value} data-step={step.key} aria-current={value === 'current' ? 'step' : undefined}>
          {icon(value, position)}<span>{step.label}</span><span className="sr-only">{text(`（${stateNames[value]}）`, ` (${stateNames[value]})`)}</span>
        </li>
        {position < steps.length - 1 && <li className="job-step-line" aria-hidden="true"/>}
      </React.Fragment>;
    })}
  </ol>;
}
