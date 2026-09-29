import { useWorkspaceText } from '../utils/workspaceText';
import './sample-loss.css';

export default function SampleLoss({ sample }: { sample: { step: number; loss?: number | null } }) {
  const text = useWorkspaceText();
  const recorded = sample.step > 0 && typeof sample.loss === 'number' && Number.isFinite(sample.loss);
  const value = recorded ? String(Number(sample.loss!.toPrecision(5))) : sample.step === 0 ? text('初始采样 · 未训练', 'Initial sample · before training') : text('未记录', 'Not recorded');
  return <div className="sample-training-loss">
    <span>{text('训练损失', 'Training loss')}</span><strong data-recorded={recorded}>{value}</strong>
  </div>;
}
