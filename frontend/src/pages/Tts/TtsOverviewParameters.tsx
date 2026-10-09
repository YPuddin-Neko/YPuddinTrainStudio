import type { TtsScopedConfig } from '../../api/tts';
import { decimalText } from '../../utils/numberText';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatRateValue } from '../JobDetail/metricPresentation';

function LearningRate({ value }: { value: number | undefined }) {
  const text = useWorkspaceText();
  if (value === undefined) return <>—</>;
  const plain = decimalText(value), rate = formatRateValue(value);
  return <>{rate}{plain !== rate && <span className="overview-parameter-plain">{text(`（${plain}）`, ` (${plain})`)}</span>}</>;
}

export default function TtsOverviewParameters({ config }: { config: TtsScopedConfig }) {
  const text = useWorkspaceText();
  if (config.engine === 'gpt-sovits-v5') return <dl className="overview-parameters">
    <div><dt>{text('模型变体', 'Model variant')}</dt><dd>{config.variant}</dd></div>
    <div><dt>{text('训练阶段', 'Training stages')}</dt><dd>{config.stage === 'both' ? 'SoVITS → GPT' : config.stage === 'gpt' ? 'GPT' : 'SoVITS'}</dd></div>
    {config.stage !== 'sovits' && <><div><dt>{text('GPT 训练轮数', 'GPT epochs')}</dt><dd>{config.gpt?.epochs.toLocaleString() ?? '—'}</dd></div><div><dt>{text('GPT 学习率', 'GPT learning rate')}</dt><dd><LearningRate value={config.gpt?.learning_rate}/></dd></div><div><dt>{text('GPT 批量大小', 'GPT batch size')}</dt><dd>{config.gpt?.batch_size ?? '—'}</dd></div></>}
    {config.stage !== 'gpt' && <><div><dt>{text('SoVITS 训练轮数', 'SoVITS epochs')}</dt><dd>{config.sovits?.epochs.toLocaleString() ?? '—'}</dd></div><div><dt>{text('SoVITS 学习率', 'SoVITS learning rate')}</dt><dd><LearningRate value={config.sovits?.learning_rate}/></dd></div><div><dt>{text('SoVITS 批量大小', 'SoVITS batch size')}</dt><dd>{config.sovits?.batch_size ?? '—'}</dd></div><div><dt>SoVITS LoRA Rank</dt><dd>{config.sovits?.lora_rank ?? '—'}</dd></div></>}
  </dl>;
  return <dl className="overview-parameters"><div><dt>{text('算法', 'Algorithm')}</dt><dd>LoRA</dd></div><div><dt>Rank / Alpha</dt><dd>{config.lora_rank} / {config.lora_alpha}</dd></div><div><dt>{text('学习率', 'Learning rate')}</dt><dd><LearningRate value={config.learning_rate}/></dd></div><div><dt>{text('优化器', 'Optimizer')}</dt><dd>AdamW</dd></div><div><dt>{text('批量大小', 'Batch size')}</dt><dd>{config.batch_size}{config.grad_accum_steps > 1 ? ` × ${config.grad_accum_steps}` : ''}</dd></div><div><dt>{text('训练更新次数', 'Optimizer updates')}</dt><dd>{config.num_iters.toLocaleString()}</dd></div></dl>;
}
