import type { TtsScopedConfig } from '../../api/tts';
import { useWorkspaceText } from '../../utils/workspaceText';

export default function TtsOverviewParameters({ config }: { config: TtsScopedConfig }) {
  const text = useWorkspaceText();
  if (config.engine === 'gpt-sovits-v5') return <dl className="overview-parameters">
    <div><dt>{text('模型变体', 'Model variant')}</dt><dd>{config.variant}</dd></div>
    <div><dt>{text('训练阶段', 'Training stages')}</dt><dd>{config.stage === 'both' ? 'SoVITS → GPT' : config.stage === 'gpt' ? 'GPT' : 'SoVITS'}</dd></div>
    {config.stage !== 'sovits' && <><div><dt>{text('GPT 训练轮数', 'GPT epochs')}</dt><dd>{config.gpt?.epochs.toLocaleString() ?? text('未知', 'Unknown')}</dd></div><div><dt>{text('GPT 学习率', 'GPT learning rate')}</dt><dd>{config.gpt?.learning_rate.toExponential() ?? text('未知', 'Unknown')}</dd></div></>}
    {config.stage !== 'gpt' && <><div><dt>{text('SoVITS 训练轮数', 'SoVITS epochs')}</dt><dd>{config.sovits?.epochs.toLocaleString() ?? text('未知', 'Unknown')}</dd></div><div><dt>{text('SoVITS 学习率', 'SoVITS learning rate')}</dt><dd>{config.sovits?.learning_rate.toExponential() ?? text('未知', 'Unknown')}</dd></div><div><dt>SoVITS LoRA Rank</dt><dd>{config.sovits?.lora_rank ?? text('未知', 'Unknown')}</dd></div></>}
  </dl>;
  return <dl className="overview-parameters"><div><dt>{text('训练更新次数', 'Optimizer updates')}</dt><dd>{config.num_iters.toLocaleString()}</dd></div><div><dt>{text('学习率', 'Learning rate')}</dt><dd>{config.learning_rate.toExponential()}</dd></div><div><dt>LoRA Rank / Alpha</dt><dd>{config.lora_rank} / {config.lora_alpha}</dd></div></dl>;
}
