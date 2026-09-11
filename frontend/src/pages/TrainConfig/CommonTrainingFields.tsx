import { useWorkspaceText } from '../../utils/workspaceText';

export default function CommonTrainingFields({ config, onChange }: { config: Record<string, any>; onChange: (config: Record<string, any>) => void }) {
  const text = useWorkspaceText();
  const inputClass = 'mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-600';
  const update = (section: string, field: string, value: unknown) => onChange({ ...config, [section]: { ...config[section], [field]: value } });
  const fields = [
    { section: 'dataset', field: 'batch_size', label: text('每批图片数', 'Batch size'), min: 1, step: '1', hint: text('显存不足时先降低此值', 'Reduce this first if memory is insufficient') },
    { section: 'loop', field: 'epochs', label: text('训练轮数', 'Training epochs'), min: 1, step: '1', hint: text('完整看过训练集的次数', 'Full passes through the training dataset') },
    { section: 'optimizer', field: 'lr', label: text('学习率', 'Learning rate'), min: 0, step: 'any', hint: text('控制每一步的更新幅度', 'Controls the size of each update') },
    { section: 'loop', field: 'grad_accum', label: text('梯度累积', 'Gradient accumulation'), min: 1, step: '1', hint: text('不增大单批显存的情况下累积更新', 'Accumulate updates without increasing batch memory') },
  ];
  return <section className="rounded-xl border border-slate-200 bg-white p-5 dark:border-slate-700 dark:bg-slate-800 space-y-4" aria-label={text('常用训练参数', 'Common training parameters')}>
    <div className="flex flex-wrap items-center justify-between gap-2"><div><h3 className="text-lg font-semibold">{text('常用训练参数', 'Common training parameters')}</h3><p className="mt-1 text-xs text-slate-500">{text('这里的修改会同步下方完整配置，并重新检查训练计划。', 'Changes update the full configuration below and revalidate the training plan.')}</p></div><a href="#full-training-config" className="text-sm text-blue-500">{text('查看完整参数 ↓', 'All parameters ↓')}</a></div>
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {fields.map(({ section, field, label, min, step, hint }) => <label key={field} className="block text-sm font-medium">{label}<input aria-label={label} className={inputClass} type="number" min={min} step={step} value={config[section]?.[field] ?? ''} onChange={(e) => update(section, field, e.target.value === '' && field === 'epochs' ? null : Number(e.target.value))} /><span className="mt-1 block text-xs font-normal text-slate-500">{hint}</span></label>)}
      <label className="block text-sm font-medium">{text('LoRA Rank', 'LoRA rank')}<input aria-label={text('LoRA Rank', 'LoRA rank')} className={inputClass} type="number" min="1" step="1" disabled={config.adapter?.rank === 'full'} placeholder={config.adapter?.rank === 'full' ? 'full' : undefined} value={config.adapter?.rank === 'full' ? '' : config.adapter?.rank ?? ''} onChange={(e) => update('adapter', 'rank', Number(e.target.value))} /><span className="mt-1 block text-xs font-normal text-slate-500">{text('模型适配器的容量；完整 rank 模式请在下方设置', 'Adapter capacity; configure full-rank mode below')}</span></label>
      <label className="block text-sm font-medium">{text('训练分辨率', 'Training resolution')}<input aria-label={text('训练分辨率', 'Training resolution')} className={inputClass} type="number" min="32" step="1" value={config.dataset?.resolutions?.[0] ?? ''} onChange={(e) => update('dataset', 'resolutions', [Number(e.target.value), ...(config.dataset?.resolutions || []).slice(1)])} /><span className="mt-1 block text-xs font-normal text-slate-500">{text('主分辨率；多个分辨率可在完整参数中设置', 'Primary resolution; configure multiple resolutions below')}</span></label>
    </div>
    <label className="block text-sm font-medium">{text('触发词（可选）', 'Trigger word (optional)')}<input className={inputClass} value={config.dataset?.caption?.trigger_word ?? ''} onChange={(e) => onChange({ ...config, dataset: { ...config.dataset, caption: { ...config.dataset?.caption, trigger_word: e.target.value || null } } })} placeholder={text('例如 my_character', 'For example my_character')} /></label>
  </section>;
}
