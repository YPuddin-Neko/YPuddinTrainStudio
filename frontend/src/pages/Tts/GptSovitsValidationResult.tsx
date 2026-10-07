import type React from 'react';
import type { TtsGptSovitsDatasetReport, TtsGptSovitsEnvironmentReport, TtsIssue, TtsValidationReport } from '../../api/tts';
import { gpuDeviceLabel } from '../../utils/gpuDevices';
import { useWorkspaceText } from '../../utils/workspaceText';

const checks = [['python', 'Python 解释器', 'Python executable'], ['upstream', '训练器版本', 'Trainer version'], ['model', '模型资产', 'Model assets'], ['dependencies', '依赖环境', 'Dependencies'], ['cuda', 'CUDA 显卡', 'CUDA GPU'], ['precision', '训练精度', 'Training precision']] as const;
type Props = {
  report: TtsValidationReport;
  dataset: TtsGptSovitsDatasetReport;
  environment: TtsGptSovitsEnvironmentReport;
  stale: boolean;
  renderIssues: (issues: TtsIssue[]) => React.ReactNode;
};
const identity = (issue: TtsIssue) => JSON.stringify([issue.code, issue.loc, issue.message, issue.severity]);

export default function GptSovitsValidationResult({ report, dataset, environment, stale, renderIssues }: Props) {
  const text = useWorkspaceText();
  const value = (number: number | null | undefined) => number == null ? text('未知', 'Unknown') : number.toLocaleString();
  const state = (key: string) => ({ available: text('可用', 'Available'), unavailable: text('不可用', 'Unavailable'), unchecked: text('未检查', 'Not checked'), disabled: text('已关闭', 'Disabled'), not_applicable: text('不适用', 'Not applicable'), blocked: text('受阻', 'Blocked'), error: text('检查失败', 'Check failed'), missing: text('未登记', 'Not registered'), checking: text('正在检查', 'Checking'), valid: text('通过', 'Passed'), invalid: text('存在无效数据', 'Contains invalid data'), stale: text('内容已变化', 'Content changed') }[key] || key);
  const summarized = new Set([...report.errors, ...report.warnings].map(identity));
  const remaining = (items: TtsIssue[] = [], represented: TtsIssue[] = []) => {
    const seen = new Set([...summarized, ...represented.map(identity)]);
    return items.filter(issue => { const key = identity(issue); if (seen.has(key)) return false; seen.add(key); return true; });
  };
  const issues = (items: TtsIssue[]) => items.length ? renderIssues(items) : null;
  const { train, validation } = dataset;
  const preparationRows: [string, React.ReactNode, React.ReactNode][] = [
    [text('数据状态', 'Data status'), <>{state(train.state)}{issues(remaining(train.issues, [...(train.preparation.issues || []), ...train.stages.flatMap(stage => stage.issues || [])]))}</>, <>{state(validation.state)}{issues(remaining(validation.issues, [...(validation.preparation.issues || []), ...validation.stages.flatMap(stage => stage.issues || [])]))}</>],
    [text('清单检查', 'Manifest check'), state(train.source_state), state(validation.source_state)],
    [text('清单音频数', 'Manifest clips'), value(train.source_summary?.clips_count), value(validation.source_summary?.clips_count)],
    [text('有效音频数', 'Valid clips'), value(train.source_summary?.valid_clips_count), value(validation.source_summary?.valid_clips_count)],
    [text('无效音频数', 'Invalid clips'), value(train.source_summary?.invalid_count), value(validation.source_summary?.invalid_count)],
    [text('预处理状态', 'Preparation status'), <>{state(train.preparation.state)}{issues(remaining(train.preparation.issues))}</>, <>{state(validation.preparation.state)}{issues(remaining(validation.preparation.issues))}</>],
    [text('预处理后样本', 'Prepared samples'), value(train.preparation.prepared_samples), value(validation.preparation.prepared_samples)],
    [text('预处理排除样本', 'Filtered samples'), value(train.preparation.filtered_samples), value(validation.preparation.filtered_samples)],
  ];
  return <div className="tts-training-report">
    {!stale && <p role="status" className="tts-training-result" data-valid={report.valid}>{report.valid ? text('检查通过，可以加入训练队列。', 'Checks passed. Ready to join the training queue.') : text('检查未通过，请处理以下问题。', 'Checks failed. Resolve the issues below.')}</p>}
    {!!report.errors.length && <section aria-label={text('检查错误', 'Check errors')}>{renderIssues(report.errors)}</section>}
    {!!report.warnings.length && <section aria-label={text('检查提醒', 'Check warnings')}><h3>{text('提醒', 'Warnings')}</h3>{renderIssues(report.warnings)}</section>}
    <h3>{text('环境检查', 'Environment checks')}</h3><ul className="tts-training-checks">{checks.map(([key, zh, en]) => { const item = environment.checks.find(check => check.key === key); return <li key={key}><div><span>{text(zh, en)}</span><strong data-state={item?.state || 'unchecked'}>{state(item?.state || 'unchecked')}</strong></div>{issues(remaining(item?.issues))}</li>; })}</ul>
    <section aria-label={text('显卡检查', 'GPU checks')}><h3>{text('显卡检查', 'GPU checks')}</h3>
      {environment.devices ? <><dl className="tts-training-facts mb-2"><div><dt>{text('选择方式', 'Selection')}</dt><dd>{environment.devices.requested_devices.length ? environment.devices.requested_devices.map(gpuDeviceLabel).join(', ') : text('自动选择', 'Automatic')}</dd></div><div><dt>{text('符合训练要求', 'Eligible for training')}</dt><dd>{environment.devices.eligible_devices.length ? environment.devices.eligible_devices.map(gpuDeviceLabel).join(', ') : text('无', 'None')}</dd></div></dl><ul className="tts-training-checks">{environment.devices.checked_devices.map(device => <li key={device.device}><div><span className="min-w-0 break-words">{gpuDeviceLabel(device.device)}{device.name ? ` · ${device.name}` : ''}</span><strong data-state={device.state}>{state(device.state)}</strong></div>{issues(remaining(device.issues))}</li>)}</ul></> : <p className="tts-training-note">{text('尚未完成显卡检查。', 'GPU checks have not completed.')}</p>}
    </section>
    <h3>{text('数据与预处理', 'Data and preparation')}</h3><div className="tts-training-table" role="region" aria-label={text('数据与预处理结果', 'Data and preparation results')} tabIndex={0}><table><thead><tr><th>{text('检查项', 'Check')}</th><th>{text('训练数据', 'Training data')}</th><th>{text('验证数据', 'Validation data')}</th></tr></thead><tbody>{preparationRows.map(([label, training, validating]) => <tr key={label}><th scope="row">{label}</th><td>{training}</td><td>{validating}</td></tr>)}</tbody></table></div>
    <p className="tts-training-note">{text('预处理样本数和阶段批次数将在训练预处理后确定。GPT-SoVITS 不使用独立验证清单。', 'Sample and stage batch counts are determined after training preparation. GPT-SoVITS does not use a separate validation manifest.')}</p>
    {[train, validation].map(split => <section key={split.split} aria-label={split.split === 'train' ? text('训练阶段', 'Training stages') : text('验证阶段', 'Validation stages')}><h3>{split.split === 'train' ? text('训练阶段', 'Training stages') : text('验证阶段', 'Validation stages')}</h3><div className="tts-training-table" role="region" aria-label={split.split === 'train' ? text('训练阶段统计', 'Training stage counts') : text('验证阶段统计', 'Validation stage counts')} tabIndex={0}><table><thead><tr><th>{text('检查项', 'Check')}</th><th>GPT</th><th>SoVITS</th></tr></thead><tbody>
      <tr><th scope="row">{text('阶段状态', 'Stage status')}</th>{(['gpt', 'sovits'] as const).map(key => { const stage = split.stages.find(item => item.stage === key); return <td key={key}>{stage ? state(stage.state) : text('未知', 'Unknown')}{issues(remaining(stage?.issues))}</td>; })}</tr>
      {([[text('每批样本数', 'Samples per batch'), 'batch_size'], [text('输入样本', 'Input samples'), 'input_samples'], [text('每轮输出批次', 'Yielded batches per pass'), 'yielded_batches_per_pass'], [text('每轮丢弃样本', 'Dropped samples per pass'), 'dropped_samples_per_pass']] as const).map(([label, field]) => <tr key={field}><th scope="row">{label}</th>{(['gpt', 'sovits'] as const).map(key => <td key={key}>{value(split.stages.find(stage => stage.stage === key)?.[field])}</td>)}</tr>)}
    </tbody></table></div></section>)}
  </div>;
}
