import React from 'react';
import type { TtsPresetConfig } from '../../api/ttsPresets';
import type { TtsIssue, TtsTrainSchema } from '../../api/tts';
import { useWorkspaceText } from '../../utils/workspaceText';
import { useEnterAnimation } from '../../utils/motion';
import TtsParameterForm from '../Tts/TtsParameterForm';
import { ttsParameterGroups } from '../Tts/ttsParameterPresentation';
import { gptSovitsParameterGroups } from '../Tts/gptSovitsParameterPresentation';
import { toDraft, parseDraft, resolveTrainingSchema, type TtsDraft, type TtsField } from '../Tts/ttsVersionFields';
import { toGptSovitsDraft, parseGptSovitsDraft, resolveGptSovitsSchema, gptSovitsValue, type GptSovitsDraft, type GptSovitsField } from '../Tts/gptSovitsVersionFields';
import { ignoredGptLearningRateFields, resolveGptSovitsFixedLearningRate } from '../Tts/gptSovitsLearningRate';
import { portableTtsParameters, presetEditingConfig, rawTtsDraft } from '../Tts/ttsPresetParameters';
import '../Tts/tts-project.css';

export type TtsPresetConfigHandle = { getConfig: () => TtsPresetConfig | null; showIssues: (issues: TtsIssue[]) => void };
const TtsPresetConfigEditor = React.forwardRef<TtsPresetConfigHandle, { identity: string; revision: number; advanced: boolean; onAdvancedChange: (advanced: boolean) => void; value: Record<string, unknown>; savedConfig?: TtsPresetConfig; schema: TtsTrainSchema; readOnly: boolean; search: string; onSearch: (value: string) => void; onChange: (value: Record<string, unknown>) => void }>(function TtsPresetConfigEditor({ identity, revision, advanced, onAdvancedChange, value, savedConfig, schema, readOnly, search, onSearch, onChange }, ref) {
  const text = useWorkspaceText(), english = text('zh', 'en') === 'en';
  const [closed, setClosed] = React.useState<Set<string>>(new Set());
  React.useEffect(() => { if (search) setClosed(new Set()); }, [search]);
  const [problems, setProblems] = React.useState<{ field: string; message: string }[]>([]);
  const root = useEnterAnimation<HTMLDivElement>(identity, { duration: 180, skipFirst: true });
  React.useEffect(() => { setProblems([]); setClosed(new Set()); }, [identity, revision]);
  const config = presetEditingConfig(value);
  const gsvSchema = config.engine === 'gpt-sovits-v5' ? resolveGptSovitsSchema(schema) : null;
  const fixedLearningRate = gsvSchema ? resolveGptSovitsFixedLearningRate(schema) : undefined;
  const voxSchema = config.engine === 'voxcpm1.5' ? resolveTrainingSchema(schema) : null;
  const reveal = (issues: { field: string; message: string }[]) => {
    setProblems(issues); onSearch(''); if (issues.length) onAdvancedChange(true); setClosed(new Set());
    requestAnimationFrame(() => {
      const focusField = fixedLearningRate && ignoredGptLearningRateFields.includes(issues[0]?.field as GptSovitsField) ? 'gpt.learning_rate' : issues[0]?.field;
      const field = [...(root.current?.querySelectorAll<HTMLElement>('[data-field]') || [])].find(node => node.dataset.field === focusField);
      field?.querySelector<HTMLElement>('[data-restore-legacy-learning-rate]:not(:disabled), .config-field-control input:not(:disabled), .config-field-control button:not(:disabled), button[role="switch"]')?.focus();
    });
  };
  React.useImperativeHandle(ref, () => ({
    getConfig: () => {
      const parsed = config.engine === 'gpt-sovits-v5' && gsvSchema ? parseGptSovitsDraft(toGptSovitsDraft(config), english, gsvSchema.properties)
        : config.engine === 'voxcpm1.5' && voxSchema ? parseDraft(toDraft(config), text('请填写符合范围的有效数值。', 'Enter a valid value within the allowed range.'), voxSchema.properties) : null;
      if (!parsed) return null;
      if (parsed.problems.length) { reveal(parsed.problems); return null; }
      return portableTtsParameters(parsed.config);
    },
    showIssues: issues => reveal(issues.map(issue => ({ field: (issue.loc[0] === 'config' ? issue.loc.slice(1) : issue.loc).join('.'), message: issue.message }))),
  }));
  const change = (draft: TtsDraft | GptSovitsDraft, field: string, next: string | boolean | string[]) => {
    if (readOnly) return;
    const raw = rawTtsDraft({ ...draft, [field]: next }, false);
    for (const key of ['python_path', 'trainer_path', 'model_path', 'pretrained_gpt', 'pretrained_sovits', 'variant']) delete raw[key];
    setProblems(previous => previous.filter(issue => issue.field !== field)); onChange(raw);
  };
  const shared = { readOnly, saving: false, english, text, problems, preset: true };
  const restoreLegacyLearningRate = () => {
    if (readOnly || !gsvSchema || !fixedLearningRate) return;
    const original = savedConfig ? presetEditingConfig(savedConfig) : null;
    const settings = value.gpt && typeof value.gpt === 'object' && !Array.isArray(value.gpt) ? value.gpt as Record<string, unknown> : {};
    const restored = { ...settings };
    for (const field of ignoredGptLearningRateFields) restored[field.slice(4)] = original?.engine === 'gpt-sovits-v5'
      ? gptSovitsValue(original, field) ?? gsvSchema.properties[field].default : gsvSchema.properties[field].default;
    setProblems(previous => previous.filter(problem => !ignoredGptLearningRateFields.includes(problem.field as GptSovitsField)));
    onChange({ ...value, gpt: restored });
  };
  const groups = config.engine === 'gpt-sovits-v5' && gsvSchema ? gptSovitsParameterGroups({ ...shared, draft: toGptSovitsDraft(config), properties: gsvSchema.properties, orderedGroups: gsvSchema.groups, change: (field: GptSovitsField, next) => change(toGptSovitsDraft(config), field, next), fixedLearningRate, onRestoreLegacyLearningRate: restoreLegacyLearningRate })
    : config.engine === 'voxcpm1.5' && voxSchema ? ttsParameterGroups({ ...shared, draft: toDraft(config), properties: voxSchema.properties, orderedGroups: voxSchema.groups, change: (field: TtsField, next) => change(toDraft(config), field, next) }) : [];
  return <div ref={root} className="tts-preset-config-editor">{groups.length ? <TtsParameterForm id="tts-preset" groups={groups} advanced={advanced} onAdvancedChange={onAdvancedChange} search={search} onSearch={next => { onSearch(next); if (next) setClosed(new Set()); }} closed={closed} onClosedChange={setClosed}/> : <p role="alert" className="studio-error">{text('参数定义与预设模型不匹配，请重新读取。', 'Parameter definitions do not match the preset model. Reload them.')}</p>}</div>;
});
export default TtsPresetConfigEditor;
