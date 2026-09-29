import { useEffect, useState, type ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Bot, RotateCcw, Save, Settings2, Trash2 } from 'lucide-react';
import { formatApiError } from '../../utils/errors';
import { useWorkspaceText } from '../../utils/workspaceText';
import StudioSelect from '../StudioSelect';
import CheckboxSelect from '../CheckboxSelect';
import type { TaggingOptions } from '../../api/types';
import VisionModelField, { VisionRuntimeNotice } from './VisionModelField';
import { DeviceField, OperationResult, RangeField, ScopeField, SelectField, TagOutputOptions } from './VisionPanelParts';
import { SERVICE_NAMES, resolveService, useCategoryLabels, useRememberedSettings, useScopeOptions, useTaggingSettings, useVisionModels, useVlmServices } from './visionHooks';
import { BUILTIN_TEMPLATES, defaultTemplate, type PromptTemplate, type VlmMode, type VlmOutput } from './vlmPrompts';
import type { PipelineOperation } from './DatasetPipelinePanel';
import './dataset-vision.css';

type Shared = { trigger: string; exclude: string };
type ModeSettings = { output: VlmOutput; existing: 'skip' | 'overwrite' | 'refine'; templateId: string; prompt: string };
type Category = NonNullable<TaggingOptions['categories']>[number];
type TaggerSettings = { categories: Category[]; model: string; general_threshold: number; character_threshold: number; device: 'auto' | 'cpu'; replace_underscore: boolean; escape_parentheses: boolean };

const SHARED: Shared = { trigger: '', exclude: '' };
const TAGGER: TaggerSettings = { categories: ['general', 'character'], model: 'wd-eva02-large-tagger-v3', general_threshold: 0.35, character_threshold: 0.85, device: 'auto', replace_underscore: true, escape_parentheses: false };
const CUSTOM_TEMPLATE_ID = '__custom__';
const modeDefaults = (mode: VlmMode): ModeSettings => {
  const output: VlmOutput = mode === 'assist' ? 'categories' : 'tags';
  const template = defaultTemplate(mode, output);
  return { output, existing: mode === 'assist' ? 'refine' : 'skip', templateId: template.id, prompt: template.prompt };
};

export default function VlmTagPanel({ mode, projectId, versionId, locked, latest, header, running, onStart, onUndo, onReview }: {
  mode: VlmMode; projectId: string; versionId: string; locked: boolean; latest?: PipelineOperation; header?: ReactNode; running?: ReactNode;
  onStart: (body: Record<string, unknown>) => Promise<void>; onUndo: (id: string) => void; onReview: () => void;
}) {
  const text = useWorkspaceText();
  const assisted = mode === 'assist';
  const location = useLocation();
  const services = useVlmServices();
  const { tagging, query: settingsQuery } = useTaggingSettings();
  const catalog = useVisionModels();
  const scopes = useScopeOptions(projectId, versionId);
  const [shared, updateShared] = useRememberedSettings('studio.vlm.service', SHARED);
  const [settings, update] = useRememberedSettings(`studio.vlm.${mode}`, modeDefaults(mode));
  const [tagger, updateTagger] = useRememberedSettings('studio.assist.tagger', TAGGER);
  const [custom, updateCustom] = useRememberedSettings<{ templates: PromptTemplate[] }>('studio.vlm.templates', { templates: [] });
  const [scope, setScope] = useState('');
  const [naming, setNaming] = useState<string | null>(null);
  const [error, setError] = useState('');
  const chosenScope = scopes.options.some(option => option.value === scope) ? scope : scopes.first;
  const vlm = tagging?.vlm;
  const resolved = resolveService(vlm, services.data?.services);
  const cuda = !!catalog.data?.runtime.providers.includes('cuda');
  const categoryLabels = useCategoryLabels();
  const taggerModel = catalog.data?.models.find(item => item.id === tagger.model && item.role === 'tagger') || catalog.data?.models.find(item => item.role === 'tagger');
  const offered = (taggerModel?.categories || ['general', 'character']) as Category[];
  const categories = tagger.categories.filter(category => offered.includes(category));
  const recommended = taggerModel?.thresholds || { general: 0.35, character: 0.85 };
  const chooseTagger = (id: string) => {
    const next = catalog.data?.models.find(item => item.id === id);
    updateTagger({ model: id, ...(next?.thresholds ? { general_threshold: next.thresholds.general, character_threshold: next.thresholds.character } : {}) });
  };
  // Refining reuses each image's own caption; the tagger only runs for images without one.
  const needsTagger = assisted && settings.existing !== 'refine';
  const taggerReady = !!taggerModel?.ready && !!catalog.data?.runtime.available && categories.length > 0;
  const templates = [...BUILTIN_TEMPLATES, ...custom.templates].filter(item => item.mode === mode && item.output === settings.output);
  const fallbackTemplate = defaultTemplate(mode, settings.output);
  const template = templates.find(item => item.id === settings.templateId);
  const selectedTemplate = template || (settings.templateId === fallbackTemplate.id ? fallbackTemplate : undefined);
  const editingCustom = settings.templateId === CUSTOM_TEMPLATE_ID || (!!template && !template.builtin);
  const effectivePrompt = editingCustom ? settings.prompt : (template || fallbackTemplate).prompt;
  const effectiveTemplateId = settings.templateId === CUSTOM_TEMPLATE_ID || template ? settings.templateId : fallbackTemplate.id;
  const keyMissing = !!resolved && !resolved.service.editable && !resolved.service.key_configured;
  const ready = !!resolved?.model && !keyMissing && !!effectivePrompt.trim() && (!needsTagger || taggerReady);
  const result = latest?.result;
  const changed = result?.changed_files ?? 0;

  useEffect(() => {
    if (settings.templateId === CUSTOM_TEMPLATE_ID || template) return;
    if (settings.templateId !== fallbackTemplate.id || settings.prompt !== fallbackTemplate.prompt) {
      update({ templateId: fallbackTemplate.id, prompt: fallbackTemplate.prompt });
    }
  }, [fallbackTemplate.id, fallbackTemplate.prompt, settings.prompt, settings.templateId, template, update]);

  const chooseOutput = (output: VlmOutput) => {
    const next = defaultTemplate(mode, output) || BUILTIN_TEMPLATES.find(item => item.mode === mode)!;
    update({ output, templateId: next.id, prompt: next.prompt });
  };
  const chooseTemplate = (id: string) => {
    if (id === CUSTOM_TEMPLATE_ID) { update({ templateId: CUSTOM_TEMPLATE_ID, prompt: effectivePrompt }); return; }
    const picked = templates.find(item => item.id === id);
    if (picked) update({ templateId: picked.id, prompt: picked.prompt });
  };
  const saveTemplate = () => {
    const name = (naming || '').trim();
    if (!name) return;
    const existing = custom.templates.find(item => item.mode === mode && item.output === settings.output && item.name === name);
    const saved: PromptTemplate = { id: existing?.id || `custom-${Date.now()}`, mode, output: settings.output, name, prompt: settings.prompt };
    updateCustom({ templates: [...custom.templates.filter(item => item.id !== saved.id), saved] });
    update({ templateId: saved.id });
    setNaming(null);
  };
  const deleteTemplate = () => {
    if (!template || template.builtin) return;
    updateCustom({ templates: custom.templates.filter(item => item.id !== template.id) });
    const fallback = defaultTemplate(mode, settings.output);
    update({ templateId: fallback.id, prompt: fallback.prompt });
  };
  const start = async () => {
    if (!resolved || !vlm) return;
    setError('');
    try {
      await onStart({
        action: assisted ? 'assisttag' : 'vlmtag',
        dataset_ids: scopes.ids(chosenScope),
        vlm: {
          provider: resolved.service.id, base_url: resolved.service.editable ? resolved.baseUrl : null, model: resolved.model,
          prompt: effectivePrompt, output: settings.output, existing: settings.existing,
          trigger_word: shared.trigger.trim() || null, exclude_tags: shared.exclude.split(',').map(tag => tag.trim()).filter(Boolean),
          temperature: vlm.temperature, max_tokens: vlm.max_tokens ?? null, image_size: vlm.image_size, image_detail: vlm.image_detail,
          concurrency: vlm.concurrency, interval: vlm.interval, timeout: vlm.timeout, retries: vlm.retries,
        },
        ...(assisted ? { tagging: { categories, model: taggerModel?.id, general_threshold: tagger.general_threshold, character_threshold: tagger.character_threshold, device: cuda ? tagger.device : 'cpu', replace_underscore: tagger.replace_underscore, escape_parentheses: tagger.escape_parentheses } } : {}),
      });
    } catch (e) { setError(formatApiError(e)); }
  };

  const outputs: { value: VlmOutput; label: string }[] = [
    { value: 'tags', label: assisted ? text('修正后的标签（TXT）', 'Corrected tags (TXT)') : text('标签（TXT）', 'Tags (TXT)') },
    { value: 'categories', label: assisted ? text('修正标签并分类（JSON）', 'Corrected, grouped tags (JSON)') : text('分类标签与描述（JSON）', 'Grouped tags and description (JSON)') },
    ...(assisted ? [{ value: 'sort' as const, label: text('只归类，不增删（JSON）', 'Group only, keep every tag (JSON)') }] : []),
    { value: 'description', label: text('自然语言描述（TXT / JSON 描述字段）', 'Description (TXT / JSON description field)') },
  ];
  const outputHint = {
    tags: text('写入同名 TXT；已有 JSON 只更新标签字段。', 'Writes the matching TXT; existing JSON updates its tag fields only.'),
    categories: text('写入同名 JSON 的数量、外观、标签、环境和描述字段。', 'Writes count, appearance, tags, environment and description fields to JSON.'),
    sort: text('写入 JSON，只重新归类已有标签，不增加或删除。', 'Writes JSON and regroups existing tags without adding or removing any.'),
    description: text('TXT 写入整段描述；JSON 只写描述字段。', 'Writes the full description to TXT; JSON gets only its description field.'),
  }[settings.output];
  const existingOptions = assisted ? [
    { value: 'refine', label: text('以已有标签为参考修正', 'Refine the existing captions') },
    { value: 'skip', label: text('跳过已有标签的图片', 'Skip captioned images') },
    { value: 'overwrite', label: text('忽略已有标签，重新打标', 'Ignore them and tag again') },
  ] : [
    { value: 'skip', label: text('跳过已有标签的图片', 'Skip captioned images') },
    { value: 'overwrite', label: text('覆盖已有标签', 'Replace existing captions') },
  ];
  const done = [
    result?.stopped ? text(`已停止：写入 ${changed} 个标签文件`, `Stopped: wrote ${changed} caption files`) : text(`已写入 ${changed} 个标签文件`, `Wrote ${changed} caption files`),
    result?.skipped ? text(`跳过 ${result.skipped} 张`, `skipped ${result.skipped}`) : '',
    result?.failed_files ? text(`${result.failed_files} 张失败`, `${result.failed_files} failed`) : '',
  ].filter(Boolean).join(text('，', ', ')) + text('。', '.');

  return <section className="vision-panel" aria-label={assisted ? text('辅助打标', 'Assisted tagging') : text('视觉大模型打标', 'Vision model tagging')} data-testid={assisted ? 'assist-panel' : 'vlm-panel'}>
    <header className="vision-panel-head">{header ?? <h3><Bot size={16}/>{assisted ? text('辅助打标', 'Assisted tagging') : text('视觉大模型打标', 'Vision model tagging')}</h3>}</header>
    {needsTagger && <VisionRuntimeNotice catalog={catalog}/>}
    <div className="vision-panel-body">
      {assisted && <>
        <h4 className="vision-section-title">{text('Tagger 模型', 'Tagger model')}</h4>
        <VisionModelField role="tagger" catalog={catalog} value={taggerModel?.id || ''} disabled={locked} onChange={chooseTagger}
          label={text('打标模型', 'Tagger model')} hint={text('先给没有标签的图片打标，结果作为参考交给视觉大模型。', 'Tags images without captions first; the tags go to the vision model as reference.')}/>
        <div className="vision-row vision-row-quad">
          <RangeField label={text('通用标签阈值', 'General tag threshold')} hint={text(`越低标签越多，也越容易出错，此模型推荐 ${recommended.general}。`, `Lower adds more tags and more mistakes; ${recommended.general} suits this model.`)}
            value={tagger.general_threshold} min={0.05} max={0.95} step={0.01} disabled={locked} onChange={value => updateTagger({ general_threshold: value })}/>
          <RangeField label={text('角色标签阈值', 'Character tag threshold')} hint={text(`角色名的把握要求，此模型推荐 ${recommended.character}。`, `Confidence needed for character names; ${recommended.character} suits this model.`)}
            value={tagger.character_threshold} min={0.05} max={0.95} step={0.01} disabled={locked} onChange={value => updateTagger({ character_threshold: value })}/>
          <div className="vision-field"><span className="vision-field-label">{text('写入类别', 'Categories')}</span>
            <CheckboxSelect aria-label={text('写入类别', 'Categories')} values={categories} disabled={locked} placeholder={text('至少选一类', 'Pick at least one')}
              options={offered.map(category => ({value:category, label:categoryLabels[category]}))} onValuesChange={values => updateTagger({categories:values as Category[]})}/>
            <span className="vision-field-hint">{text('用这些类别的标签作为参考。', 'Use these tag categories as reference.')}</span></div>
          {cuda && <DeviceField value={tagger.device} disabled={locked} onChange={device => updateTagger({ device })}/>}
        </div>
        <TagOutputOptions replaceUnderscore={tagger.replace_underscore} escapeParentheses={tagger.escape_parentheses}
          disabled={locked} onChange={patch => updateTagger({
            ...(patch.replaceUnderscore === undefined ? {} : { replace_underscore: patch.replaceUnderscore }),
            ...(patch.escapeParentheses === undefined ? {} : { escape_parentheses: patch.escapeParentheses }),
          })}/>
        <h4 className="vision-section-title">{text('视觉大模型', 'Vision model')}</h4>
      </>}
      <div className="vision-row vision-row-quad">
        <div className="vision-field"><span className="vision-field-label">{text('模型服务', 'Service')}</span>
          <span className="vision-service-summary" data-testid="vlm-service-summary">
            {resolved ? <span>{text(...(SERVICE_NAMES[resolved.service.id] || [resolved.service.id, resolved.service.id]))} · {resolved.model || <em>{text('未选择模型', 'No model chosen')}</em>}</span>
              : settingsQuery.isPending || services.isPending ? <span>{text('读取中…', 'Loading…')}</span> : <span>—</span>}
            <Link className="ui-btn ui-btn-sm" to="/settings/environment?tab=tagging" state={{ backgroundLocation: location.state?.backgroundLocation ?? location }}><Settings2 size={13}/>{text('修改', 'Change')}</Link></span>
          <span className="vision-field-hint">{keyMissing ? text('先在设置中保存该服务的 API 密钥。', 'Save this service’s API key in Settings first.') : resolved && !resolved.model ? text('先在设置中选择模型。', 'Choose a model in Settings first.') : text('服务、密钥和请求参数在 设置 → 打标 中修改。', 'Change the service, key and requests under Settings → Tagging.')}</span></div>
        <ScopeField scopes={scopes} value={chosenScope} onChange={setScope} disabled={locked} label={text('打标范围', 'Images')} hint={text('暂不训练的图片不会打标。', 'Images held out of training are skipped.')}/>
        <SelectField label={text('写入内容', 'Result')} hint={outputHint} value={settings.output} options={outputs} disabled={locked} onChange={value => chooseOutput(value as VlmOutput)}/>
        <SelectField label={text('已有标签', 'Existing captions')} value={settings.existing} options={existingOptions} disabled={locked} onChange={value => update({ existing: value as ModeSettings['existing'] })}
          hint={assisted ? text('没有标签的图片先用 Tagger 打标。', 'Images without captions are tagged by the tagger first.') : text('已有标签的图片是否发送给模型。', 'Whether captioned images are sent to the model.')}/>
        <label className="vision-field"><span className="vision-field-label">{text('触发词', 'Trigger word')}</span>
          <input type="text" aria-label={text('触发词', 'Trigger word')} value={shared.trigger} maxLength={200} disabled={locked} placeholder={text('可选，例如 mychar', 'Optional, e.g. mychar')} onChange={event => updateShared({ trigger: event.target.value })}/>
          <span className="vision-field-hint">{text('TXT 写在最前面，JSON 写入触发词字段。', 'First in TXT; the trigger field in JSON.')}</span></label>
        <label className="vision-field"><span className="vision-field-label">{text('排除标签', 'Excluded tags')}</span>
          <input type="text" aria-label={text('排除标签', 'Excluded tags')} value={shared.exclude} disabled={locked} placeholder={text('例如 simple background', 'e.g. simple background')} onChange={event => updateShared({ exclude: event.target.value })}/>
          <span className="vision-field-hint">{text('这些标签不会写入，用逗号分隔。', 'Never written; separate with commas.')}</span></label>
      </div>
      <div className="vision-field vision-prompt">
        <div className="vision-prompt-head"><span className="vision-field-label">{text('提示词', 'Prompt')}</span>
          <StudioSelect aria-label={text('提示词模板', 'Prompt template')} value={effectiveTemplateId} placeholder={text('已修改', 'Edited')} disabled={locked} options={[{ value: CUSTOM_TEMPLATE_ID, label: text('自定义', 'Custom') }, ...templates.map(item => ({ value: item.id, label: item.builtin ? text(item.name, item.nameEn || item.name) : item.name }))]} onValueChange={chooseTemplate}/>
          {selectedTemplate && settings.prompt !== selectedTemplate.prompt && <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" disabled={locked} onClick={() => update({ prompt: selectedTemplate.prompt })}><RotateCcw size={13}/>{text('恢复模板', 'Restore template')}</button>}
          {naming === null
            ? <button type="button" className="ui-btn ui-btn-sm" disabled={locked || !editingCustom || !settings.prompt.trim()} onClick={() => setNaming(template && !template.builtin ? template.name : '')}><Save size={13}/>{text('保存为模板', 'Save as template')}</button>
            : <form className="vision-template-name" onSubmit={event => { event.preventDefault(); saveTemplate(); }}>
              <input type="text" autoFocus maxLength={60} aria-label={text('模板名称', 'Template name')} placeholder={text('模板名称', 'Template name')} value={naming} onChange={event => setNaming(event.target.value)}/>
              <button type="submit" className="ui-btn ui-btn-sm ui-btn-primary" disabled={!naming.trim()}>{text('保存', 'Save')}</button>
              <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet" onClick={() => setNaming(null)}>{text('取消', 'Cancel')}</button></form>}
          {template && !template.builtin && naming === null && <button type="button" className="ui-btn ui-btn-sm ui-btn-quiet ui-btn-danger" disabled={locked} onClick={deleteTemplate}><Trash2 size={13}/>{text('删除模板', 'Delete template')}</button>}
        </div>
        <textarea aria-label={text('提示词', 'Prompt')} value={effectivePrompt} rows={9} maxLength={20000} disabled={locked || !editingCustom} spellCheck={false} onChange={event => update({ prompt: event.target.value })}/>
        <span className="vision-field-hint">{assisted ? text('{tags} 换成参考标签，{trigger} 换成触发词；没有 {tags} 时参考标签附在末尾。', '{tags} becomes the reference tags and {trigger} the trigger word; without {tags} the reference is appended.') : text('{trigger} 换成触发词，没有触发词时含它的行会删去。', '{trigger} becomes the trigger word; lines with it are dropped when there is none.')}</span>
      </div>
    </div>
    <footer className="vision-panel-actions">{running ? <div className="vision-running">{running}</div> : <>
      <button type="button" className="ui-btn ui-btn-primary" disabled={locked || !ready || !chosenScope} onClick={() => void start()}><Bot size={15}/>{text('开始打标', 'Start tagging')}</button>
      {error && <p role="alert" className="vision-error">{error}</p>}
      <OperationResult operation={latest} locked={locked} onUndo={onUndo} undoLabel={text('撤销本次打标', 'Undo this run')} done={done}>
        {!!changed && <button type="button" className="ui-link" onClick={onReview}>{text('查看标签', 'Review captions')}</button>}
      </OperationResult>
      {latest?.status === 'completed' && !latest.result.undone_by && !!result?.stopped_reason && <p role="alert" className="vision-error">{result.stopped_reason}</p>}
      {latest?.status === 'completed' && !latest.result.undone_by && !!result?.failures?.length && <details className="vision-failures">
        <summary>{text(`查看 ${result.failures.length} 张失败的图片`, `Show ${result.failures.length} failed images`)}</summary>
        <ul>{result.failures.map(item => <li key={item.rel_path}><strong>{item.rel_path}</strong>{item.error}</li>)}</ul>
      </details>}
    </>}</footer>
  </section>;
}
