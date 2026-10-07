import React from 'react';
import { Loader2, Play } from 'lucide-react';
import { ttsApi, type TtsCheckpoint, type TtsIssue, type TtsSampleBody } from '../../api/tts';
import { ApiError, type Job } from '../../api/types';
import Dialog from '../../components/Dialog';
import ConfigHelp from '../../components/ConfigHelp';
import { PathInput } from '../../components/PathBrowser';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import TtsGpuPicker from './TtsGpuPicker';
import { readSampleAttempt, sampleRequestKey } from './ttsResultRequests';
import './tts-results.css';

type Props = { checkpoint: TtsCheckpoint; onClose: () => void; onCreated: (job: Job) => void };
export default function TtsSampleDialog(props: Props) {
  return <SampleDialog key={props.checkpoint.source_job_id} {...props}/>;
}
function SampleDialog({ checkpoint, onClose, onCreated }: Props) {
  const text = useWorkspaceText();
  const [attempt, setAttempt] = React.useState(() => readSampleAttempt(checkpoint.source_job_id));
  const [chosen] = React.useState(checkpoint);
  const [prompt, setPrompt] = React.useState(attempt?.body.text || '');
  const [referenceAudio, setReferenceAudio] = React.useState(attempt?.body.reference_audio || '');
  const [referenceText, setReferenceText] = React.useState(attempt?.body.reference_text || '');
  const [seed, setSeed] = React.useState(String(attempt?.body.seed ?? 42));
  const [cfg, setCfg] = React.useState(String(attempt?.body.cfg_value ?? 2));
  const [steps, setSteps] = React.useState(String(attempt?.body.inference_timesteps ?? 10));
  const [gpu, setGpu] = React.useState<string[]>(attempt?.body.gpu_devices || []);
  const [gpuValid, setGpuValid] = React.useState(false);
  const [busy, setBusy] = React.useState(false), [resetting, setResetting] = React.useState(false);
  const [error, setError] = React.useState(attempt?.message || '');
  const [issues, setIssues] = React.useState<TtsIssue[]>([]);
  const [invalid, setInvalid] = React.useState<Record<string, string>>({});
  const pending = React.useRef(false), mounted = React.useRef(true);
  const id = React.useId();
  React.useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const current = attempt?.checkpoint || chosen;
  const unchanged = checkpoint.id === chosen.id && checkpoint.revision === chosen.revision;
  const eligible = checkpoint.can_preview && unchanged;
  const frozen = busy || !!attempt;
  const referenceParent = (() => {
    const path = referenceAudio.trim(), separator = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
    if (!path || separator < 0) return path ? '.' : undefined;
    if (separator === 0) return path[0];
    return /^[a-z]:$/i.test(path.slice(0, separator)) ? path.slice(0, separator + 1) : path.slice(0, separator);
  })();
  const fieldError = (name: string) => invalid[name] && <p id={`${id}-${name}-error`} role="alert" className="tts-result-error">{invalid[name]}</p>;
  const submit = async () => {
    if (pending.current || attempt?.blocked || !attempt && (!eligible || !gpuValid)) return;
    let request = attempt;
    setError(''); setIssues([]);
    if (!request) {
      const problems: Record<string, string> = {};
      if (!prompt.trim() || prompt.length > 4000) problems.text = text('填写 1–4000 个字符的试听文本。', 'Enter 1–4,000 characters of preview text.');
      if (!!referenceAudio.trim() !== !!referenceText.trim()) problems.reference_audio = text('参考音频和参考转写需要同时填写，或同时留空。', 'Fill in both reference audio and its transcript, or leave both empty.');
      if (!seed.trim() || !Number.isInteger(Number(seed)) || Number(seed) < 0 || Number(seed) >= 4294967296) problems.seed = text('种子须为 0–4294967295 的整数。', 'Seed must be an integer from 0 to 4294967295.');
      if (!cfg.trim() || !Number.isFinite(Number(cfg)) || Number(cfg) < 0 || Number(cfg) > 20) problems.cfg_value = text('CFG 须为 0–20 的数值。', 'CFG must be a number from 0 to 20.');
      if (!steps.trim() || !Number.isInteger(Number(steps)) || Number(steps) < 1 || Number(steps) > 100) problems.inference_timesteps = text('推理步数须为 1–100 的整数。', 'Inference steps must be an integer from 1 to 100.');
      setInvalid(problems);
      if (Object.keys(problems).length) return;
      const body: TtsSampleBody = { checkpoint_id: chosen.id, checkpoint_revision: chosen.revision, text: prompt, reference_audio: referenceAudio.trim(), reference_text: referenceText, seed: Number(seed), cfg_value: Number(cfg), inference_timesteps: Number(steps), gpu_devices: [...gpu] };
      request = { sourceJobId: chosen.source_job_id, checkpoint: structuredClone(chosen), key: crypto.randomUUID(), body, blocked: false };
      try { sessionStorage.setItem(sampleRequestKey(request.sourceJobId), JSON.stringify(request)); setAttempt(request); }
      catch { setError(text('浏览器无法保留试听请求，请允许网站存储后重试。', 'This browser cannot retain the preview request. Enable site storage and try again.')); return; }
    }
    pending.current = true; setBusy(true);
    try {
      const job = await ttsApi.createSample(request.sourceJobId, request.body, request.key);
      try { if (readSampleAttempt(request.sourceJobId)?.key === request.key) sessionStorage.removeItem(sampleRequestKey(request.sourceJobId)); } catch { /* Retaining the original request still replays the same job. */ }
      if (mounted.current) { setAttempt(null); onCreated(job); }
    } catch (failure) {
      if (!mounted.current) return;
      setError(formatApiError(failure));
      if (failure instanceof ApiError) {
        const reported = Array.isArray(failure.details?.issues) ? failure.details.issues as TtsIssue[] : [];
        setIssues(reported.filter(issue => issue.message !== failure.message));
        if (failure.status >= 400 && failure.status < 500 && failure.code !== 'tts.request_pending') {
          const blocked = { ...request, blocked: true, message: formatApiError(failure) };
          setAttempt(blocked);
          try { sessionStorage.setItem(sampleRequestKey(request.sourceJobId), JSON.stringify(blocked)); } catch { /* The original key remains safe to replay. */ }
        }
      }
    } finally { pending.current = false; if (mounted.current) setBusy(false); }
  };
  const reset = () => {
    try { sessionStorage.removeItem(sampleRequestKey(checkpoint.source_job_id)); }
    catch { setError(text('无法移除原请求，请检查浏览器的网站存储权限。', 'Cannot remove the original request. Check site storage permissions.')); setResetting(false); return; }
    setAttempt(null); setResetting(false); setError(''); setIssues([]); setInvalid({});
  };
  return <><Dialog title={text('生成试听音频', 'Generate preview audio')} onClose={onClose} closeDisabled={busy} wide>
    <form className="tts-sample-form" onSubmit={event => { event.preventDefault(); void submit(); }} aria-busy={busy}>
      <p className="tts-result-muted">{current.name} · {text('更新步数', 'Update step')} {current.step ?? text('未知', 'Unknown')}</p>
      <details className="tts-result-identity"><summary>{text('检查点身份', 'Checkpoint identity')}</summary><dl><div><dt>ID</dt><dd tabIndex={0}>{current.id}</dd></div><div><dt>{text('版本', 'Revision')}</dt><dd tabIndex={0}>{current.revision}</dd></div></dl></details>
      {attempt && <p role="status" className="tts-result-muted">{attempt.blocked ? text('原请求无法继续。请先确认任务状态，再决定是否创建新试听。', 'The original request cannot continue. Check the task status before creating another preview.') : text('试听结果尚待确认。继续确认会重放原请求，不会重复创建任务。', 'The preview request needs confirmation. Confirming replays the original request without creating a duplicate job.')}</p>}
      {!eligible && !attempt && <p role="alert" className="tts-result-error">{!unchanged ? text('检查点内容已更新，请关闭后重新选择。', 'This checkpoint changed. Close the dialog and select it again.') : checkpoint.unavailable_reason?.message || text('该检查点暂不能试听。', 'This checkpoint cannot be previewed.')}</p>}
      {error && <p role="alert" className="tts-result-error">{error}</p>}
      {!!issues.length && <ul className="tts-result-issues">{issues.map((issue, index) => <li key={`${issue.code}:${index}`} data-severity={issue.severity}>{issue.message}</li>)}</ul>}
      <fieldset disabled={frozen}>
        <label htmlFor={`${id}-text`}>{text('试听文本', 'Preview text')}</label><textarea id={`${id}-text`} rows={3} maxLength={4000} value={prompt} onChange={event => setPrompt(event.target.value)} aria-invalid={!!invalid.text} aria-describedby={invalid.text ? `${id}-text-error` : undefined}/>{fieldError('text')}
        <div className="tts-sample-reference"><div className="tts-sample-label"><label>{text('参考音频', 'Reference audio')}</label><ConfigHelp label={text('参考音频说明', 'About reference audio')}>{text('可选，填写训练服务可访问的 44100 Hz 单声道 PCM WAV 文件。使用参考音频时须提供其对应的完整转写；不填写时，参考音频和参考转写同时留空。', 'Optional server-accessible 44100 Hz mono PCM WAV file. Provide its complete transcript when using reference audio; otherwise leave both fields empty.')}</ConfigHelp></div><PathInput ariaLabel={text('参考音频路径', 'Reference audio path')} value={referenceAudio} onChange={setReferenceAudio} browsePath={referenceParent}/>{fieldError('reference_audio')}<p className="tts-result-muted">{text('留空时不使用参考录音。', 'Leave empty to generate without reference audio.')}</p>
          <label htmlFor={`${id}-reference-text`}>{text('参考转写', 'Reference transcript')}</label><textarea id={`${id}-reference-text`} rows={2} value={referenceText} onChange={event => setReferenceText(event.target.value)}/></div>
        <div className="tts-sample-numbers">{[
          { field: 'seed', zh: '请求种子', en: 'Requested seed', value: seed, change: setSeed, hint: text('0–4294967295 的整数；实际使用值见生成结果。', 'Integer from 0 to 4294967295. The generated result records the seed actually used.') },
          { field: 'cfg_value', zh: 'CFG 引导强度', en: 'CFG strength', value: cfg, change: setCfg, hint: text('范围 0–20。', 'Range: 0–20.') },
          { field: 'inference_timesteps', zh: '推理步数', en: 'Inference steps', value: steps, change: setSteps, hint: text('范围 1–100，填写整数。', 'Integer from 1 to 100.') },
        ].map(item => <div key={item.field}><label htmlFor={`${id}-${item.field}`}>{text(item.zh, item.en)}</label><input id={`${id}-${item.field}`} type="text" inputMode={item.field === 'cfg_value' ? 'decimal' : 'numeric'} value={item.value} onChange={event => item.change(event.target.value)} aria-invalid={!!invalid[item.field]} aria-describedby={`${id}-${item.field}-hint${invalid[item.field] ? ` ${id}-${item.field}-error` : ''}`}/><p id={`${id}-${item.field}-hint`} className="tts-result-muted">{item.hint}</p>{fieldError(item.field)}</div>)}</div>
      </fieldset>
      <div className="tts-sample-gpu"><span>{text('运行显卡', 'Run on GPU')}</span><TtsGpuPicker value={gpu} onChange={setGpu} disabled={frozen} onValidityChange={setGpuValid}/></div>
      <footer className="tts-result-actions"><button type="button" className="ui-btn" disabled={busy} onClick={onClose}>{text('关闭', 'Close')}</button>{attempt && <button type="button" className="ui-btn" disabled={busy} onClick={() => setResetting(true)}>{text('重新创建试听', 'Create another preview')}</button>}<button type="submit" className="ui-btn ui-btn-primary" disabled={busy || !!attempt?.blocked || !attempt && (!eligible || !gpuValid)}>{busy ? <Loader2 size={14} className="animate-spin"/> : <Play size={14}/>} {busy ? text('正在确认…', 'Confirming…') : attempt ? text('继续确认原请求', 'Confirm original request') : text('加入试听队列', 'Queue preview')}</button></footer>
    </form>
  </Dialog>{resetting && <Dialog nested title={text('重新创建试听', 'Create another preview')} onClose={() => setResetting(false)}><p>{text('原请求可能已创建任务，请先查看队列。继续会放弃本地原请求记录；再次提交将创建新的试听任务。', 'The original request may already have created a job. Check the queue first. Continuing discards the local request record; submitting again creates a new preview job.')}</p><div className="tts-result-actions"><button type="button" className="ui-btn" onClick={() => setResetting(false)}>{text('保留原请求', 'Keep original request')}</button><button type="button" className="ui-btn ui-btn-primary" onClick={reset}>{text('已确认，重新创建', 'Confirmed, create another')}</button></div></Dialog>}</>;
}
