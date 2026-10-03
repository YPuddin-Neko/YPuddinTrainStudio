import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ArrowLeft, ArrowRight, Check, Cpu, Download, FolderOpen, Globe2, KeyRound, Languages, Loader2 } from 'lucide-react';
import { apiClient, READ_TIMEOUT_MS } from '../../api/client';
import type { Settings } from '../../api/types';
import BrandMark from '../../components/BrandMark';
import { DownloadSourceFields, type DownloadSources } from '../Settings/DownloadPreferences';
import { DOWNLOAD_SOURCE_DEFAULTS } from '../../utils/downloadSources';
import { useWorkspaceText } from '../../utils/workspaceText';
import { prefersReducedMotion } from '../../utils/motion';
import { formatApiError } from '../../utils/errors';
import { StorageStep, KeysStep, ModelsStep, RuntimeStep, type KeyDraft } from './SetupResources';
import '../../styles/settings.css';
import './setup.css';

const stepIcons = [Languages, Globe2, FolderOpen, KeyRound, Download, Cpu];
export default function Setup() {
  const navigate = useNavigate();
  const { i18n } = useTranslation();
  const text = useWorkspaceText();
  const [settings, setSettings] = React.useState<Settings | null>(null);
  const [step, setStep] = React.useState(0);
  const [language, setLanguage] = React.useState<'zh-CN' | 'en'>('zh-CN');
  const [sources, setSources] = React.useState<DownloadSources>(DOWNLOAD_SOURCE_DEFAULTS);
  const [modelContinue, setModelContinue] = React.useState(false);
  const [keys, setKeys] = React.useState<KeyDraft>({ huggingface: '', modelscope: '' });
  const [busy, setBusy] = React.useState(false);
  const busyRef = React.useRef(false);
  const [error, setError] = React.useState('');
  const [attempt, retry] = React.useReducer(value => value + 1, 0);
  const body = React.useRef<HTMLDivElement>(null);
  const heading = React.useRef<HTMLHeadingElement>(null);
  const previousStep = React.useRef(0);
  React.useEffect(() => {
    const controller = new AbortController();
    setError('');
    void apiClient.get<Settings>('/settings', { silent: true, signal: controller.signal, timeout: READ_TIMEOUT_MS }).then(value => {
      if (controller.signal.aborted) return;
      setSettings(value); setLanguage(value.ui.language); setSources(value.downloads ?? DOWNLOAD_SOURCE_DEFAULTS);
      void i18n.changeLanguage(value.ui.language);
      document.documentElement.classList.toggle('dark', value.ui.theme === 'dark' || value.ui.theme === 'system' && (window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false));
    }).catch(failure => { if (!controller.signal.aborted) setError(formatApiError(failure)); });
    return () => controller.abort();
  }, [i18n, attempt]);
  const loaded = settings !== null;
  React.useLayoutEffect(() => {
    const direction = step >= previousStep.current ? 1 : -1;
    previousStep.current = step;
    if (!body.current || prefersReducedMotion()) return;
    const animation = body.current.animate?.([{ opacity: 0, transform: `translateX(${direction * 16}px)` }, { opacity: 1, transform: 'translateX(0)' }], { duration: 230, easing: 'cubic-bezier(.2,.75,.25,1)' });
    return () => animation?.cancel();
  }, [step, loaded]);
  const move = (next: number) => { setError(''); if (next === 4) setModelContinue(false); setStep(next); requestAnimationFrame(() => heading.current?.focus({ preventScroll: true })); };
  const persist = async (payload: unknown) => {
    const result = await apiClient.put<Settings>('/settings', payload, { silent: true });
    setSettings(result);
    window.dispatchEvent(new CustomEvent('studio.settings.changed', { detail: result }));
    return result;
  };
  const saveStep = async (finish = false) => {
    if (busyRef.current) return;
    busyRef.current = true; setBusy(true); setError('');
    try {
      if (finish) {
        await persist({ ui: { language, onboarding_completed: true } });
        try { localStorage.setItem('i18nextLng', language); } catch { /* The server retains the preference. */ }
        navigate('/', { replace: true });
      } else {
        if (step === 0) {
          await persist({ ui: { language } });
          try { localStorage.setItem('i18nextLng', language); } catch { /* The server retains the preference. */ }
        }
        if (step === 1) await persist({ downloads: sources });
        if (step === 3) {
          for (const provider of ['huggingface', 'modelscope'] as const) {
            if (!keys[provider].trim()) continue;
            try { await apiClient.put(`/credentials/${provider}`, { token: keys[provider].trim() }, { silent: true }); }
            catch { throw new Error(text('访问密钥保存失败，请检查后重试。', 'Could not save the access key. Check it and retry.')); }
            setKeys(old => ({ ...old, [provider]: '' }));
          }
          window.dispatchEvent(new Event('credentials.changed'));
        }
        move(step + 1);
      }
    } catch (failure) { setError(formatApiError(failure)); }
    finally { busyRef.current = false; setBusy(false); }
  };
  const stepNames = [text('语言', 'Language'), text('软件下载源', 'Package sources'), text('存储位置', 'Storage'), text('访问密钥', 'Access keys'), text('模型下载', 'Models'), text('运行环境', 'Environment')];
  const titles = [text('先选择你的语言', 'Choose your language'), text('从哪里下载软件？', 'Where should software come from?'), text('查看文件存储位置', 'Review your storage folders'), text('添加模型访问密钥', 'Add your model access keys'), text('准备第一个模型', 'Prepare your first model'), text('查看当前运行环境', 'Review your runtime environment')];
  const descriptions = [text('欢迎使用 YPuddin Train Studio。几步完成首次设置。', 'Welcome to YPuddin Train Studio. Let’s set up your workspace.'), text('默认自动选择，也可以指定常用的官方站点或镜像。', 'Let Studio choose, or select an official site or mirror.'), text('训练数据、模型和缓存将分别保存在以下目录。', 'Your training data, models and caches are stored separately.'), text('用于下载需要授权的模型，这一步可以跳过。', 'For models that require authorization. This step is optional.'), text('按模型类型下载训练组件，也可以稍后再添加。', 'Download the components you need, or add them later.'), text('以下是训练器当前检测到的软件与设备。', 'Here is the software and hardware detected by Studio.')];
  const Icon = stepIcons[step];
  return <main className="setup-page">
    <section className="setup-window" aria-label={text('首次设置', 'First-time setup')}>
      <aside className="setup-sidebar"><div className="setup-brand"><BrandMark/><span>YPuddin<strong>Train Studio</strong></span></div>
        <div className="setup-welcome"><span>{text('开始之前', 'BEFORE YOU BEGIN')}</span><h1>{text('让训练，\n从这里开始。', 'Your workspace,\nready to train.')}</h1></div>
        <ol className="setup-steps">{stepNames.map((name, index) => <li key={index} aria-current={index === step ? 'step' : undefined} className={index < step ? 'is-complete' : ''}><span className="setup-step-number">{index < step ? <Check size={14}/> : index + 1}</span><span>{name}</span></li>)}</ol>
        <p className="setup-sidebar-note">{text('所有选项都可以稍后在设置中修改。', 'You can change these options later in Settings.')}</p>
      </aside>
      <div className="setup-main">
        <header className="setup-topline"><span>{text('首次设置', 'First-time setup')}<span className="setup-step-count">{step + 1} / 6</span></span>{step === 0 && <button type="button" className="setup-skip ui-link" disabled={!settings || busy} onClick={() => void saveStep(true)}>{text('跳过引导', 'Skip setup')}<ArrowRight size={14}/></button>}</header>
        {!settings ? <div className="setup-loading">{error ? <div role="alert" className="setup-error">{error}<button className="ui-btn" onClick={retry}>{text('重试', 'Retry')}</button></div> : <Loader2 size={24} className="animate-spin" aria-label={text('正在读取设置', 'Loading settings')}/>}</div> : <>
          <div ref={body} className="setup-body" key={step}><span className="setup-page-icon"><Icon size={26} strokeWidth={1.7}/></span><h2 tabIndex={-1} ref={heading}>{titles[step]}</h2><p className="setup-description">{descriptions[step]}</p>
            <div className="setup-step-content">{step === 0 && <fieldset className="setup-languages" disabled={busy}><legend className="sr-only">{text('界面语言', 'Interface language')}</legend>{([{ value: 'zh-CN', label: '简体中文', detail: '中文' }, { value: 'en', label: 'English', detail: '英语' }] as const).map(option => <label key={option.value} className={language === option.value ? 'is-selected' : ''}><input type="radio" name="setup-language" value={option.value} checked={language === option.value} onChange={() => { setLanguage(option.value); void i18n.changeLanguage(option.value); }}/><span><strong>{option.label}</strong><small>{option.value === 'zh-CN' ? 'Chinese, Simplified' : text('英语', 'English, International')}</small></span><span className="setup-radio-mark">{language === option.value && <Check size={14}/>}</span></label>)}</fieldset>}
            {step === 1 && <DownloadSourceFields value={sources} onChange={setSources} disabled={busy}/>}
            {step === 2 && <StorageStep settings={settings}/>}
            {step === 3 && <KeysStep draft={keys} onChange={setKeys} disabled={busy}/>}
            {step === 4 && <ModelsStep onContinueChange={setModelContinue}/>}
            {step === 5 && <RuntimeStep/>}</div>
          </div>
          <footer className="setup-footer">{error && <div role="alert" className="setup-error">{error}</div>}<div className="setup-footer-actions"><div>{step > 0 && <button type="button" className="ui-btn setup-back" disabled={busy} onClick={() => move(step - 1)}><ArrowLeft size={15}/>{text('上一步', 'Back')}</button>}</div><div className="setup-forward-actions">{(step === 3 || step === 4) && <button type="button" className="ui-btn setup-optional" disabled={busy} onClick={() => { if (step === 3) setKeys({ huggingface: '', modelscope: '' }); move(step + 1); }}>{text('暂时跳过', 'Skip for now')}</button>}<button type="button" className="ui-btn ui-btn-primary setup-next" disabled={busy || step === 3 && !Object.values(keys).some(key => key.trim()) || step === 4 && !modelContinue} onClick={() => void saveStep(step === 5)}>{busy ? <Loader2 size={16} className="animate-spin"/> : null}{step === 5 ? text('进入训练器', 'Open Studio') : step === 3 && Object.values(keys).some(key => key.trim()) ? text('保存并继续', 'Save and continue') : text('下一步', 'Next')}{!busy && <ArrowRight size={16}/>}</button></div></div></footer>
        </>}
      </div>
    </section>
  </main>;
}
