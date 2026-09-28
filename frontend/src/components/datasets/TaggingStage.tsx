import { useState, type ReactNode } from 'react';
import { useWorkspaceText } from '../../utils/workspaceText';
import { SlidingIndicator } from '../motion';
import AutoTagPanel from './AutoTagPanel';
import VlmTagPanel from './VlmTagPanel';
import type { PipelineOperation } from './DatasetPipelinePanel';

type Mode = 'tagger' | 'vlm' | 'assist';
const MODE_KEY = 'studio.tagging.mode';

/** Image tagging: a local tagger, a vision model, or the tagger's tags checked by a vision model. */
export default function TaggingStage({ projectId, versionId, locked, latest, active, progress, onStart, onUndo, onReview }: {
  projectId: string; versionId: string; locked: boolean; latest: (action: string) => PipelineOperation | undefined;
  active?: PipelineOperation; progress: ReactNode;
  onStart: (body: Record<string, unknown>) => Promise<void>; onUndo: (id: string) => void; onReview: () => void;
}) {
  const text = useWorkspaceText();
  const [mode, setMode] = useState<Mode>(() => { try { const saved = localStorage.getItem(MODE_KEY); return saved === 'vlm' || saved === 'assist' ? saved : 'tagger'; } catch { return 'tagger'; } });
  const choose = (next: Mode) => { setMode(next); try { localStorage.setItem(MODE_KEY, next); } catch { /* the choice lasts for this page only */ } };
  const header = <div className="ui-segmented tagging-modes" role="group" aria-label={text('打标方式', 'Tagging method')}>
    {([['tagger', text('Tagger 模型', 'Tagger model')], ['vlm', text('视觉大模型', 'Vision model')], ['assist', text('辅助打标', 'Assisted tagging')]] as const).map(([value, label]) =>
      <button key={value} type="button" aria-pressed={mode === value} onClick={() => choose(value)}>{label}</button>)}
    <SlidingIndicator className="ui-segmented-thumb"/>
  </div>;
  const action = mode === 'tagger' ? 'autotag' : mode === 'vlm' ? 'vlmtag' : 'assisttag';
  // A run of another method keeps its progress above this panel.
  const own = active?.action === action;
  const common = { projectId, versionId, locked, header, onStart, onUndo, onReview, running: own ? progress : null };
  return <>{active && !own && progress}{mode === 'tagger' ? <AutoTagPanel {...common} latest={latest('autotag')}/>
    : <VlmTagPanel key={mode} mode={mode} {...common} latest={latest(action)}/>}</>;
}
