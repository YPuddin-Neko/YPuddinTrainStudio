import { useWorkspaceText } from '../utils/workspaceText';
import { SlidingIndicator } from './motion';

export default function ParameterModeToggle({ advanced, onChange }: { advanced: boolean; onChange: (advanced: boolean) => void }) {
  const text = useWorkspaceText();
  return <div className="parameter-mode-toggle ui-segmented" role="group" aria-label={text('参数显示模式', 'Parameter display mode')}>
    <button type="button" aria-pressed={!advanced} onClick={() => onChange(false)}>{text('简单', 'Simple')}</button>
    <button type="button" aria-pressed={advanced} onClick={() => onChange(true)}>{text('高级', 'Advanced')}</button>
    <SlidingIndicator className="ui-segmented-thumb"/>
  </div>;
}
