import { useWorkspaceText } from '../utils/workspaceText';

export default function ParameterModeToggle({ advanced, onChange }: { advanced: boolean; onChange: (advanced: boolean) => void }) {
  const text = useWorkspaceText();
  return <div className="parameter-mode-toggle" role="group" aria-label={text('参数显示模式', 'Parameter display mode')}>
    <button type="button" aria-pressed={!advanced} onClick={() => onChange(false)}>{text('简单', 'Simple')}</button>
    <button type="button" aria-pressed={advanced} onClick={() => onChange(true)}>{text('高级', 'Advanced')}</button>
  </div>;
}
