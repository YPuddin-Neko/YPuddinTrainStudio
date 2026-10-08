import { SlidingIndicator } from '../../components/motion';
import { useWorkspaceText } from '../../utils/workspaceText';
export default function PresetTypeNavigation({ value, onChange, disabled = false }: { value: 'image' | 'tts'; onChange: (value: 'image' | 'tts') => void; disabled?: boolean }) {
  const text = useWorkspaceText();
  return <div className="ui-segmented presets-type-navigation" role="group" aria-label={text('预设类型', 'Preset type')}>
    <button type="button" aria-pressed={value === 'image'} disabled={disabled} onClick={() => { if (value !== 'image') onChange('image'); }}>{text('图像训练', 'Image training')}</button>
    <button type="button" aria-pressed={value === 'tts'} disabled={disabled} onClick={() => { if (value !== 'tts') onChange('tts'); }}>{text('语音训练', 'Speech training')}</button><SlidingIndicator className="ui-segmented-thumb"/>
  </div>;
}
