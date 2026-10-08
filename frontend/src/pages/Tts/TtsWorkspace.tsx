import { Link, Navigate, useSearchParams } from 'react-router-dom';
import { useWorkspaceText } from '../../utils/workspaceText';
import './tts-import.css';

export default function TtsWorkspace() {
  const text = useWorkspaceText();
  const [params] = useSearchParams();
  const job = params.get('job');
  if (job) return <Navigate replace to={`/jobs/${encodeURIComponent(job)}?tab=audio`}/>;
  return <section className="tts-legacy-entry">
    <h1>{text('语音训练', 'Speech training')}</h1>
    <p>{text('从项目页创建语音项目，在版本内准备音频、设置参数和查看训练结果。', 'Create a speech project, then prepare audio, configure training and review results in its versions.')}</p>
    <p>{text('旧配置和此浏览器的未保存草稿可在版本的训练参数页通过“导入旧草稿”迁入。历史训练与试听仍可从任务队列打开。', 'Import the legacy configuration or this browser’s unsaved draft from a version’s training parameters page. Historical training and previews remain accessible from the queue.')}</p>
    <nav><Link className="ui-btn ui-btn-primary" to="/projects">{text('前往项目', 'Open projects')}</Link><Link className="ui-btn" to="/queue?view=history&type=tts_train">{text('历史语音训练', 'Speech training history')}</Link><Link className="ui-btn" to="/queue?view=history&type=tts_sample">{text('历史试听任务', 'Preview history')}</Link></nav>
  </section>;
}
