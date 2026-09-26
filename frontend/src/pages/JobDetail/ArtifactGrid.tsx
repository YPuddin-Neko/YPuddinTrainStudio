import React from 'react';
import { useTranslation } from 'react-i18next';
import { Download, Layers, Package } from 'lucide-react';
import { apiUrl } from '../../api/client';
import type { JobCheckpoint } from '../../api/types';
import CopyButton from '../../components/CopyButton';
import { LazyImage } from '../../components/Loading';
import { epochAt, epochText, inEpochs, parseEpochQuery } from '../../utils/epochFilter';
import { formatBytes, formatTime } from '../../utils/format';
import { sampleSource } from '../../utils/sampleMedia';
import { useWorkspaceText } from '../../utils/workspaceText';
import EpochSearch from './EpochSearch';
import './job-samples.css';

function fileName(path: string): string {
  return path.replace(/\\/g, '/').split('/').pop() || path;
}

function SkeletonGrid() {
  return <div className="artifact-grid" aria-busy="true">{Array.from({ length: 5 }, (_, index) => <div key={index} className="artifact-card artifact-card-skeleton" aria-hidden="true">
    <span className="artifact-preview ui-skeleton"/>
    <div className="artifact-body">{Array.from({ length: 4 }, (_, line) => <span key={line} className="ui-skeleton"/>)}</div>
  </div>)}</div>;
}

/** Saved weights and training states as cards, newest first, each with the preview of its step. */
export default function ArtifactGrid({ checkpoints, stepsPerEpoch, loaded, resuming, canResume, onResume, onOpenSample }: {
  checkpoints: JobCheckpoint[]; stepsPerEpoch?: number | null; loaded: boolean; resuming: boolean; canResume: boolean;
  onResume: (checkpoint: JobCheckpoint) => void; onOpenSample: (url: string) => void;
}) {
  const { t } = useTranslation();
  const text = useWorkspaceText();
  const [query, setQuery] = React.useState('');
  const ranges = React.useMemo(() => parseEpochQuery(query), [query]);
  const shown = React.useMemo(() => [...checkpoints].reverse().filter(item => inEpochs(epochAt(item, stepsPerEpoch), ranges)), [checkpoints, stepsPerEpoch, ranges]);
  const kindLabel = (kind: string) => kind === 'full' ? text('训练状态 · 可续训', 'Training state') : kind === 'model' ? text('全量模型组件', 'Full-model components') : kind === 'weights' ? text('权重', 'Weights') : kind;

  if (!loaded) return <section className="artifact-browser" aria-label={text('产物', 'Outputs')}><SkeletonGrid/></section>;
  if (!checkpoints.length) return <div className="sample-empty"><Layers size={30} aria-hidden="true"/><p>{text('暂无产物', 'No outputs yet')}</p><span>{text('训练到保存步数或轮次后，权重和训练状态会出现在这里。', 'Weights and training states appear here once training reaches a save step or epoch.')}</span></div>;

  return <section className="artifact-browser" aria-label={text('产物', 'Outputs')}>
    <div className="sample-toolbar">
      <EpochSearch value={query} onChange={setQuery} label={text('按轮次搜索产物', 'Search outputs by epoch')}/>
      <span className="sample-toolbar-count">{ranges && ranges !== 'invalid' ? text(`找到 ${shown.length} 个，共 ${checkpoints.length} 个`, `${shown.length} of ${checkpoints.length}`) : text(`共 ${checkpoints.length} 个`, `${checkpoints.length} outputs`)}</span>
    </div>
    {!shown.length ? <div className="sample-empty"><Package size={26} aria-hidden="true"/><p>{text('没有符合轮次的产物', 'No outputs in these epochs')}</p><button type="button" className="ui-link" onClick={() => setQuery('')}>{text('清除搜索', 'Clear search')}</button></div>
      : <div className="artifact-grid">{shown.map(item => {
        const name = fileName(item.path);
        const epoch = epochAt(item, stepsPerEpoch);
        const loss = typeof item.loss === 'number' && Number.isFinite(item.loss) ? String(Number(item.loss.toPrecision(5))) : '—';
        return <article key={`${item.path}-${item.step}`} className="artifact-card" aria-label={name}>
          <button type="button" className="artifact-preview" disabled={!item.sample_url} onClick={() => item.sample_url && onOpenSample(item.sample_url)}
            aria-label={item.sample_url ? text(`查看第 ${item.step} 步的采样图`, `Open the step ${item.step} preview`) : text('此步没有采样图', 'No preview at this step')}>
            {item.sample_url ? <LazyImage src={sampleSource(item.sample_url)} alt="" loading="lazy"/>
              : <span className="artifact-preview-empty"><Package size={22} aria-hidden="true"/>{text('此步没有采样图', 'No preview at this step')}</span>}
            <span className="artifact-kind" data-kind={item.kind}>{kindLabel(item.kind)}{item.ema ? ' · EMA' : ''}</span>
          </button>
          <div className="artifact-body">
            <strong className="artifact-name" title={name}>{name}</strong>
            <dl className="artifact-facts">
              <div><dt>{text('轮次', 'Epoch')}</dt><dd>{epochText(epoch)}</dd></div>
              <div><dt>{text('步数', 'Step')}</dt><dd>{item.step}</dd></div>
              <div><dt>{text('保存时 Loss', 'Loss at save')}</dt><dd>{loss}</dd></div>
              <div><dt>{text('大小', 'Size')}</dt><dd>{formatBytes(item.size)}</dd></div>
              <div className="is-wide"><dt>{text('保存时间', 'Saved')}</dt><dd>{formatTime(item.created_at)}</dd></div>
            </dl>
            <div className="artifact-actions">
              {item.kind === 'full' ? <button type="button" className="ui-btn ui-btn-sm" disabled={resuming || !canResume} onClick={() => onResume(item)}>{t('job.continueTraining')}</button>
                : item.artifact_id ? <a className="ui-btn ui-btn-sm" href={apiUrl(`/artifacts/${item.artifact_id}/download`)}><Download size={14}/>{t('job.download')}</a>
                  : <span>{t('job.downloadUnavailable')}</span>}
              <CopyButton value={item.path} label={text('复制本机路径', 'Copy local path')}/>
            </div>
          </div>
        </article>;
      })}</div>}
  </section>;
}
