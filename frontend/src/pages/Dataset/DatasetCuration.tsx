import { useQuery } from '@tanstack/react-query';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { apiClient } from '../../api/client';
import type { DatasetInfo, DatasetSource } from '../../api/types';
import StudioSelect from '../../components/StudioSelect';
import { projectUrl } from '../../utils/projectVersions';
import { useWorkspaceText } from '../../utils/workspaceText';
import { formatApiError } from '../../utils/errors';
import { DatasetWorkspace } from './Dataset';

export default function DatasetCuration() {
  const { id = '', versionId = '' } = useParams();
  const [params, setParams] = useSearchParams();
  const text = useWorkspaceText();
  const query = useQuery({
    queryKey: ['curation-folders', id, versionId],
    queryFn: ({signal}) => apiClient.get<Array<DatasetInfo | DatasetSource>>(`/projects/${id}/datasets`, { params: {version_id: versionId, include_cache: false}, signal, silent: true }),
  });
  const sources = (query.data || []).map(item => 'source' in item ? (item as DatasetInfo).source : item as DatasetSource).filter(source => !source.is_reg);
  const current = sources.find(source => source.id === params.get('dataset')) || sources[0];
  const selector = <label className="dataset-curation-folder">{text('目录', 'Folder')}<StudioSelect aria-label={text('筛选目录', 'Folder to curate')} value={current?.id || ''} options={sources.map(source => ({ value: source.id, label: source.path.replace(/\\/g,'/').split('/').filter(Boolean).pop() || source.id }))} onValueChange={value => setParams(previous => { const next = new URLSearchParams(previous); next.set('dataset', value); return next; })}/></label>;
  if (current) return <DatasetWorkspace key={current.id} id={current.id} curation selector={selector}/>;
  return <div className="dataset-curation-empty">
    <Link to={projectUrl(id,versionId,'data')}><ArrowLeft size={14}/>{text('返回训练数据','Back to training data')}</Link>
    <h1>{text('训练集筛选','Training set curation')}</h1>
    {query.isError ? <div role="alert">{formatApiError(query.error)}<button onClick={() => void query.refetch()}>{text('重试','Retry')}</button></div> : query.isPending ? <p role="status">{text('正在读取训练目录…','Loading training folders…')}</p> : <p>{text('当前版本还没有训练图片。','This version has no training images yet.')}</p>}
  </div>;
}
