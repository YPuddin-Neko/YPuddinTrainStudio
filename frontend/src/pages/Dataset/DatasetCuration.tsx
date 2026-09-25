import { Navigate, useParams, useSearchParams } from 'react-router-dom';
import { projectUrl } from '../../utils/projectVersions';

/** Curation lives in the training data tabs; older links keep working. */
export default function DatasetCuration() {
  const { id = '', versionId = '' } = useParams();
  const [params] = useSearchParams();
  const dataset = params.get('dataset');
  return <Navigate replace to={`${projectUrl(id, versionId, 'data')}&data_step=curate${dataset ? `&dataset=${encodeURIComponent(dataset)}` : ''}`}/>;
}
