import TtsResultsView from './TtsResultsView';

export default function TtsVersionResults({ projectId, versionId }: { projectId: string; versionId: string }) {
  return <TtsResultsView key={`${projectId}:${versionId}`} scope={{ projectId, versionId }}/>;
}
