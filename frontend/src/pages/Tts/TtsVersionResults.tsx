import TtsResultsView from './TtsResultsView';

export default function TtsVersionResults({ projectId, versionId, engine }: { projectId: string; versionId: string; engine?: string }) {
  return <TtsResultsView key={`${projectId}:${versionId}`} scope={{ projectId, versionId }} engine={engine}/>;
}
