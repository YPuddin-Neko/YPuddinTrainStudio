import TtsResultsView, { SingleSampleResult } from './TtsResultsView';

export default function TtsJobOutputs({ jobId, training, live = false, engine }: { jobId: string; training: boolean; live?: boolean; engine?: string }) {
  return training ? <TtsResultsView key={jobId} scope={{ jobId }} live={live} engine={engine}/> : <SingleSampleResult key={jobId} jobId={jobId} live={live}/>;
}
