import TtsResultsView, { SingleSampleResult } from './TtsResultsView';

export default function TtsJobOutputs({ jobId, training, live = false }: { jobId: string; training: boolean; live?: boolean }) {
  return training ? <TtsResultsView key={jobId} scope={{ jobId }} live={live}/> : <SingleSampleResult key={jobId} jobId={jobId} live={live}/>;
}
