import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { apiClient } from '../../api/client';
import { Artifact, DatasetSource, Job, JobListResponse, Project } from '../../api/types';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { PathInput } from '../../components/PathBrowser';
import { Database, Layers, Box, SlidersHorizontal, Plus, ExternalLink } from 'lucide-react';

type Tab = 'datasets' | 'config' | 'jobs' | 'artifacts';

const fmtTime = (t: number | string | null | undefined) =>
  t == null ? '--' : typeof t === 'number' ? new Date(t * 1000).toLocaleString() : new Date(t).toLocaleString();

export default function ProjectDetail() {
  const { id } = useParams<{ id: string }>();
  const [project, setProject] = React.useState<Project | null>(null);
  const [tab, setTab] = React.useState<Tab>('datasets');

  const [datasets, setDatasets] = React.useState<DatasetSource[]>([]);
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [artifacts, setArtifacts] = React.useState<Artifact[]>([]);

  // 注册数据集表单
  const [regOpen, setRegOpen] = React.useState(false);
  const [regPath, setRegPath] = React.useState('');
  const [regRepeats, setRegRepeats] = React.useState(1);
  const [regCaptionExt, setRegCaptionExt] = React.useState('.txt');
  const [regIsReg, setRegIsReg] = React.useState(false);
  const [regPriorWeight, setRegPriorWeight] = React.useState(1.0);
  const [regClassPrompt, setRegClassPrompt] = React.useState('');
  const [registering, setRegistering] = React.useState(false);

  const fetchProject = React.useCallback(() => {
    if (id) apiClient.get<Project>(`/projects/${id}`).then(setProject).catch(console.error);
  }, [id]);

  const fetchDatasets = React.useCallback(() => {
    if (id)
      apiClient
        .get<Array<DatasetSource | { source: DatasetSource }>>(`/projects/${id}/datasets`)
        .then((d) =>
          setDatasets(
            Array.isArray(d) ? d.map((item: any) => (item && item.source ? item.source : item)) : []
          )
        )
        .catch(console.error);
  }, [id]);

  React.useEffect(() => {
    fetchProject();
    fetchDatasets();
    if (id) {
      apiClient.get<JobListResponse | Job[]>(`/jobs`, { params: { project_id: id } })
        .then((res) => setJobs(Array.isArray(res) ? res : res.items || []))
        .catch(console.error);
      apiClient.get<Artifact[]>(`/artifacts`, { params: { project_id: id } })
        .then((res) => setArtifacts(Array.isArray(res) ? res : []))
        .catch(console.error);
    }
  }, [id, fetchProject, fetchDatasets]);

  // 数据集变化事件 → 刷新数据集列表
  useEventStream(EVENT_TYPES.DATASET_CHANGED, () => {
    fetchDatasets();
  });

  const handleRegister = () => {
    if (!id || !regPath.trim()) return;
    setRegistering(true);
    apiClient.post(`/projects/${id}/datasets`, {
      path: regPath.trim(),
      repeats: regRepeats,
      caption_ext: regCaptionExt,
      is_reg: regIsReg,
      prior_weight: regPriorWeight,
      class_prompt: regClassPrompt.trim() || null,
    })
      .then(() => {
        setRegOpen(false);
        setRegPath('');
        fetchDatasets();
      })
      .catch(console.error)
      .finally(() => setRegistering(false));
  };

  if (!project) return <div className="text-slate-500">Loading…</div>;

  const tabs: Array<{ key: Tab; label: string; icon: React.ReactNode }> = [
    { key: 'datasets', label: `Datasets (${datasets.length})`, icon: <Database className="w-4 h-4" /> },
    { key: 'config', label: 'Training Config', icon: <SlidersHorizontal className="w-4 h-4" /> },
    { key: 'jobs', label: `Jobs (${jobs.length})`, icon: <Layers className="w-4 h-4" /> },
    { key: 'artifacts', label: `Artifacts (${artifacts.length})`, icon: <Box className="w-4 h-4" /> },
  ];

  return (
    <div className="space-y-6" data-testid="project-detail-page">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold">{project.name}</h2>
          <p className="text-slate-500 mt-1">{project.note || 'No description'}</p>
          <p className="text-xs text-slate-400 mt-0.5">Created {fmtTime(project.created_at)}</p>
        </div>
        <Link
          to={`/projects/${id}/train`}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium"
        >
          New Training Job
        </Link>
      </div>

      <div className="border-b border-slate-200 dark:border-slate-700">
        <nav className="flex space-x-6 text-sm font-medium">
          {tabs.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`flex items-center space-x-2 py-3 border-b-2 ${
                tab === t.key
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
              data-testid={`tab-${t.key}`}
            >
              {t.icon}
              <span>{t.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {tab === 'datasets' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-semibold">Registered Sources</h3>
            <button
              onClick={() => setRegOpen(true)}
              className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              data-testid="register-dataset-btn"
            >
              <Plus className="w-4 h-4" />
              <span>Register Dataset</span>
            </button>
          </div>

          {datasets.length === 0 ? (
            <div className="p-10 text-center text-slate-400 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
              No datasets registered for this project.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {datasets.map((ds) => (
                <Link
                  key={ds.id}
                  to={`/datasets/${ds.id}`}
                  className="block p-4 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 hover:border-blue-500 transition-colors"
                  data-testid={`dataset-card-${ds.id}`}
                >
                  <div className="flex justify-between items-start">
                    <span className="font-mono text-sm break-all">{ds.path}</span>
                    <ExternalLink className="w-4 h-4 text-slate-400 flex-shrink-0" />
                  </div>
                  <div className="flex space-x-3 mt-2 text-xs text-slate-400">
                    <span>×{ds.repeats} repeats</span>
                    <span>{ds.caption_ext}</span>
                    {ds.is_reg && <span className="text-amber-500">regularization</span>}
                    {ds.class_prompt && <span>class: {ds.class_prompt}</span>}
                  </div>
                </Link>
              ))}
            </div>
          )}

          {regOpen && (
            <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4" onClick={() => setRegOpen(false)}>
              <div
                className="bg-white dark:bg-slate-800 rounded-xl max-w-lg w-full p-6 space-y-4 shadow-xl"
                onClick={(e) => e.stopPropagation()}
                data-testid="register-dataset-modal"
              >
                <h3 className="font-semibold text-lg">Register Dataset Source</h3>
                <div className="space-y-3 text-sm">
                  <div>
                    <label className="text-xs text-slate-400">Directory path (server)</label>
                    <PathInput value={regPath} onChange={setRegPath} placeholder="/path/to/images" />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-slate-400">Repeats</label>
                      <input
                        type="number" min={1} value={regRepeats}
                        onChange={(e) => setRegRepeats(Number(e.target.value))}
                        className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-400">Caption extension</label>
                      <input
                        type="text" value={regCaptionExt}
                        onChange={(e) => setRegCaptionExt(e.target.value)}
                        className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600 font-mono"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-400">Prior weight</label>
                      <input
                        type="number" step="0.1" value={regPriorWeight}
                        onChange={(e) => setRegPriorWeight(Number(e.target.value))}
                        className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600"
                      />
                    </div>
                    <div className="flex items-center space-x-2 pt-5">
                      <input
                        type="checkbox" checked={regIsReg}
                        onChange={(e) => setRegIsReg(e.target.checked)}
                        className="rounded text-blue-600"
                      />
                      <span className="text-xs">Regularization set</span>
                    </div>
                  </div>
                  <div>
                    <label className="text-xs text-slate-400">Class prompt (optional)</label>
                    <input
                      type="text" value={regClassPrompt}
                      onChange={(e) => setRegClassPrompt(e.target.value)}
                      className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600"
                    />
                  </div>
                </div>
                <div className="flex justify-end space-x-2 pt-2">
                  <button onClick={() => setRegOpen(false)} className="px-4 py-2 text-sm rounded bg-slate-200 dark:bg-slate-700">Cancel</button>
                  <button
                    onClick={handleRegister}
                    disabled={registering || !regPath.trim()}
                    className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                    data-testid="register-submit-btn"
                  >
                    {registering ? 'Registering…' : 'Register'}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {tab === 'config' && (
        <div className="p-10 text-center bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 space-y-3">
          <p className="text-slate-500">Edit this project's training configuration in the schema-driven editor.</p>
          <Link
            to={`/projects/${id}/train`}
            className="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium"
          >
            Open Training Config
          </Link>
        </div>
      )}

      {tab === 'jobs' && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          {jobs.length === 0 ? (
            <p className="p-10 text-center text-slate-400">No jobs for this project yet.</p>
          ) : (
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 dark:bg-slate-800/50 text-xs text-slate-400 border-b dark:border-slate-700">
                <tr>
                  <th className="p-3">ID</th>
                  <th className="p-3">Name</th>
                  <th className="p-3">Type</th>
                  <th className="p-3">Status</th>
                  <th className="p-3">Progress</th>
                  <th className="p-3">Created</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                {jobs.map((job) => (
                  <tr key={job.id} className="hover:bg-slate-50 dark:hover:bg-slate-750">
                    <td className="p-3 font-mono text-xs">{job.id}</td>
                    <td className="p-3">
                      <Link to={`/jobs/${job.id}`} className="text-blue-500 hover:underline">{job.name}</Link>
                    </td>
                    <td className="p-3 capitalize">{job.type}</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        job.status === 'running'
                          ? 'bg-green-100 text-green-700 dark:bg-green-950/40 dark:text-green-400'
                          : job.status === 'failed'
                          ? 'bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-400'
                          : 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300'
                      }`}>
                        {job.status}
                      </span>
                    </td>
                    <td className="p-3 text-xs">
                      {job.progress?.step != null && job.progress?.total_steps != null
                        ? `${job.progress.step} / ${job.progress.total_steps}`
                        : '--'}
                    </td>
                    <td className="p-3 text-xs text-slate-400">{fmtTime(job.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {tab === 'artifacts' && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          {artifacts.length === 0 ? (
            <p className="p-10 text-center text-slate-400">No artifacts for this project yet.</p>
          ) : (
            <ul className="divide-y divide-slate-200 dark:divide-slate-700 text-sm">
              {artifacts.map((a) => (
                <li key={a.id} className="p-4 flex justify-between items-center">
                  <div>
                    <div className="font-mono text-xs font-semibold">{a.name}</div>
                    <div className="text-xs text-slate-400 mt-0.5">{a.algo} rank {a.rank} · {fmtTime(a.created_at)}</div>
                  </div>
                  <a href={`/api/artifacts/${a.id}/download`} className="text-xs text-blue-500 hover:underline">Download</a>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
