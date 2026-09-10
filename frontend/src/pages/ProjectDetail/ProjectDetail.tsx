import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { apiClient } from '../../api/client';
import { Artifact, DatasetSource, Job, JobListResponse, Project } from '../../api/types';
import { useEventStream } from '../../events/useEventStream';
import { EVENT_TYPES } from '../../events/eventTypes';
import { PathInput } from '../../components/PathBrowser';
import { formatBytes, formatTime } from '../../utils/format';
import { Database, Layers, Box, SlidersHorizontal, Plus, ExternalLink } from 'lucide-react';

type Tab = 'datasets' | 'config' | 'jobs' | 'artifacts';

// 与 Queue 页保持一致的状态徽标配色：running 绿 / paused 琥珀 / failed 红 / 其它 slate
const statusBadgeClass = (status: string) =>
  status === 'running'
    ? 'bg-green-100 text-green-700 dark:bg-green-950/40 dark:text-green-400'
    : status === 'paused'
    ? 'bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400'
    : status === 'failed'
    ? 'bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-400'
    : 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300';

export default function ProjectDetail() {
  const { t } = useTranslation();
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

  if (!project) return <div className="text-slate-500">{t('common.loading')}</div>;

  const tabs: Array<{ key: Tab; label: string; icon: React.ReactNode }> = [
    { key: 'datasets', label: `${t('projectDetail.tabDatasets')} (${datasets.length})`, icon: <Database className="w-4 h-4" /> },
    { key: 'config', label: t('projectDetail.tabConfig'), icon: <SlidersHorizontal className="w-4 h-4" /> },
    { key: 'jobs', label: `${t('projectDetail.tabJobs')} (${jobs.length})`, icon: <Layers className="w-4 h-4" /> },
    { key: 'artifacts', label: `${t('projectDetail.tabArtifacts')} (${artifacts.length})`, icon: <Box className="w-4 h-4" /> },
  ];

  const headerStats = [
    { key: 'datasets', icon: <Database className="w-3.5 h-3.5 text-blue-500" />, label: t('projectDetail.tabDatasets'), count: datasets.length },
    { key: 'jobs', icon: <Layers className="w-3.5 h-3.5 text-indigo-500" />, label: t('projectDetail.tabJobs'), count: jobs.length },
    { key: 'artifacts', icon: <Box className="w-3.5 h-3.5 text-emerald-500" />, label: t('projectDetail.tabArtifacts'), count: artifacts.length },
  ];

  return (
    <div className="space-y-6" data-testid="project-detail-page">
      <div className="flex justify-between items-end">
        <div className="space-y-2">
          <div>
            <h2 className="text-2xl font-bold">{project.name}</h2>
            <p className="text-slate-500 mt-1">{project.note || t('projects.noNote')}</p>
            <p className="text-xs text-slate-400 mt-0.5">
              {t('projects.createdAt')} {formatTime(project.created_at)}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {headerStats.map((s) => (
              <span
                key={s.key}
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs text-slate-500 dark:text-slate-400"
                data-testid={`stat-${s.key}`}
              >
                {s.icon}
                <span>{s.label}</span>
                <span className="font-mono font-semibold text-slate-700 dark:text-slate-200">{s.count}</span>
              </span>
            ))}
          </div>
        </div>
        <Link
          to={`/projects/${id}/train`}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium"
        >
          {t('projectDetail.newTrainingJob')}
        </Link>
      </div>

      <div className="border-b border-slate-200 dark:border-slate-700">
        <nav className="flex space-x-6 text-sm font-medium">
          {tabs.map((tabItem) => (
            <button
              key={tabItem.key}
              onClick={() => setTab(tabItem.key)}
              className={`flex items-center space-x-2 py-3 border-b-2 ${
                tab === tabItem.key
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
              data-testid={`tab-${tabItem.key}`}
            >
              {tabItem.icon}
              <span>{tabItem.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {tab === 'datasets' && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="font-semibold">{t('projectDetail.registeredSources')}</h3>
            <button
              onClick={() => setRegOpen(true)}
              className="flex items-center space-x-1.5 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              data-testid="register-dataset-btn"
            >
              <Plus className="w-4 h-4" />
              <span>{t('projectDetail.registerDataset')}</span>
            </button>
          </div>

          {datasets.length === 0 ? (
            <div
              className="p-10 text-center bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 space-y-2"
              data-testid="datasets-empty"
            >
              <Database className="w-8 h-8 mx-auto text-slate-300 dark:text-slate-600" />
              <p className="text-slate-500 dark:text-slate-400 font-medium">{t('projectDetail.noDatasets')}</p>
              <p className="text-xs text-slate-400">
                {t('projectDetail.noDatasetsHint', '点击「注册数据集」添加训练数据目录。')}
              </p>
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
                  <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-xs text-slate-400">
                    <span className="font-mono">×{ds.repeats} {t('projectDetail.repeats')}</span>
                    <span className="font-mono">{ds.caption_ext}</span>
                    {ds.is_reg && <span className="text-amber-500">{t('projectDetail.regularization')}</span>}
                    {ds.class_prompt && (
                      <span>{t('projectDetail.classPromptShort', '类提示')}: {ds.class_prompt}</span>
                    )}
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
                <h3 className="font-semibold text-lg">{t('projectDetail.registerModalTitle', '注册数据集源')}</h3>
                <div className="space-y-3 text-sm">
                  <div>
                    <label className="text-xs text-slate-400">{t('projectDetail.dirPath')}</label>
                    <PathInput value={regPath} onChange={setRegPath} placeholder="/path/to/images" />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-slate-400">{t('projectDetail.repeats')}</label>
                      <input
                        type="number" min={1} value={regRepeats}
                        onChange={(e) => setRegRepeats(Number(e.target.value))}
                        className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-400">{t('projectDetail.captionExt')}</label>
                      <input
                        type="text" value={regCaptionExt}
                        onChange={(e) => setRegCaptionExt(e.target.value)}
                        className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600 font-mono"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-slate-400">{t('projectDetail.priorWeight')}</label>
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
                      <span className="text-xs">{t('projectDetail.isReg')}</span>
                    </div>
                  </div>
                  <div>
                    <label className="text-xs text-slate-400">{t('projectDetail.classPrompt')}</label>
                    <input
                      type="text" value={regClassPrompt}
                      onChange={(e) => setRegClassPrompt(e.target.value)}
                      className="w-full px-2 py-1.5 border rounded dark:bg-slate-900 dark:border-slate-600"
                    />
                  </div>
                </div>
                <div className="flex justify-end space-x-2 pt-2">
                  <button onClick={() => setRegOpen(false)} className="px-4 py-2 text-sm rounded bg-slate-200 dark:bg-slate-700">{t('common.cancel')}</button>
                  <button
                    onClick={handleRegister}
                    disabled={registering || !regPath.trim()}
                    className="px-4 py-2 text-sm rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
                    data-testid="register-submit-btn"
                  >
                    {registering ? t('projectDetail.registering') : t('projectDetail.register')}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {tab === 'config' && (
        <div className="p-10 text-center bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 space-y-3">
          <SlidersHorizontal className="w-8 h-8 mx-auto text-slate-300 dark:text-slate-600" />
          <p className="text-slate-500">{t('projectDetail.configHint')}</p>
          <Link
            to={`/projects/${id}/train`}
            className="inline-block px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium"
          >
            {t('projectDetail.openTrainConfig')}
          </Link>
        </div>
      )}

      {tab === 'jobs' && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          {jobs.length === 0 ? (
            <div className="p-10 text-center space-y-2" data-testid="jobs-empty">
              <Layers className="w-8 h-8 mx-auto text-slate-300 dark:text-slate-600" />
              <p className="text-slate-500 dark:text-slate-400 font-medium">{t('projectDetail.noJobs')}</p>
              <p className="text-xs text-slate-400">
                {t('projectDetail.noJobsHint', '点击右上角「新建训练任务」开始训练。')}
              </p>
            </div>
          ) : (
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-50 dark:bg-slate-800/50 text-xs text-slate-400 border-b dark:border-slate-700">
                <tr>
                  <th className="p-3">{t('queue.id')}</th>
                  <th className="p-3">{t('queue.name')}</th>
                  <th className="p-3">{t('queue.type')}</th>
                  <th className="p-3">{t('common.status')}</th>
                  <th className="p-3">{t('queue.progress')}</th>
                  <th className="p-3">{t('queue.created')}</th>
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
                      <span className={`px-2 py-0.5 rounded text-xs ${statusBadgeClass(job.status)}`}>
                        {job.status}
                      </span>
                    </td>
                    <td className="p-3 text-xs">
                      {job.progress?.step != null && job.progress?.total_steps != null && job.progress.total_steps > 0 ? (
                        <div className="space-y-1">
                          <div className="font-mono text-slate-500">
                            {job.progress.step} / {job.progress.total_steps}
                          </div>
                          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
                            <div
                              className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                              style={{ width: `${Math.min(100, (job.progress.step / job.progress.total_steps) * 100)}%` }}
                            />
                          </div>
                        </div>
                      ) : '--'}
                    </td>
                    <td className="p-3 text-xs text-slate-400">{formatTime(job.created_at)}</td>
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
            <div className="p-10 text-center space-y-2" data-testid="artifacts-empty">
              <Box className="w-8 h-8 mx-auto text-slate-300 dark:text-slate-600" />
              <p className="text-slate-500 dark:text-slate-400 font-medium">{t('projectDetail.noArtifacts')}</p>
              <p className="text-xs text-slate-400">
                {t('projectDetail.noArtifactsHint', '训练任务完成后，产物会出现在这里。')}
              </p>
            </div>
          ) : (
            <ul className="divide-y divide-slate-200 dark:divide-slate-700 text-sm">
              {artifacts.map((a) => (
                <li key={a.id} className="p-4 flex justify-between items-center">
                  <div>
                    <div className="font-mono text-xs font-semibold">{a.name}</div>
                    <div className="text-xs text-slate-400 mt-0.5">
                      {a.algo || '--'} {t('projectDetail.rankShort', 'rank')} {a.rank ?? '--'}
                      {' · '}<span className="font-mono">{formatBytes(a.size)}</span>
                      {a.step != null && ` · ${t('job.stepCol')} ${a.step}`}
                      {' · '}{formatTime(a.created_at)}
                    </div>
                  </div>
                  <a href={`/api/artifacts/${a.id}/download`} className="text-xs text-blue-500 hover:underline">{t('common.download')}</a>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
