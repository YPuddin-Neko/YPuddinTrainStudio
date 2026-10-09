import { Routes, Route, Navigate, useLocation, useNavigate, type Location } from 'react-router-dom';
import React from 'react';
import Layout from './components/Layout';
import SettingsDrawer from './components/SettingsDrawer';
import OnboardingGate from './pages/Setup/OnboardingGate';
import './pages/Setup/setup.css';
import SettingsRedirect from './pages/Settings/SettingsRedirect';
import { projectUrl } from './utils/projectVersions';
import { ResourceActivityContext, useResourceCacheEvents } from './api/resourcePolicy';

const Setup = React.lazy(() => import('./pages/Setup/Setup'));
const Dashboard = React.lazy(() => import('./pages/Dashboard/Dashboard'));
const Projects = React.lazy(() => import('./pages/Projects/Projects'));
const Presets = React.lazy(() => import('./pages/Presets/Presets'));
const ProjectRoute = React.lazy(() => import('./pages/ProjectDetail/ProjectRoute'));
const Dataset = React.lazy(() => import('./pages/Dataset/Dataset'));
const Sampling = React.lazy(() => import('./pages/Sampling/Sampling'));
const Queue = React.lazy(() => import('./pages/Queue/Queue'));
const TtsWorkspace = React.lazy(() => import('./pages/Tts/TtsWorkspace'));
const JobDetail = React.lazy(() => import('./pages/JobDetail/JobDetail'));
const Settings = React.lazy(() => import('./pages/Settings/Settings'));
const EnvironmentSettings = React.lazy(() => import('./pages/Settings/EnvironmentSettings'));
const Preferences = React.lazy(() => import('./pages/Settings/Preferences'));
const PageSettings = React.lazy(() => import('./pages/Settings/PageSettings'));
const TrainerUpdates = React.lazy(() => import('./pages/Settings/TrainerUpdates'));

function LegacyOutputsRedirect() {
  const location = useLocation(); const params = new URLSearchParams(location.search);
  const project = params.get('project_id') || params.get('project');
  const job = params.get('job_id') || params.get('job');
  return <Navigate replace to={job ? `/jobs/${encodeURIComponent(job)}` : project ? projectUrl(project,params.get('version_id'),'results') : '/projects'}/>;
}
function EnvironmentRoute() {
  const location = useLocation();
  return new URLSearchParams(location.search).get('tab') === 'artifacts' ? <LegacyOutputsRedirect/> : <EnvironmentSettings/>;
}
function settingsRoutes() {
  return <Route path="settings" element={<Settings/>}><Route index element={<SettingsRedirect/>}/><Route path="environment" element={<EnvironmentRoute/>}/><Route path="preferences" element={<Preferences/>}/><Route path="page" element={<PageSettings/>}/><Route path="charts" element={<PageSettings focus="charts"/>}/><Route path="updates" element={<TrainerUpdates/>}/></Route>;
}
export default function AppRoutes() {
  useResourceCacheEvents();
  const location = useLocation(); const navigate = useNavigate();
  const previous = (location.state as {backgroundLocation?: Location} | null)?.backgroundLocation;
  const background = location.pathname.startsWith('/settings') && previous?.pathname && !previous.pathname.startsWith('/settings') ? previous : null;
  const closeSettings = () => { if(background)navigate(`${background.pathname}${background.search}${background.hash}`,{replace:true,state:background.state}); };
  return <>
    <ResourceActivityContext.Provider value={!background}><div className="route-surface" aria-hidden={background ? true : undefined} {...(background ? {inert:''} : {})}><Routes location={background || location}>
      <Route path="/setup" element={<React.Suspense fallback={<div className="setup-loading"/>}><Setup/></React.Suspense>}/>
      <Route path="/" element={<OnboardingGate><Layout navigationKey={location.key}/></OnboardingGate>}>
        <Route index element={<Dashboard/>}/><Route path="projects" element={<Projects/>}/>
        <Route path="presets" element={<Presets/>}/><Route path="tts" element={<TtsWorkspace/>}/>
        <Route path="projects/:id" element={<ProjectRoute/>}/><Route path="projects/:id/train" element={<ProjectRoute view="train"/>}/>
        <Route path="projects/:id/v/:versionId" element={<ProjectRoute/>}/><Route path="projects/:id/v/:versionId/train" element={<ProjectRoute view="train"/>}/>
        <Route path="projects/:id/v/:versionId/curate" element={<ProjectRoute view="curate"/>}/><Route path="datasets/:id" element={<Dataset/>}/><Route path="queue" element={<Queue/>}/><Route path="sampling" element={<Sampling/>}/><Route path="jobs/:id" element={<JobDetail/>}/>
        <Route path="artifacts" element={<LegacyOutputsRedirect/>}/><Route path="models" element={<SettingsRedirect tab="models"/>}/>
        {settingsRoutes()}
      </Route>
    </Routes></div></ResourceActivityContext.Provider>
    {background && <SettingsDrawer onClose={closeSettings}><Routes>{settingsRoutes()}</Routes></SettingsDrawer>}
  </>;
}
