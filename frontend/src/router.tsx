import { Routes, Route } from 'react-router-dom';
import React, { Suspense } from 'react';
import Layout from './components/Layout';
import SettingsRedirect from './pages/Settings/SettingsRedirect';

// Code-split pages with React.lazy
const Dashboard = React.lazy(() => import('./pages/Dashboard/Dashboard'));
const Projects = React.lazy(() => import('./pages/Projects/Projects'));
const ProjectDetail = React.lazy(() => import('./pages/ProjectDetail/ProjectDetail'));
const Dataset = React.lazy(() => import('./pages/Dataset/Dataset'));
const TrainConfig = React.lazy(() => import('./pages/TrainConfig/TrainConfig'));
const Queue = React.lazy(() => import('./pages/Queue/Queue'));
const JobDetail = React.lazy(() => import('./pages/JobDetail/JobDetail'));
const Settings = React.lazy(() => import('./pages/Settings/Settings'));
const EnvironmentSettings = React.lazy(() => import('./pages/Settings/EnvironmentSettings'));
const Preferences = React.lazy(() => import('./pages/Settings/Preferences'));

export default function AppRoutes() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-64 text-slate-400">Loading page...</div>}>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="projects" element={<Projects />} />
          <Route path="projects/:id" element={<ProjectDetail />} />
          <Route path="projects/:id/train" element={<TrainConfig />} />
          <Route path="datasets/:id" element={<Dataset />} />
          <Route path="queue" element={<Queue />} />
          <Route path="jobs/:id" element={<JobDetail />} />
          <Route path="artifacts" element={<SettingsRedirect tab="artifacts" />} />
          <Route path="models" element={<SettingsRedirect tab="models" />} />
          <Route path="settings" element={<Settings />}>
            <Route index element={<SettingsRedirect />} />
            <Route path="environment" element={<EnvironmentSettings />} />
            <Route path="preferences" element={<Preferences />} />
          </Route>
        </Route>
      </Routes>
    </Suspense>
  );
}
