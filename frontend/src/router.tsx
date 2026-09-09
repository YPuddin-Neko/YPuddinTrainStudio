import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard/Dashboard';
import Projects from './pages/Projects/Projects';
import ProjectDetail from './pages/ProjectDetail/ProjectDetail';
import TrainConfig from './pages/TrainConfig/TrainConfig';
import Queue from './pages/Queue/Queue';
import JobDetail from './pages/JobDetail/JobDetail';
import Artifacts from './pages/Artifacts/Artifacts';
import Models from './pages/Models/Models';
import Settings from './pages/Settings/Settings';

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="projects" element={<Projects />} />
        <Route path="projects/:id" element={<ProjectDetail />} />
        <Route path="projects/:id/train" element={<TrainConfig />} />
        <Route path="queue" element={<Queue />} />
        <Route path="jobs/:id" element={<JobDetail />} />
        <Route path="artifacts" element={<Artifacts />} />
        <Route path="models" element={<Models />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
