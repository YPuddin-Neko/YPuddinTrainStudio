import { createContext } from 'react';
import type { WorkspaceStep } from '../ProjectWorkflow';
import type { ProjectVersion, VersionedProject } from '../../utils/projectVersions';

export interface ProjectSidebarSelection {
  project: VersionedProject;
  versionId?: string;
  versions: ProjectVersion[];
  current?: ProjectVersion;
  active: WorkspaceStep;
  routeKey: string;
  pathname: string;
}

/** Layout retains project navigation; page actions only live while their owner is mounted. */
export const ProjectSidebarContext = createContext<{
  target: HTMLDivElement | null;
  closeNavigation: () => void;
  register?: (selection: ProjectSidebarSelection, beforeAction?: () => Promise<void>) => () => void;
} | null>(null);
