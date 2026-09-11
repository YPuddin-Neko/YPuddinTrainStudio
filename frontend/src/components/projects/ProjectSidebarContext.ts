import { createContext } from 'react';

/** The page owns its project controls; Layout only owns their DOM destination. */
export const ProjectSidebarContext = createContext<{
  target: HTMLDivElement | null;
  closeNavigation: () => void;
} | null>(null);
