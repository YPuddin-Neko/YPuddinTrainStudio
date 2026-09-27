import React from 'react';
import { createPortal } from 'react-dom';
import { TopbarContext } from './topbarContext';

const NARROW = '(max-width: 767px)';

function useNarrowScreen() {
  const [narrow, setNarrow] = React.useState(() => typeof window !== 'undefined' && !!window.matchMedia?.(NARROW).matches);
  React.useEffect(() => {
    const media = window.matchMedia?.(NARROW);
    if (!media) return;
    const update = () => setNarrow(media.matches);
    update();
    media.addEventListener?.('change', update);
    return () => media.removeEventListener?.('change', update);
  }, []);
  return narrow;
}

/**
 * Shows the page's location in the top bar. Phones have no room there, so it stays in the page, unless
 * `topbarOnly`: a page whose title already says where it is then shows nothing there (nor in the settings drawer).
 */
export default function TopbarBreadcrumb({ children, topbarOnly = false }: { children: React.ReactElement; topbarOnly?: boolean }) {
  const slot = React.useContext(TopbarContext);
  const narrow = useNarrowScreen();
  if (slot && !narrow) return createPortal(children, slot);
  return topbarOnly ? null : children;
}
