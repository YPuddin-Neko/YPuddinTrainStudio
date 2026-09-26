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

/** Shows the page's location in the top bar. Phones have no room there, so it stays in the page. */
export default function TopbarBreadcrumb({ children }: { children: React.ReactElement }) {
  const slot = React.useContext(TopbarContext);
  const narrow = useNarrowScreen();
  return slot && !narrow ? createPortal(children, slot) : children;
}
