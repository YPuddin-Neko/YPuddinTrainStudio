import React from 'react';
import { useBlocker, type BlockerFunction, type Location } from 'react-router-dom';

interface Props {
  shouldBlock: (destination: Location) => boolean;
  beforeLeave: () => Promise<void>;
  onError: (error: unknown) => void;
}

/** The data router owns POP/PUSH/REPLACE history restoration while drafts save. */
export default function DatasetNavigationGuard(props: Props) {
  const callbacks = React.useRef(props);
  React.useLayoutEffect(() => { callbacks.current = props; });
  const blocker = useBlocker(React.useCallback<BlockerFunction>(({ nextLocation }) => callbacks.current.shouldBlock(nextLocation), []));
  const saving = React.useRef(false);
  React.useEffect(() => {
    if (blocker.state !== 'blocked' || saving.current) return;
    saving.current = true;
    void callbacks.current.beforeLeave().then(() => blocker.proceed()).catch(error => {
      callbacks.current.onError(error);
      blocker.reset();
    }).finally(() => { saving.current = false; });
  }, [blocker]);
  return null;
}
