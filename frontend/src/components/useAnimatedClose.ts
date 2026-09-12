import React from 'react';

/** Keep a dismissed surface mounted just long enough to finish its exit. */
export function useAnimatedClose(onClose: () => void, disabled = false) {
  const [closing, setClosing] = React.useState(false);
  const latest = React.useRef({onClose,disabled}); latest.current={onClose,disabled};
  const timer = React.useRef<ReturnType<typeof setTimeout>>();
  React.useEffect(() => () => clearTimeout(timer.current), []);
  const requestClose = React.useCallback(() => {
    if (latest.current.disabled || timer.current) return;
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {latest.current.onClose();return;}
    setClosing(true);
    timer.current=setTimeout(() => {timer.current=undefined;latest.current.onClose();setClosing(false);},180);
  },[]);
  return {closing,requestClose};
}
