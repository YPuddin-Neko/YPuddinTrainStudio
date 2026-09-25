import React from 'react';

export const EASE_OUT = 'cubic-bezier(.2,.75,.25,1)';

export function prefersReducedMotion() {
  return typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}

/** Fade a surface in (with a small rise) whenever `key` changes, without remounting it. */
export function useEnterAnimation<T extends HTMLElement>(key: unknown, { distance = 4, duration = 180, skipFirst = false } = {}) {
  const ref = React.useRef<T>(null);
  const first = React.useRef(true);
  React.useLayoutEffect(() => {
    const element = ref.current;
    const initial = first.current;
    first.current = false;
    if ((initial && skipFirst) || !element || typeof element.animate !== 'function' || prefersReducedMotion()) return;
    const animation = element.animate([{ opacity: 0, transform: `translateY(${distance}px)` }, { opacity: 1, transform: 'none' }], { duration, easing: EASE_OUT });
    return () => animation.cancel();
  }, [key, distance, duration, skipFirst]);
  return ref;
}
