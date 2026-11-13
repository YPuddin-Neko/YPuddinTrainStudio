import React from 'react';

function containsPopoverTarget(details: HTMLDetailsElement, target: Node | null) {
  if (!target) return false;
  if (details.contains(target)) return true;
  // A select in the disclosure may render its choices outside it.
  return Array.from(details.querySelectorAll('[aria-controls]')).some(trigger => {
    const controlled = details.ownerDocument.getElementById(trigger.getAttribute('aria-controls') || '');
    return (controlled?.closest('.studio-select-menu') ?? controlled)?.contains(target);
  });
}

export default function PopoverDetails(props: React.ComponentPropsWithoutRef<'details'>) {
  const ref = React.useRef<HTMLDetailsElement>(null);

  React.useEffect(() => {
    const details = ref.current;
    if (!details) return;
    const outside = (event: PointerEvent) => {
      if (details.open && !containsPopoverTarget(details, event.target as Node | null)) details.open = false;
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented || !details.open) return;
      if (event.target instanceof Element && event.target.closest('[role="combobox"][aria-expanded="true"]')) return;
      event.preventDefault();
      event.stopPropagation();
      details.open = false;
      details.querySelector<HTMLElement>(':scope > summary')?.focus({ preventScroll: true });
    };
    const document = details.ownerDocument;
    document.addEventListener('pointerdown', outside, true);
    // Handle Escape before the enclosing dialog can close itself.
    details.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', outside, true);
      details.removeEventListener('keydown', escape);
    };
  }, []);

  return <details {...props} ref={ref} data-popover/>;
}
