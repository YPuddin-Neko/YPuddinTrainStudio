import React from 'react';
import { EASE_OUT, prefersReducedMotion } from '../../utils/motion';

type Props = {
  text: string;
  value: number | null | undefined;
  contextKey: string;
  updateKey: number;
  active: boolean;
  delta?: boolean;
  duration?: number;
};
type Reading = Pick<Props, 'text' | 'value' | 'contextKey' | 'updateKey' | 'active'>;
type Change = { id: number; from: string; direction: number; amount: number };

function changeText(value: number) {
  return `${value > 0 ? '+' : '-'}${Math.abs(value).toFixed(4)}`;
}

/** Keep each digit attached to its decimal place when a counter gains a digit. */
function characters(text: string) {
  let number = 0;
  return text.split(/(\d+(?:\.\d+)?)/).flatMap((part, partIndex) => {
    if (!/^\d/.test(part)) return [{ key: `separator-${partIndex}`, character: part, digit: false }];
    const point = part.indexOf('.') < 0 ? part.length : part.indexOf('.');
    const group = number++;
    return [...part].map((character, index) => ({ key: `${group}:${point - index}`, character, digit: character !== '.' }));
  });
}

function Digit({ character, previous, change, duration }: { character: string; previous?: string; change: Change | null; duration: number }) {
  const currentRef = React.useRef<HTMLSpanElement>(null);
  const previousRef = React.useRef<HTMLSpanElement>(null);
  const lastUpdate = React.useRef<number | null>(null);
  React.useLayoutEffect(() => {
    const current = currentRef.current;
    const outgoing = previousRef.current;
    if (!change || !current || !outgoing || previous === character || typeof current.animate !== 'function' || prefersReducedMotion() || document.hidden) return;
    const now = performance.now();
    const interrupted = lastUpdate.current != null && now - lastUpdate.current < duration;
    lastUpdate.current = now;
    // During a burst show the newest reading immediately, never queue intermediate digits.
    if (interrupted) return;
    const distance = change.direction > 0 ? '100%' : '-100%';
    const exit = change.direction > 0 ? '-100%' : '100%';
    const options = { duration, easing: EASE_OUT };
    const arriving = current.animate([{ transform: `translateY(${distance})`, opacity: 0 }, { transform: 'translateY(0)', opacity: 1 }], options);
    const leaving = outgoing.animate([{ transform: 'translateY(0)', opacity: 1 }, { transform: `translateY(${exit})`, opacity: 0 }], options);
    return () => { arriving.cancel(); leaving.cancel(); };
  }, [character, previous, change, duration]);
  return <span className="metric-digit"><span ref={currentRef}>{character}</span><span ref={previousRef} className="metric-digit-previous">{previous ?? character}</span></span>;
}

export default function MetricValue({ text, value, contextKey, updateKey, active, delta = false, duration = 420 }: Props) {
  const [visible, setVisible] = React.useState(() => !document.hidden);
  const enabled = active && visible;
  const [reading, setReading] = React.useState<Reading>({ text, value, contextKey, updateKey, active: enabled });
  const [change, setChange] = React.useState<Change | null>(null);
  const [sequence, setSequence] = React.useState(0);
  if (reading.text !== text || !Object.is(reading.value, value) || reading.contextKey !== contextKey || reading.updateKey !== updateKey || reading.active !== enabled) {
    const sameContext = reading.contextKey === contextKey && reading.active && enabled;
    const valid = typeof value === 'number' && Number.isFinite(value) && typeof reading.value === 'number' && Number.isFinite(reading.value);
    const amount = valid ? value! - reading.value! : 0;
    const fresh = sameContext && reading.updateKey !== updateKey && valid && Number.isFinite(amount) && amount !== 0;
    setReading({ text, value, contextKey, updateKey, active: enabled });
    if (fresh) {
      setSequence(sequence + 1);
      setChange({ id: sequence + 1, from: reading.text, direction: Math.sign(amount), amount });
    } else if (!sameContext || !valid || reading.text !== text || reading.value !== value) setChange(null);
  }
  React.useEffect(() => {
    const updateVisibility = () => setVisible(!document.hidden);
    document.addEventListener('visibilitychange', updateVisibility);
    return () => document.removeEventListener('visibilitychange', updateVisibility);
  }, []);
  React.useEffect(() => {
    if (!change) return;
    const timer = window.setTimeout(() => setChange(null), 1400);
    return () => window.clearTimeout(timer);
  }, [change]);
  const previous = new Map(characters(change?.from ?? text).map(part => [part.key, part.character]));
  const shownChange = delta && change ? changeText(change.amount) : null;
  return <span className="metric-value">
    <span className="metric-value-readable sr-only">{text}</span>
    <span className="metric-value-digits" aria-hidden="true">{characters(text).map(part => part.digit
      ? <Digit key={`${contextKey}:${part.key}`} character={part.character} previous={previous.get(part.key)} change={change} duration={duration}/>
      : <span key={part.key}>{part.character}</span>)}</span>
    {delta && <span className="metric-change-slot">{shownChange && <span key={change!.id} className="job-stat-delta" data-direction={change!.direction < 0 ? 'down' : 'up'} title={shownChange}>{shownChange}</span>}</span>}
  </span>;
}
