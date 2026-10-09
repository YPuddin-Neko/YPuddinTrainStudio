import React from 'react';
import { Loader2, Pause, Play, Volume2, VolumeX } from 'lucide-react';
import { useWorkspaceText } from '../utils/workspaceText';
import './audio-player.css';

interface Props {
  src: string;
  label: string;
  autoPlay?: boolean;
  playRequest?: number;
  loadError: string;
  playError: string;
}

function clock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor(whole / 60) % 60;
  return `${hours ? `${hours}:${String(minutes).padStart(2, '0')}` : minutes}:${String(whole % 60).padStart(2, '0')}`;
}

function AudioControls({ src, label, autoPlay = false, playRequest = 0, loadError, playError }: Props) {
  const text = useWorkspaceText();
  const audio = React.useRef<HTMLAudioElement>(null);
  const active = React.useRef(false);
  const attempt = React.useRef(0);
  const wantsPlayback = React.useRef(false);
  const previousVolume = React.useRef(1);
  const [playing, setPlaying] = React.useState(false);
  const [pending, setPending] = React.useState(false);
  const [waiting, setWaiting] = React.useState(false);
  const [time, setTime] = React.useState(0);
  const [duration, setDuration] = React.useState(0);
  const [volume, setVolume] = React.useState(1);
  const [muted, setMuted] = React.useState(false);
  const [failure, setFailure] = React.useState<'load' | 'play' | null>(null);
  const errorId = React.useId();

  React.useEffect(() => {
    const element = audio.current;
    active.current = true;
    if (element && element.getAttribute('src') !== src) element.setAttribute('src', src);
    return () => {
      active.current = false;
      attempt.current += 1;
      wantsPlayback.current = false;
      element?.pause();
      element?.removeAttribute('src');
      element?.load();
    };
  }, [src]);

  const play = React.useCallback(() => {
    const element = audio.current;
    if (!element) return;
    const currentAttempt = ++attempt.current;
    wantsPlayback.current = true;
    setPending(true);
    setFailure(current => current === 'load' ? current : null);
    if (element.error) element.load();
    if (element.ended) element.currentTime = 0;
    void Promise.resolve(element.play()).then(() => {
      if (!active.current || attempt.current !== currentAttempt) return;
      setPending(false);
      setPlaying(true);
    }).catch(error => {
      if (!active.current || attempt.current !== currentAttempt) return;
      wantsPlayback.current = false;
      setPending(false);
      setPlaying(false);
      setWaiting(false);
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        setFailure(current => current === 'load' ? current : 'play');
      }
    });
  }, []);

  React.useEffect(() => { if (autoPlay) play(); }, [autoPlay, playRequest, play]);

  const stopped = () => {
    if (!active.current) return;
    attempt.current += 1;
    wantsPlayback.current = false;
    setPlaying(false);
    setPending(false);
    setWaiting(false);
  };
  const toggle = () => {
    if (wantsPlayback.current || playing) {
      stopped();
      audio.current?.pause();
    } else play();
  };
  const timing = (element: HTMLAudioElement) => {
    if (!active.current) return;
    setTime(Number.isFinite(element.currentTime) ? element.currentTime : 0);
    setDuration(Number.isFinite(element.duration) && element.duration > 0 ? element.duration : 0);
  };
  const seek = (value: number) => {
    const element = audio.current;
    if (!element || !duration) return;
    const next = Math.min(duration, Math.max(0, value));
    element.currentTime = next;
    setTime(next);
  };
  const seekKey = (event: React.KeyboardEvent<HTMLInputElement>) => {
    const target = { ArrowLeft: time - 5, ArrowRight: time + 5, ArrowDown: time - 5, ArrowUp: time + 5, Home: 0, End: duration, PageDown: time - 10, PageUp: time + 10 }[event.key];
    if (target === undefined) return;
    event.preventDefault();
    seek(target);
  };
  const syncVolume = (element: HTMLAudioElement) => {
    if (!active.current) return;
    setVolume(element.volume);
    setMuted(element.muted);
    if (element.volume > 0) previousVolume.current = element.volume;
  };
  const changeVolume = (value: number) => {
    const element = audio.current;
    if (!element) return;
    element.volume = value;
    element.muted = false;
    syncVolume(element);
  };
  const toggleMute = () => {
    const element = audio.current;
    if (!element) return;
    if (element.muted || element.volume === 0) {
      if (element.volume === 0) element.volume = previousVolume.current;
      element.muted = false;
    } else element.muted = true;
    syncVolume(element);
  };
  const silent = muted || volume === 0;
  const elapsed = clock(time);
  const total = duration ? clock(duration) : '--:--';
  const busy = pending || waiting;
  const playLabel = playing || pending ? text('暂停', 'Pause') : text('播放', 'Play');
  const muteLabel = silent ? text('取消静音', 'Unmute') : text('静音', 'Mute');

  return <section className="studio-audio-player" role="group" aria-label={label} aria-describedby={failure ? errorId : undefined}>
    <audio ref={audio} hidden preload="none" src={src}
      onLoadedMetadata={event => timing(event.currentTarget)} onDurationChange={event => timing(event.currentTarget)} onTimeUpdate={event => timing(event.currentTarget)}
      onVolumeChange={event => syncVolume(event.currentTarget)}
      onPlay={() => { if (active.current) { wantsPlayback.current = true; setPlaying(true); setFailure(null); } }}
      onPlaying={() => { if (active.current) { setPending(false); setWaiting(false); setFailure(null); } }}
      onWaiting={() => { if (active.current && wantsPlayback.current) setWaiting(true); }}
      onPause={stopped} onEnded={stopped}
      onError={() => { if (active.current) { stopped(); setFailure('load'); } }}/>
    <div className="studio-audio-controls">
      <button type="button" className="ui-btn ui-btn-icon studio-audio-play" onClick={toggle} aria-label={playLabel} title={playLabel}>
        {busy ? <Loader2 size={16} className="studio-audio-loading" aria-hidden="true"/> : playing ? <Pause size={16} aria-hidden="true"/> : <Play size={16} aria-hidden="true"/>}
      </button>
      <input className="studio-audio-seek" type="range" min={0} max={duration || 1} step={0.1} value={Math.min(time, duration || 0)} disabled={!duration}
        style={{ '--audio-fill': `${duration ? Math.min(100, time / duration * 100) : 0}%` } as React.CSSProperties}
        aria-label={text('播放进度', 'Playback position')} aria-valuetext={text(`${elapsed}，共 ${total}`, `${elapsed} of ${total}`)}
        onChange={event => seek(Number(event.target.value))} onKeyDown={seekKey}/>
      <span className="studio-audio-time" aria-hidden="true">{elapsed}<span> / </span>{total}</span>
      <div className="studio-audio-volume">
        <button type="button" className="ui-btn ui-btn-sm ui-btn-icon ui-btn-quiet" onClick={toggleMute} aria-label={muteLabel} title={muteLabel}>
          {silent ? <VolumeX size={15} aria-hidden="true"/> : <Volume2 size={15} aria-hidden="true"/>}
        </button>
        <input type="range" min={0} max={1} step={0.05} value={muted ? 0 : volume} onChange={event => changeVolume(Number(event.target.value))}
          style={{ '--audio-fill': `${(muted ? 0 : volume) * 100}%` } as React.CSSProperties}
          aria-label={text('音量', 'Volume')} aria-valuetext={`${Math.round((muted ? 0 : volume) * 100)}%`}/>
      </div>
    </div>
    <span className="sr-only" role="status">{busy ? text('正在加载音频…', 'Loading audio…') : ''}</span>
    {failure && <p className="studio-audio-error" id={errorId} role="alert">{failure === 'load' ? loadError : playError}</p>}
  </section>;
}

export default function AudioPlayer(props: Props) {
  return <AudioControls key={props.src} {...props}/>;
}
