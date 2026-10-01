import React from 'react';
import { ImageOff, Loader2 } from 'lucide-react';

/** A spinner with its label, for panels and sections that are still loading. */
export function LoadingNote({ label, block = false, className = '' }: { label: string; block?: boolean; className?: string }) {
  const Element = block ? 'div' : 'span';
  return <Element role="status" className={`ui-loading${block ? ' ui-loading-block' : ''}${className ? ` ${className}` : ''}`}>
    <Loader2 size={block ? 18 : 14} className="animate-spin" aria-hidden="true"/><span>{label}</span>
  </Element>;
}

/**
 * An image that shimmers while it downloads and fades in once decoded. The caller's box is the frame:
 * it must be positioned, and the placeholder fills it.
 */
type LazyImageProps = React.ImgHTMLAttributes<HTMLImageElement> & { src: string; fallback?: React.ReactNode };

export function LazyImage(props: LazyImageProps) {
  // Each source gets its own load state, so a new picture always starts from the placeholder.
  return <ImageLoadState key={props.src} {...props}/>;
}

function ImageLoadState({ src, alt, className = '', fallback, onLoad, onError, ...props }: LazyImageProps) {
  const [loaded, setLoaded] = React.useState(false);
  const [failed, setFailed] = React.useState(false);
  // A cached image can finish before its load listener runs.
  const measure = React.useCallback((image: HTMLImageElement | null) => { if (image?.complete && image.naturalWidth > 0) setLoaded(true); }, []);
  if (failed) return <span className="ui-image-fallback">{fallback ?? <ImageOff size={22} aria-hidden="true"/>}</span>;
  return <>
    {!loaded && <span className="ui-skeleton ui-image-placeholder" aria-hidden="true"/>}
    <img {...props} ref={measure} src={src} alt={alt} className={`ui-image${loaded ? '' : ' is-pending'}${className ? ` ${className}` : ''}`}
      onLoad={event => { setLoaded(true); onLoad?.(event); }} onError={event => { setFailed(true); onError?.(event); }}/>
  </>;
}
