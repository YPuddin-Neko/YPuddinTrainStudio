import type { SVGProps } from 'react';

/**
 * A training mask, drawn like lucide's icons: the frame is the image, the shaded band is left out of
 * training and the circle is what still trains. Stands for automatic masks of every kind.
 */
export default function MaskIcon({ size = 24, strokeWidth = 2, ...props }: SVGProps<SVGSVGElement> & { size?: number | string }) {
  return <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>
    <path d="M6 3h12a3 3 0 0 1 3 3v12a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3V6a3 3 0 0 1 3-3zM12 7.5a4.5 4.5 0 1 0 0 9a4.5 4.5 0 1 0 0-9z" fill="currentColor" fillOpacity={0.22} fillRule="evenodd" stroke="none"/>
    <rect x="3" y="3" width="18" height="18" rx="3"/>
    <circle cx="12" cy="12" r="4.5"/>
  </svg>;
}
