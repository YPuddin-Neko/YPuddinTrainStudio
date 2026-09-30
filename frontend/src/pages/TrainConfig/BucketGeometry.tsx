import {bucketGeometry, type BucketRoute, type BucketSize} from './bucketSizing';

export default function BucketGeometry({size,route,fit,anchor,count}:{size:BucketSize;route?:BucketRoute;fit:'pad'|'crop';anchor?:string;count?:number}) {
  const geometry=bucketGeometry(size,route || {width:size.w,height:size.h,resized_width:size.w,resized_height:size.h},fit,anchor);
  const sourceFirst=fit==='crop';
  const outer=sourceFirst?geometry.source:geometry.canvas;
  const right=outer.x+outer.width, bottom=outer.y+outer.height;
  const insetX=Math.min(2.5,outer.width/2), insetY=Math.min(2.5,outer.height/2);
  const widthGap=Math.abs((route?.resized_width ?? size.w)-size.w);
  const heightGap=Math.abs((route?.resized_height ?? size.h)-size.h);
  // Keep subpixel padding and crop strips visible without enlarging their area.
  const edges=[
    geometry.left>0 ? `M${outer.x},${outer.y+insetY}V${bottom-insetY}` : '',
    geometry.top>0 ? `M${outer.x+insetX},${outer.y}H${right-insetX}` : '',
    widthGap>geometry.left ? `M${right},${outer.y+insetY}V${bottom-insetY}` : '',
    heightGap>geometry.top ? `M${outer.x+insetX},${bottom}H${right-insetX}` : '',
  ].filter(Boolean).join(' ');
  return <span className="bucket-geometry" aria-hidden="true" data-fit={fit} data-offset={`${geometry.left},${geometry.top}`}>
    <svg viewBox="0 0 80 56" focusable="false">
      {sourceFirst ? <><rect rx={2.5} {...geometry.source} className={geometry.changed?'bucket-geometry-crop':'bucket-geometry-image'}/><rect rx={2.5} {...geometry.canvas} className="bucket-geometry-image"/></>
        : <><rect rx={2.5} {...geometry.canvas} className="bucket-geometry-pad"/><rect rx={2.5} {...geometry.source} className="bucket-geometry-image"/></>}
      <rect rx={2.5} {...geometry.canvas} className="bucket-geometry-boundary"/>
      {edges && <path d={edges} className="bucket-geometry-edge"/>}
    </svg>
    {count!=null && <span className="bucket-geometry-count">{count}</span>}
  </span>;
}
