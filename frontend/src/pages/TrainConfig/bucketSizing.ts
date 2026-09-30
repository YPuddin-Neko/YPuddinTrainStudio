export type BucketRoute = {width:number;height:number;resized_width:number;resized_height:number};
export type BucketSize = {w:number;h:number};
const anchors:Record<string,[number,number]> = {top_left:[0,0],top:[1,0],top_right:[2,0],left:[0,1],center:[1,1],right:[2,1],bottom_left:[0,2],bottom:[1,2],bottom_right:[2,2]};

/** Use the loader's integer offsets; the SVG scales the complete transform uniformly. */
export function bucketGeometry(size:BucketSize, route:BucketRoute, fit:'pad'|'crop', anchor='center') {
  const rw=route.resized_width, rh=route.resized_height;
  const [ax,ay]=anchors[anchor] || anchors.center;
  const crop=fit==='crop';
  const left=crop ? Math.floor((rw-size.w)*ax/2) : Math.floor((size.w-rw)/2);
  const top=crop ? Math.floor((rh-size.h)*ay/2) : Math.floor((size.h-rh)/2);
  const width=Math.max(size.w,rw), height=Math.max(size.h,rh);
  const scale=56/Math.max(width,height);
  const x=(80-width*scale)/2, y=(56-height*scale)/2;
  return {scale,left,top,
    source:{x:x+(crop?0:left*scale),y:y+(crop?0:top*scale),width:rw*scale,height:rh*scale},
    canvas:{x:x+(crop?left*scale:0),y:y+(crop?top*scale:0),width:size.w*scale,height:size.h*scale},
    changed:rw!==size.w || rh!==size.h,
  };
}
