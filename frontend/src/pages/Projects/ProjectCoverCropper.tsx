import React from 'react';
import { Move, RotateCcw } from 'lucide-react';
import { useWorkspaceText } from '../../utils/workspaceText';
import { cropForView, cropImageStyle, INITIAL_CROP_VIEW, viewForCrop, type CoverCrop, type CropView } from './coverCrop';
import './cover-crop.css';

export default function ProjectCoverCropper({ file, initialCrop, onCancel, onApply }: {
  file: File; initialCrop?: CoverCrop; onCancel: () => void; onApply: (crop: CoverCrop) => void;
}) {
  const text = useWorkspaceText();
  const [source, setSource] = React.useState('');
  const [size, setSize] = React.useState<{ width: number; height: number } | null>(null);
  const [view, setView] = React.useState<CropView>(INITIAL_CROP_VIEW);
  const [failed, setFailed] = React.useState(false);
  const frame = React.useRef<HTMLDivElement>(null);
  const drag = React.useRef<{ id: number; x: number; y: number; crop: CoverCrop; width: number; height: number } | null>(null);
  React.useEffect(() => {
    const url = URL.createObjectURL(file); setSource(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const crop = size ? cropForView(size.width, size.height, view) : null;
  const baseCrop = size ? cropForView(size.width, size.height, INITIAL_CROP_VIEW) : null;
  const maxZoom = size && baseCrop ? Math.max(1, Math.min(4, baseCrop.width * size.width, baseCrop.height * size.height)) : 4;
  const tooSmall = !!(size && crop && (crop.width * size.width < 1 || crop.height * size.height < 1));
  const changeView = (next: CropView) => {
    if (!size) return;
    next = { ...next, zoom: Math.max(1, Math.min(maxZoom, next.zoom)) };
    const bounded = cropForView(size.width, size.height, next);
    setView({ zoom: next.zoom, cx: bounded.x + bounded.width / 2, cy: bounded.y + bounded.height / 2 });
  };
  return <div className="cover-crop-editor">
    <p id="cover-crop-help">{text('拖动或缩放图片裁切封面。', 'Drag or zoom to crop the cover.')}</p>
    <div ref={frame} className="cover-crop-frame" tabIndex={0} role="group" aria-label={text('封面裁切区域', 'Cover crop area')} aria-describedby="cover-crop-help cover-crop-keyboard"
      onPointerDown={event => {
        if (!crop || event.button !== 0 || drag.current) return;
        event.preventDefault(); event.currentTarget.focus();
        const bounds = event.currentTarget.getBoundingClientRect();
        drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY, crop, width: bounds.width, height: bounds.height };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={event => {
        const start = drag.current; if (!start || event.pointerId !== start.id) return;
        changeView({ zoom: view.zoom, cx: start.crop.x + start.crop.width / 2 - (event.clientX - start.x) / start.width * start.crop.width,
          cy: start.crop.y + start.crop.height / 2 - (event.clientY - start.y) / start.height * start.crop.height });
      }}
      onPointerUp={event => { if (drag.current?.id === event.pointerId) { drag.current = null; event.currentTarget.releasePointerCapture(event.pointerId); } }}
      onPointerCancel={() => { drag.current = null; }} onLostPointerCapture={() => { drag.current = null; }}
      onKeyDown={event => {
        if (!crop || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
        event.preventDefault(); const step = event.shiftKey ? .08 : .02;
        changeView({ ...view, cx: view.cx + (event.key === 'ArrowLeft' ? step : event.key === 'ArrowRight' ? -step : 0) * crop.width,
          cy: view.cy + (event.key === 'ArrowUp' ? step : event.key === 'ArrowDown' ? -step : 0) * crop.height });
      }}>
      {source && <img src={source} alt={text('待裁切的项目封面', 'Project cover to crop')} draggable={false} style={crop ? cropImageStyle(crop) : { visibility: 'hidden' }}
        onLoad={event => {
          const image = event.currentTarget;
          if (!image.naturalWidth || !image.naturalHeight) { setFailed(true); return; }
          setSize({ width: image.naturalWidth, height: image.naturalHeight });
          setView(initialCrop ? viewForCrop(image.naturalWidth, image.naturalHeight, initialCrop) : INITIAL_CROP_VIEW);
          frame.current?.focus();
        }} onError={() => { setFailed(true); setSize(null); }}/>}
      {crop && <div className="cover-crop-grid" aria-hidden="true"/>}
      {!crop && <span role={failed ? 'alert' : 'status'}>{failed ? text('无法读取这张图片，请换一张 JPEG、PNG 或 WebP 图片。', 'Could not read this image. Choose another JPEG, PNG or WebP image.') : text('读取图片…', 'Loading image…')}</span>}
    </div>
    <div className="cover-crop-toolbar"><label><span>{text('缩放', 'Zoom')}</span><input type="range" min={1} max={maxZoom} step={.01} value={view.zoom} disabled={!crop || maxZoom === 1} onChange={event => changeView({ ...view, zoom: Number(event.target.value) })}/><output>{view.zoom.toFixed(2)}×</output></label>
      <button type="button" className="projects-page-button" disabled={!crop} onClick={() => setView(INITIAL_CROP_VIEW)}><RotateCcw size={14}/>{text('重置', 'Reset')}</button></div>
    <small id="cover-crop-keyboard"><Move size={13} aria-hidden="true"/>{text('也可用方向键移动图片 · 封面比例 16:10', 'Arrow keys also move the image · Cover ratio 16:10')}</small>
    {tooSmall && <p role="alert" className="project-editor-error">{text('图片尺寸太小，请选择更大的图片。', 'This image is too small. Choose a larger image.')}</p>}
    <div className="project-editor-footer"><button type="button" className="projects-page-button" onClick={onCancel}>{text('取消裁切', 'Cancel crop')}</button><button type="button" className="projects-create-button" disabled={!crop || failed || tooSmall} onClick={() => crop && !tooSmall && onApply(crop)}>{text('使用此裁切', 'Use this crop')}</button></div>
  </div>;
}
