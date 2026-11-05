import type { PaintDocumentState } from './paintDocument';
import type { MaskDocumentState } from './maskDocument';
import type { PaintInfo } from './paintApi';

export interface PaintDraft {
  datasetId: string;
  info: PaintInfo;
  paint: PaintDocumentState;
  mask: MaskDocumentState;
  history: ('paint'|'mask')[];
  historyPosition: number;
}
export type PaintDraftStore = Map<string, PaintDraft>;
export const paintDraftKey = (datasetId: string, relPath: string) => JSON.stringify([datasetId, relPath]);
export const paintDraftDirty = (draft: PaintDraft) => draft.paint.position !== draft.paint.savedPosition || draft.mask.position !== draft.mask.savedPosition;
export interface PaintEditorState { dirty: boolean; canUndo: boolean; canRedo: boolean; canRestore: boolean; busy: boolean; switching?: boolean; available: boolean; message: string }
