import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import type { CaptionFile, DatasetImage, DatasetImagesPage } from '../../api/types';
import type { ImageSort } from './ImageSortSelect';

/** Read caption metadata in pages; image pixels still load only for visible thumbnails and the active canvas. */
export function useDatasetImageCatalog(datasetId: string | undefined, sort: ImageSort = 'filename', includeEditStatus = false, captionFile?: CaptionFile) {
  return useQuery<DatasetImage[]>({
    queryKey: ['dataset-image-catalog', datasetId, sort, includeEditStatus, captionFile], enabled: !!datasetId,
    refetchOnWindowFocus: false, refetchOnReconnect: false,
    placeholderData: captionFile ? (previous, query) => query?.queryKey[1] === datasetId ? previous : undefined : undefined,
    queryFn: async ({ signal }) => {
      const items: DatasetImage[] = [];
      let page = 1;
      while (true) {
        const result = await apiClient.get<DatasetImagesPage>(`/datasets/${datasetId}/images`, { params: { page, page_size: 200, sort, ...(includeEditStatus ? {include_edit_status:true} : {}), ...(captionFile ? {caption_file:captionFile} : {}) }, signal, silent: true });
        items.push(...result.items);
        if (items.length >= result.total) return items;
        if (!result.items.length) throw new Error('Image list changed while loading. Refresh and try again.');
        page++;
      }
    },
  });
}
