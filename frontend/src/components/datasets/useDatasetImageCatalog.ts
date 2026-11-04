import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import type { DatasetImage, DatasetImagesPage } from '../../api/types';
import type { ImageSort } from './ImageSortSelect';

/** Read caption metadata in pages; image pixels still load only for visible thumbnails and the active canvas. */
export function useDatasetImageCatalog(datasetId: string | undefined, sort: ImageSort = 'filename', includeEditStatus = false) {
  return useQuery<DatasetImage[]>({
    queryKey: ['dataset-image-catalog', datasetId, sort, includeEditStatus], enabled: !!datasetId,
    refetchOnWindowFocus: false, refetchOnReconnect: false,
    queryFn: async ({ signal }) => {
      const items: DatasetImage[] = [];
      let page = 1;
      while (true) {
        const result = await apiClient.get<DatasetImagesPage>(`/datasets/${datasetId}/images`, { params: { page, page_size: 200, sort, ...(includeEditStatus ? {include_edit_status:true} : {}) }, signal, silent: true });
        items.push(...result.items);
        if (items.length >= result.total) return items;
        if (!result.items.length) throw new Error('Image list changed while loading. Refresh and try again.');
        page++;
      }
    },
  });
}
