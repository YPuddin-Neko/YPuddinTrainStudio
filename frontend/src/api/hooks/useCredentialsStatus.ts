import { apiClient, READ_TIMEOUT_MS } from '../client';
import { useResourceQuery } from '../resourcePolicy';

export const credentialProviders = ['huggingface', 'modelscope', 'danbooru', 'gelbooru', 'e621', 'rule34'] as const;
export type CredentialProvider = typeof credentialProviders[number];
export type CredentialStates = Record<CredentialProvider, { configured: boolean }>;

/** Only public configuration flags belong in the shared cache. */
export async function readCredentialStatus(signal?: AbortSignal): Promise<Partial<CredentialStates>> {
  try {
    const result = await apiClient.get<Partial<CredentialStates>>('/credentials', { silent: true, signal, timeout: READ_TIMEOUT_MS });
    return Object.fromEntries(credentialProviders.filter(provider => typeof result[provider]?.configured === 'boolean')
      .map(provider => [provider, { configured: result[provider]!.configured }]));
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw error;
    // Server failures must not put stored or echoed secrets into the query cache.
    // eslint-disable-next-line preserve-caught-error -- The cause may contain a credential or echoed token.
    throw new Error('Could not load access-key status');
  }
}
export function useCredentialsStatus() {
  return useResourceQuery({ queryKey: ['credentials'], queryFn: ({ signal }) => readCredentialStatus(signal) });
}
