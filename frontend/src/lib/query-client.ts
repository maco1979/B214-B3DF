import { QueryClient } from '@tanstack/react-query'
import { persistQueryClient } from '@tanstack/react-query-persist-client'
import { createSyncStoragePersister } from '@tanstack/query-sync-storage-persister'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 1000 * 30,
    },
  },
})

if (typeof window !== 'undefined') {
  const persister = createSyncStoragePersister({ storage: window.localStorage })

  persistQueryClient({
    queryClient,
    persister,
    maxAge: 1000 * 60 * 60 * 12, // 12h 以内视为新鲜缓存
    buster: 'v1',
  })
}

