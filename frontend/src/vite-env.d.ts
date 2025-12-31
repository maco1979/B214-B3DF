/// <reference types="vite/client" />

declare module '*.svg' {
  const content: string
  export default content
}

declare module '*.png' {
  const content: string
  export default content
}

declare module '*.jpg' {
  const content: string
  export default content
}

declare module '*.jpeg' {
  const content: string
  export default content
}

declare module '*.gif' {
  const content: string
  export default content
}

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  // Add other environment variables here if needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
