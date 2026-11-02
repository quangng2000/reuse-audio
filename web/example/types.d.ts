interface ImportMetaEnv {
  readonly VITE_MODEL_URL?: string;
  readonly VITE_MODEL_SHA256?: string;
}
interface ImportMeta { readonly env: ImportMetaEnv }
