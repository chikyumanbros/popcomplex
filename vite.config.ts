import { defineConfig } from 'vite';

export default defineConfig({
  assetsInclude: ['**/*.wgsl'],
  build: {
    // Smaller deploy; no public .map URLs. For prod debugging use `vite build --sourcemap` or
    // switch to `true` / `'hidden'` (emits maps without //# sourceMappingURL in bundles).
    sourcemap: false,
    target: 'es2022',
  },
  server: {
    watch: {
      usePolling: true,
      useFsEvents: false,
    },
  },
});
