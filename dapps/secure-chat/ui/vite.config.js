import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy IPFS API to avoid browser CORS issues during development
      '/ipfs-api': {
        target: 'http://localhost:5001',
        changeOrigin: true,
        secure: false,
        // Strip the prefix so `/ipfs-api/api/v0/...` becomes `/api/v0/...` on the IPFS API
        rewrite: (path) => path.replace(/^\/ipfs-api/, ''),
        headers: {
          // Ensure the proxied request presents an Origin acceptable to IPFS API
          Origin: 'http://localhost:5174'
        }
      }
    }
  }
});
