import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // In development the Python API runs separately on :8000.
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
