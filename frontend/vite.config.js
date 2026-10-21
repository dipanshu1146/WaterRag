import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // BACKEND_URL is the public API origin, not a secret. Expose only this
  // explicitly named prefix to the browser bundle for the deployed frontend.
  envPrefix: ['VITE_', 'BACKEND_'],
})
