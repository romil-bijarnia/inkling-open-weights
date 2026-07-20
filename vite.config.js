import { defineConfig } from "vite";

export default defineConfig({
  publicDir: "data",
  build: {
    target: "es2022",
    sourcemap: true,
  },
});
