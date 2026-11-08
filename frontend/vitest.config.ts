import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    coverage: {
      include: ["components/**/*.tsx", "app/api/**/*.ts"],
      reporter: ["text"],
    },
  },
});
