import createNextIntlPlugin from "next-intl/plugin";
import type { NextConfig } from "next";
import path from "path";

const withNextIntl = createNextIntlPlugin(
  "./src/i18n/request.ts"
);

const nextConfig: NextConfig = {
  output: "standalone",
  transpilePackages: ["@ucotron/ui"],
  webpack: (config) => {
    // Resolve dependencies from dashboard's node_modules for symlinked packages
    config.resolve.modules = [
      path.resolve(__dirname, "node_modules"),
      ...(config.resolve.modules || []),
    ];
    return config;
  },
};

export default withNextIntl(nextConfig);
