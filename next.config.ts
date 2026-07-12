import type { NextConfig } from "next";
import { frameAncestors } from "./lib/allowed-origins";

const FRAME_ANCESTORS = frameAncestors();

const nextConfig: NextConfig = {
  // Lets a second local dev server use its own compiler cache. This avoids a
  // port-3002 verification server contending with a developer's normal .next
  // directory on port 3001.
  distDir: process.env.NEXT_DIST_DIR || ".next",
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "Content-Security-Policy",
            value: `frame-ancestors ${FRAME_ANCESTORS}`,
          },
        ],
      },
      {
        source: "/client.js",
        headers: [{ key: "Access-Control-Allow-Origin", value: "*" }],
      },
    ];
  },
};

export default nextConfig;
