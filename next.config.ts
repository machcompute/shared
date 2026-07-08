import type { NextConfig } from "next";

const FRAME_ANCESTORS = [
  "https://machcomputing.com",
  "https://*.machcomputing.com",
  "http://localhost:*",
].join(" ");

const nextConfig: NextConfig = {
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
    ];
  },
};

export default nextConfig;
