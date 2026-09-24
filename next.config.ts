import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // opencv-js ships a single ~13MB file with its WASM inlined as base64;
  // keep it out of the bundling/parsing pass and just `require` it natively.
  // ffmpeg-static resolves its binary path relative to `__dirname` at
  // require-time; letting Next.js bundle it rewrites `__dirname` to a
  // virtual path that doesn't exist on disk, so the binary can't be found.
  serverExternalPackages: ["@techstark/opencv-js", "ffmpeg-static"],
  // Next.js keeps a single long-lived, keep-alive HTTP agent for
  // server-side fetch() across the whole process. If a pooled HTTP/2
  // session to a host (api.replicate.com) gets destroyed - e.g. by an idle
  // timeout between the initial predictions POST and a later poll GET -
  // that agent doesn't evict it, so every subsequent fetch to that host
  // fails forever with ERR_HTTP2_INVALID_SESSION until the process
  // restarts. Disabling Keep-Alive makes every fetch open a fresh
  // connection instead of reusing the pool, which avoids that class of
  // failure entirely (confirmed: an app-level retry-once did NOT recover,
  // because the immediate retry was handed the exact same dead session).
  httpAgentOptions: {
    keepAlive: false,
  },
};

export default nextConfig;
