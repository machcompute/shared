export const allowedDomains: string[] = (process.env.NEXT_PUBLIC_ALLOWED_DOMAINS ?? "")
  .split(",")
  .map((domain) => domain.trim())
  .filter(Boolean);

if (!allowedDomains.length) {
  throw new Error(
    "NEXT_PUBLIC_ALLOWED_DOMAINS is not set. Define the comma-separated list of domains allowed to embed the engine (see .env.example)."
  );
}

const isLocalHostname = (hostname: string) =>
  hostname === "localhost" || hostname === "127.0.0.1";

function hostnameMatches(hostname: string, pattern: string): boolean {
  if (pattern.startsWith("*.")) return hostname.endsWith(pattern.slice(1));
  return hostname === pattern;
}

export function originAllowed(origin: string): boolean {
  let url: URL;
  try {
    url = new URL(origin);
  } catch {
    return false;
  }
  if (!allowedDomains.some((pattern) => hostnameMatches(url.hostname, pattern))) {
    return false;
  }
  if (isLocalHostname(url.hostname)) {
    return url.protocol === "http:" || url.protocol === "https:";
  }
  return url.protocol === "https:";
}

export function frameAncestors(): string {
  return allowedDomains
    .map((pattern) =>
      isLocalHostname(pattern)
        ? `http://${pattern}:* https://${pattern}:*`
        : `https://${pattern}`
    )
    .join(" ");
}
