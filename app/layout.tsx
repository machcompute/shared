import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "machcomputing shared LLM engine",
  robots: { index: false, follow: false },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
