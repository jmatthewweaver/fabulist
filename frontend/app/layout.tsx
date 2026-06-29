import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Fabulist",
  description: "AI-augmented interactive fiction",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      {/* min-h-[100dvh] (not min-h-screen / 100vh): on mobile 100vh sits behind the browser
          toolbar and forces a spurious vertical scrollbar; dvh tracks the visible viewport. */}
      <body className={`${inter.className} bg-stone-950 text-stone-100 min-h-[100dvh]`}>
        {children}
      </body>
    </html>
  );
}
