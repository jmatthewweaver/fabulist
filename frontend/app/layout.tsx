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
      <body className={`${inter.className} bg-stone-950 text-stone-100 min-h-screen`}>
        {children}
      </body>
    </html>
  );
}
