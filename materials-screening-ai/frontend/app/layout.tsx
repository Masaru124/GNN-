import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Fraunces } from "next/font/google";
import "./globals.css";
import { Providers } from "@/components/providers";
import { AuthGate } from "@/components/auth-gate";
import { Navbar } from "@/components/layout/Navbar";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "MatScreen AI",
    template: "%s · MatScreen AI",
  },
  description:
    "AI-powered materials screening assistant with calibrated GNN confidence and uncertainty quantification.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${plexSans.variable} ${plexMono.variable} ${fraunces.variable}`}
    >
      <body className="min-h-screen bg-background font-sans antialiased">
        <Providers>
          <AuthGate>
            <div className="flex min-h-screen flex-col">
              <Navbar />
              <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 sm:px-6 lg:px-8 lg:py-10">
                {children}
              </main>
              <footer className="border-t border-border">
                <div className="mx-auto flex w-full max-w-[1400px] items-center justify-between px-4 py-4 text-[11px] text-muted-foreground sm:px-6 lg:px-8">
                  <span className="font-mono uppercase tracking-[0.18em]">
                    MatScreen AI · v2.0
                  </span>
                  <span className="font-mono uppercase tracking-[0.18em]">
                    Multi-Scale GNN + Conformal 90%
                  </span>
                </div>
              </footer>
            </div>
          </AuthGate>
        </Providers>
      </body>
    </html>
  );
}
