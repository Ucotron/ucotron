import type { Metadata } from "next";
import { Chakra_Petch, Inter, JetBrains_Mono } from "next/font/google";
import "../globals.css";

const chakraPetch = Chakra_Petch({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-display",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Sign In - Ucotron",
  description: "Sign in to Ucotron Dashboard",
};

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${chakraPetch.variable} ${inter.variable} ${jetbrainsMono.variable} dark`}
      >
        <div className="flex min-h-screen items-center justify-center bg-background">
          {/* Background grid effect */}
          <div
            className="pointer-events-none fixed inset-0 opacity-[0.03]"
            style={{
              backgroundImage:
                "linear-gradient(rgba(0,240,255,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(0,240,255,0.3) 1px, transparent 1px)",
              backgroundSize: "40px 40px",
            }}
          />
          <div className="relative z-10 w-full max-w-md px-4">{children}</div>
        </div>
      </body>
    </html>
  );
}
