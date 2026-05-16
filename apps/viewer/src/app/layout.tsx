/**
 * GeoEngine — Root Layout (Next.js 14 App Router)
 */

import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title:       "GeoEngine — 3D Геопросторовий Рушій",
  description: "Реальний 3D рельєф, супутникові знімки, симуляції",
  viewport:    "width=device-width, initial-scale=1",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="uk">
      <body className="bg-black overflow-hidden antialiased">
        {children}
      </body>
    </html>
  )
}
