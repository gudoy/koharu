import type { Metadata } from 'next'
import { ThemeProvider } from 'next-themes'

import './globals.css'
import './fonts/noto-sans-cjk.css'
import Providers from '@/app/providers'

export const metadata: Metadata = {
  title: 'Koharu',
}

function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang='en-US' suppressHydrationWarning style={{ backgroundColor: 'transparent' }}>
      <body className='antialiased' style={{ backgroundColor: 'transparent' }}>
        <ThemeProvider attribute='class' defaultTheme='system' enableSystem>
          <Providers>{children}</Providers>
        </ThemeProvider>
      </body>
    </html>
  )
}

export default RootLayout
