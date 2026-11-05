import { withSentryConfig } from '@sentry/nextjs/config'
import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  devIndicators: false,
  // The UI is a static export, so the switch is read when it is built or served.
  env: {
    NEXT_PUBLIC_KOHARU_DISABLE_TELEMETRY: process.env.KOHARU_DISABLE_TELEMETRY ?? '',
  },
  transpilePackages: ['@koharu/bridge', '@koharu/ui'],
  output: 'export',
  images: {
    unoptimized: true,
  },
}

export default withSentryConfig(nextConfig, {
  org: process.env.SENTRY_ORG,
  project: 'nextjs',
  authToken: process.env.SENTRY_AUTH_TOKEN,
  silent: !process.env.CI,
})
