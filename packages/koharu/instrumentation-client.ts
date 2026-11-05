import * as Sentry from '@sentry/nextjs'

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  // `KOHARU_DISABLE_TELEMETRY=1` when the UI is built or served sends nothing.
  enabled: process.env.NEXT_PUBLIC_KOHARU_DISABLE_TELEMETRY !== '1',
})

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart
