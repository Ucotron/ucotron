# =============================================================================
# Ucotron Dashboard — Next.js standalone build
# =============================================================================
# Context: repo root (to access packages/ui for @ucotron/ui dependency)
#
# Stage 1: Install dependencies
# Stage 2: Build Next.js app with standalone output
# Stage 3: Minimal runtime with Node.js
# =============================================================================

# ─── Dependencies Stage ──────────────────────────────────────────────────────
FROM node:20-alpine AS deps

WORKDIR /app

# Copy the UI package that dashboard depends on via file: reference
COPY packages/ui/ ./packages/ui/

# Copy dashboard package manifests
COPY dashboard/package.json dashboard/package-lock.json ./dashboard/

# Install UI package dependencies first, then dashboard dependencies
RUN cd packages/ui && npm install --ignore-scripts
RUN cd dashboard && npm ci

# ─── Build Stage ─────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy UI package with its node_modules
COPY --from=deps /app/packages/ui ./packages/ui

# Copy dashboard with its node_modules
COPY --from=deps /app/dashboard/node_modules ./dashboard/node_modules

# Copy dashboard source
COPY dashboard/ ./dashboard/

# Set Next.js to produce standalone output
ENV NEXT_TELEMETRY_DISABLED=1
# Dummy env vars required at build time (modules init at import)
ENV DATABASE_URL="postgresql://build:build@localhost:5432/build"
ENV BETTER_AUTH_SECRET="build-time-dummy-secret-not-used-at-runtime"
ENV RESEND_API_KEY="re_build_dummy"
ENV NEXT_PUBLIC_APP_URL="http://localhost:3000"

RUN cd dashboard && npm run build

# ─── Runtime Stage ───────────────────────────────────────────────────────────
FROM node:20-alpine AS runtime

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy standalone output
COPY --from=builder /app/dashboard/.next/standalone ./
COPY --from=builder /app/dashboard/.next/static ./dashboard/.next/static
COPY --from=builder /app/dashboard/public ./dashboard/public

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/ || exit 1

CMD ["node", "dashboard/server.js"]
