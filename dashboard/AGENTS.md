# Ucotron Dashboard

## Build & Run

```bash
# Install dependencies
npm install

# Development server
npm run dev

# Production build
npm run build

# Lint
npm run lint

# Type check
npx tsc --noEmit
```

## Database (Drizzle ORM + Neon PostgreSQL)

```bash
# Generate migrations
npm run db:generate

# Push schema changes directly
npm run db:push

# Open Drizzle Studio
npm run db:studio
```

## Authentication (BetterAuth)

### Files
- `src/lib/auth.ts` - BetterAuth configuration
- `src/lib/db/index.ts` - Database connection (Neon + Drizzle)
- `src/lib/db/schema.ts` - Auth schema (users, sessions, accounts, api_keys, roles)

### Environment Variables
See `.env.example` for required variables:
- `DATABASE_URL` - Neon PostgreSQL connection string
- `BETTER_AUTH_SECRET` - Secret for auth token signing (32+ chars)
- `NEXT_PUBLIC_APP_URL` - App URL for callbacks
- OAuth provider credentials (Google, GitHub, Microsoft)
- `RESEND_API_KEY` - For email verification

### Session Configuration
- 30-day sessions
- Cookie caching enabled (5 minutes)
- Cross-subdomain cookies supported

### OAuth Providers
Configured providers:
- Google
- GitHub
- Microsoft

## Patterns

### Database Schema
- Uses Drizzle ORM with PostgreSQL dialect
- Neon serverless driver for connection pooling
- Tables follow BetterAuth naming conventions (singular: `user`, `session`, `account`)

### Auth Integration
- BetterAuth with Drizzle adapter
- Email/password + OAuth authentication
- API key management schema ready (api_keys table)
- RBAC schema ready (roles, user_roles tables)

## Dependencies

- `better-auth` - Authentication library
- `drizzle-orm` - ORM for database operations
- `@neondatabase/serverless` - Neon PostgreSQL driver
- `drizzle-kit` - Schema migrations and introspection
