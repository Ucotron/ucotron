# Dashboard QA Report — V2-017

**Date:** 2026-02-26
**Server:** localhost:8420 (ucotron_server v0.1.0, no LLM)
**Dashboard:** localhost:3000 (Next.js 15.5.12 dev mode)
**Auth:** Middleware bypass (SKIP_AUTH=true) — no database available

---

## Page Load Results

| # | Page | URL | Status | Notes |
|---|------|-----|--------|-------|
| 1 | Overview | /overview | PASS | Connected to server, shows stats, models, namespaces |
| 2 | Memories | /memories | PASS | Search, filters, type dropdown, pagination all render |
| 3 | Entities | /entities | PASS | List/detail split view renders correctly |
| 4 | Graph | /graph | PASS | Type filter, limit selector, search, legend all render |
| 5 | Memory Chat | /memory-chat | PASS | Chat input, LLM Settings button, namespace indicator |
| 6 | Agents | /agents | PASS | Create Agent button, Refresh, empty state |
| 7 | Connectors | /connectors | FIXED | Was showing error banner for "scheduling not enabled"; now shows clean disabled state |
| 8 | Conversations | /conversations | PASS | Namespace label, empty state message |
| 9 | Tracing | /tracing | PASS | Query input, max memories selector, Trace button |
| 10 | Settings | /settings | PASS | 4 tabs (Profile, API Keys, Notifications, Preferences), forms render |
| 11 | Search | /search | PASS | Search input, Basic/Augment toggle, example queries |
| 12 | Namespaces | /namespaces | PASS | Shows 2 namespaces with stats, Switch/Delete actions |
| 13 | Onboarding | /onboarding | PASS | 4-step wizard (Welcome, Connect Data, Search, Explore) |
| 14 | Login | /login | PASS | OAuth buttons (Google, GitHub, Microsoft), email/password form |
| 15 | Sign Up | /signup | PASS | Same OAuth + name/email/password form |
| 16 | Forgot Password | /forgot-password | PASS | Email input, back to login link |
| 17 | Verify Email | /verify-email | PASS | Shows "No verification token" (expected without token) |
| 18 | Reset Password | /reset-password | PASS | Shows "Invalid link" (expected without token) |
| 19 | Root (/) | / | PASS | Redirects to /overview correctly |

**Result: 18/18 pages load (1 bug fixed during QA)**

---

## Sidebar Navigation

- All 10 sidebar links present: Overview, Memories, Entities, Graph, Memory Chat, Agents, Connectors, Conversations, Tracing, Settings
- Active page correctly highlighted (cyan background)
- Clicking each link navigates to correct URL
- Navigation state persists correctly
- **PASS**

---

## Bugs Found & Fixed

### BUG-D1: Connectors page shows error when scheduling not enabled (FIXED)
- **Severity:** P3 (cosmetic, non-blocking)
- **Description:** Connectors page called `/api/v1/connectors/schedules` which returns 400 when connector scheduling feature is not enabled on the server. This showed a raw error banner.
- **Fix:** Added `schedulingDisabled` state that detects this specific error message and shows a clean "Connector Scheduling Not Available" card instead. Also disabled the "Add Connector" button when scheduling is unavailable.
- **File:** `dashboard/src/app/(dashboard)/connectors/page.tsx`

---

## Minor Issues (Not Fixed — Low Priority)

### ISSUE-D2: ucotron_logo.png missing alt text
- **Severity:** P4 (accessibility warning)
- **Description:** Image with src "/ucotron_logo.png" has empty alt attribute. Next.js logs a warning.
- **Impact:** Console warning only, no visual impact

### ISSUE-D3: /api/auth/get-session returns 500 without database
- **Severity:** P4 (expected behavior without DB)
- **Description:** BetterAuth session endpoint fails without a PostgreSQL database connection. This is expected — the auth provider gracefully handles null sessions.
- **Impact:** Console error only, no visual impact (middleware bypass prevents redirects)

### ISSUE-D4: Settings heading shows "Configuracion" (Spanish)
- **Severity:** P4 (i18n)
- **Description:** The settings page heading shows "Configuracion" instead of "Settings" or "Configuration". This appears to be a locale/i18n issue where the Spanish translation is used by default.
- **Impact:** Minor visual inconsistency

---

## Test Environment

- **Auth bypass:** Added `SKIP_AUTH=true` env var check in `dashboard/src/middleware.ts` for local dev QA
- **No database:** Dashboard tested without PostgreSQL — all pages render, auth API calls fail gracefully
- **Server data:** 2 namespaces (default: 100 memories, v2007-bench: 25 memories + 136 entities)

---

## Summary

- **18/18 pages load successfully**
- **1 bug found and fixed** (connectors scheduling error)
- **3 minor issues documented** (logo alt, auth without DB, i18n heading)
- **Sidebar navigation works correctly**
- **No blank pages or crashes**
