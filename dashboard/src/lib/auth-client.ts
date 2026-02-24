import { createAuthClient } from "better-auth/react";

export const authClient = createAuthClient({
  baseURL: process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",
});

export const { signIn, signUp, signOut, useSession } = authClient;

/** Refresh the tier cookie after session changes. */
export async function refreshTierCookie(): Promise<void> {
  try {
    const session = await authClient.getSession();
    if (session?.data?.user) {
      document.cookie = `ucotron-tier=free; path=/; max-age=${60 * 60 * 24 * 30}; SameSite=Lax`;
    }
  } catch {
    // Silently ignore — tier cookie is best-effort
  }
}
