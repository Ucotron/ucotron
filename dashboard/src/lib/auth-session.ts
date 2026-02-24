import { headers } from "next/headers";
import { auth } from "./auth";

export interface SessionInfo {
  user: {
    id: string;
    email: string;
    name: string;
    image?: string | null;
    emailVerified: boolean;
    createdAt: Date;
    updatedAt: Date;
  };
  session: {
    id: string;
    userId: string;
    expiresAt: Date;
    token: string;
    createdAt: Date;
    updatedAt: Date;
    ipAddress?: string | null;
    userAgent?: string | null;
  };
}

export async function getServerSession(): Promise<{ user: unknown; session: unknown } | null> {
  const session = await auth.api.getSession({
    headers: await headers(),
  });
  return session;
}

export async function getSessionInfo(): Promise<SessionInfo | null> {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session?.user?.id) {
    return null;
  }

  return {
    user: session.user as SessionInfo["user"],
    session: session.session as SessionInfo["session"],
  };
}
