"use client";

import {
  createContext,
  useContext,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import { authClient, signIn, signUp, signOut, useSession, refreshTierCookie } from "@/lib/auth-client";

interface AuthUser {
  id: string;
  name: string;
  email: string;
  emailVerified: boolean;
  image?: string | null;
  createdAt: Date;
  updatedAt: Date;
}

interface AuthSession {
  id: string;
  token: string;
  expiresAt: Date;
  createdAt: Date;
  updatedAt: Date;
  ipAddress?: string | null;
  userAgent?: string | null;
  userId: string;
}

interface AuthContextValue {
  /** Current user object, null if not authenticated */
  user: AuthUser | null;
  /** Current session object, null if not authenticated */
  session: AuthSession | null;
  /** Whether the session is currently being fetched */
  isLoading: boolean;
  /** Whether the user is authenticated */
  isAuthenticated: boolean;
  /** Sign in with OAuth provider */
  signInWithProvider: (provider: "google" | "github" | "microsoft") => Promise<void>;
  /** Sign in with email and password */
  signInWithEmail: (email: string, password: string) => Promise<void>;
  /** Sign up with email and password */
  signUpWithEmail: (email: string, password: string, name: string) => Promise<void>;
  /** Sign out the current user */
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  session: null,
  isLoading: true,
  isAuthenticated: false,
  signInWithProvider: async () => {},
  signInWithEmail: async () => {},
  signUpWithEmail: async () => {},
  signOut: async () => {},
});

export function useAuth(): AuthContextValue {
  return useContext(AuthContext);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const { data, isPending } = useSession();

  useEffect(() => {
    if (data?.user) {
      refreshTierCookie().catch(() => {});
    }
  }, [data?.user]);

  const handleSignInWithProvider = useCallback(
    async (provider: "google" | "github" | "microsoft") => {
      await signIn.social({
        provider,
        callbackURL: "/",
      });
    },
    [],
  );

  const handleSignInWithEmail = useCallback(
    async (email: string, password: string) => {
      await signIn.email({
        email,
        password,
      });
    },
    [],
  );

  const handleSignUpWithEmail = useCallback(
    async (email: string, password: string, name: string) => {
      await signUp.email({
        email,
        password,
        name,
      });
    },
    [],
  );

  const handleSignOut = useCallback(async () => {
    await signOut();
  }, []);

  const value: AuthContextValue = {
    user: data?.user ?? null,
    session: data?.session ?? null,
    isLoading: isPending,
    isAuthenticated: !!data?.user,
    signInWithProvider: handleSignInWithProvider,
    signInWithEmail: handleSignInWithEmail,
    signUpWithEmail: handleSignUpWithEmail,
    signOut: handleSignOut,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}
