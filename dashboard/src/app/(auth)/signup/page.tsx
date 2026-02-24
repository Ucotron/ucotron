"use client";

import { useState, Suspense } from "react";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { signIn, signUp } from "@/lib/auth-client";

function SignupPageContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const callbackUrl = searchParams.get("callbackUrl") || "/overview";
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [oauthLoading, setOauthLoading] = useState<string | null>(null);

  async function handleEmailSignup(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const result = await signUp.email({
        name,
        email,
        password,
        callbackURL: callbackUrl,
      });
      if (result.error) {
        setError(result.error.message ?? "Failed to create account");
        setLoading(false);
      } else {
        router.push(callbackUrl);
      }
    } catch {
      setError("Something went wrong. Please try again.");
      setLoading(false);
    }
  }

  async function handleOAuthLogin(provider: "google" | "github" | "microsoft") {
    setError(null);
    setOauthLoading(provider);
    try {
      await signIn.social({
        provider,
        callbackURL: callbackUrl,
      });
    } catch {
      setError(`Failed to sign in with ${provider}`);
      setOauthLoading(null);
    }
  }

  return (
    <div className="glass-card rounded-xl p-8">
      {/* Logo / Brand */}
      <div className="mb-8 text-center">
        <h1 className="font-display text-2xl font-bold uppercase tracking-wider text-primary">
          Ucotron
        </h1>
        <p className="mt-2 text-sm text-foreground/60">
          Create your account
        </p>
      </div>

      {/* OAuth Buttons */}
      <div className="space-y-3">
        <button
          type="button"
          onClick={() => handleOAuthLogin("google")}
          disabled={!!oauthLoading}
          className="flex w-full items-center justify-center gap-3 rounded-lg border border-border bg-background/50 px-4 py-2.5 text-sm font-medium transition-colors hover:bg-muted disabled:opacity-50"
        >
          {oauthLoading === "google" ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <GoogleIcon />
          )}
          Continue with Google
        </button>

        <button
          type="button"
          onClick={() => handleOAuthLogin("github")}
          disabled={!!oauthLoading}
          className="flex w-full items-center justify-center gap-3 rounded-lg border border-border bg-background/50 px-4 py-2.5 text-sm font-medium transition-colors hover:bg-muted disabled:opacity-50"
        >
          {oauthLoading === "github" ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <GitHubIcon />
          )}
          Continue with GitHub
        </button>

        <button
          type="button"
          onClick={() => handleOAuthLogin("microsoft")}
          disabled={!!oauthLoading}
          className="flex w-full items-center justify-center gap-3 rounded-lg border border-border bg-background/50 px-4 py-2.5 text-sm font-medium transition-colors hover:bg-muted disabled:opacity-50"
        >
          {oauthLoading === "microsoft" ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <MicrosoftIcon />
          )}
          Continue with Microsoft
        </button>
      </div>

      {/* Divider */}
      <div className="my-6 flex items-center gap-3">
        <div className="h-px flex-1 bg-border" />
        <span className="text-xs uppercase tracking-wider text-foreground/40">
          or
        </span>
        <div className="h-px flex-1 bg-border" />
      </div>

      {/* Email / Password Form */}
      <form onSubmit={handleEmailSignup} className="space-y-4">
        <div>
          <label
            htmlFor="name"
            className="mb-1.5 block text-sm font-medium text-foreground/80"
          >
            Name
          </label>
          <input
            id="name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Your name"
            required
            className="w-full rounded-lg border border-border bg-background/50 px-4 py-2.5 text-sm placeholder:text-foreground/30 focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>

        <div>
          <label
            htmlFor="email"
            className="mb-1.5 block text-sm font-medium text-foreground/80"
          >
            Email
          </label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
            required
            className="w-full rounded-lg border border-border bg-background/50 px-4 py-2.5 text-sm placeholder:text-foreground/30 focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>

        <div>
          <label
            htmlFor="password"
            className="mb-1.5 block text-sm font-medium text-foreground/80"
          >
            Password
          </label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Create a password"
            required
            minLength={8}
            className="w-full rounded-lg border border-border bg-background/50 px-4 py-2.5 text-sm placeholder:text-foreground/30 focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
          <p className="mt-1 text-xs text-foreground/40">
            Must be at least 8 characters
          </p>
        </div>

        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}

        <button
          type="submit"
          disabled={loading}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-background transition-colors hover:bg-primary/90 disabled:opacity-50"
        >
          {loading && <Loader2 className="h-4 w-4 animate-spin" />}
          Create Account
        </button>
      </form>

      {/* Footer */}
      <p className="mt-6 text-center text-sm text-foreground/50">
        Already have an account?{" "}
        <Link
          href="/login"
          className="font-medium text-primary hover:text-primary/80"
        >
          Sign in
        </Link>
      </p>
    </div>
  );
}

function GoogleIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24">
      <path
        fill="#4285F4"
        d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"
      />
      <path
        fill="#34A853"
        d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
      />
      <path
        fill="#FBBC05"
        d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
      />
      <path
        fill="#EA4335"
        d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
      />
    </svg>
  );
}

function GitHubIcon() {
  return (
    <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 24 24">
      <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z" />
    </svg>
  );
}

function MicrosoftIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24">
      <path fill="#F25022" d="M1 1h10v10H1z" />
      <path fill="#00A4EF" d="M1 13h10v10H1z" />
      <path fill="#7FBA00" d="M13 1h10v10H13z" />
      <path fill="#FFB900" d="M13 13h10v10H13z" />
    </svg>
  );
}

export default function SignupPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center"><div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" /></div>}>
      <SignupPageContent />
    </Suspense>
  );
}
