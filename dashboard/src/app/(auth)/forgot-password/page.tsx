"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Loader2, ArrowLeft, CheckCircle } from "lucide-react";
import { authClient } from "@/lib/auth-client";

export default function ForgotPasswordPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await authClient.requestPasswordReset({
        email,
        redirectTo: `${window.location.origin}/reset-password`,
      });
      setSubmitted(true);
    } catch (err) {
      setError("Failed to send reset email. Please try again.");
      setLoading(false);
    }
  }

  if (submitted) {
    return (
      <div className="glass-card rounded-xl p-8">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-green-500/20">
            <CheckCircle className="h-6 w-6 text-green-500" />
          </div>
          <h1 className="font-display text-2xl font-bold uppercase tracking-wider text-primary">
            Check your email
          </h1>
          <p className="mt-2 text-sm text-foreground/60">
            We sent a password reset link to
          </p>
          <p className="text-sm font-medium text-foreground">{email}</p>
        </div>

        <p className="text-center text-sm text-foreground/50">
          Didn&apos;t receive the email? Check your spam folder, or{" "}
          <button
            onClick={() => setSubmitted(false)}
            className="text-primary hover:text-primary/80"
          >
            try again
          </button>
        </p>

        <div className="mt-6 text-center">
          <Link
            href="/login"
            className="inline-flex items-center gap-2 text-sm text-foreground/60 hover:text-foreground"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to login
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card rounded-xl p-8">
      <div className="mb-8 text-center">
        <h1 className="font-display text-2xl font-bold uppercase tracking-wider text-primary">
          Forgot password?
        </h1>
        <p className="mt-2 text-sm text-foreground/60">
          Enter your email and we&apos;ll send you a reset link
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
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

        {error && <p className="text-sm text-destructive">{error}</p>}

        <button
          type="submit"
          disabled={loading}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-background transition-colors hover:bg-primary/90 disabled:opacity-50"
        >
          {loading && <Loader2 className="h-4 w-4 animate-spin" />}
          Send Reset Link
        </button>
      </form>

      <div className="mt-6 text-center">
        <Link
          href="/login"
          className="inline-flex items-center gap-2 text-sm text-foreground/60 hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to login
        </Link>
      </div>
    </div>
  );
}
