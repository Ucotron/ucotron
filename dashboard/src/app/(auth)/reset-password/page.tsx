"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2, CheckCircle, XCircle } from "lucide-react";
import { authClient } from "@/lib/auth-client";

export default function ResetPasswordPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get("token");

  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [invalidToken, setInvalidToken] = useState(!token);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (password.length < 8) {
      setError("Password must be at least 8 characters");
      return;
    }

    if (!token) {
      setInvalidToken(true);
      return;
    }

    setLoading(true);

    try {
      await authClient.resetPassword({
        token,
        newPassword: password,
      });
      setSuccess(true);
      setTimeout(() => {
        router.push("/login");
      }, 3000);
    } catch (err) {
      setError("Failed to reset password. The link may have expired.");
      setLoading(false);
    }
  }

  if (success) {
    return (
      <div className="glass-card rounded-xl p-8">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-green-500/20">
            <CheckCircle className="h-6 w-6 text-green-500" />
          </div>
          <h1 className="font-display text-2xl font-bold uppercase tracking-wider text-primary">
            Password reset
          </h1>
          <p className="mt-2 text-sm text-foreground/60">
            Your password has been successfully reset.
          </p>
          <p className="mt-2 text-sm text-foreground/50">
            Redirecting to login...
          </p>
        </div>
      </div>
    );
  }

  if (invalidToken) {
    return (
      <div className="glass-card rounded-xl p-8">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-red-500/20">
            <XCircle className="h-6 w-6 text-red-500" />
          </div>
          <h1 className="font-display text-2xl font-bold uppercase tracking-wider text-primary">
            Invalid link
          </h1>
          <p className="mt-2 text-sm text-foreground/60">
            This password reset link is invalid or has expired.
          </p>
        </div>

        <Link
          href="/forgot-password"
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-background transition-colors hover:bg-primary/90"
        >
          Request a new reset link
        </Link>

        <div className="mt-6 text-center">
          <Link
            href="/login"
            className="inline-flex items-center gap-2 text-sm text-foreground/60 hover:text-foreground"
          >
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
          Reset your password
        </h1>
        <p className="mt-2 text-sm text-foreground/60">
          Enter your new password below
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label
            htmlFor="password"
            className="mb-1.5 block text-sm font-medium text-foreground/80"
          >
            New Password
          </label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Enter new password"
            required
            minLength={8}
            className="w-full rounded-lg border border-border bg-background/50 px-4 py-2.5 text-sm placeholder:text-foreground/30 focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
          <p className="mt-1 text-xs text-foreground/40">
            Must be at least 8 characters
          </p>
        </div>

        <div>
          <label
            htmlFor="confirmPassword"
            className="mb-1.5 block text-sm font-medium text-foreground/80"
          >
            Confirm Password
          </label>
          <input
            id="confirmPassword"
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="Confirm new password"
            required
            minLength={8}
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
          Reset Password
        </button>
      </form>

      <div className="mt-6 text-center">
        <Link
          href="/login"
          className="text-sm text-foreground/60 hover:text-foreground"
        >
          Back to login
        </Link>
      </div>
    </div>
  );
}
