"use client";

import { useEffect, useState, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { CheckCircle, XCircle, Loader2 } from "lucide-react";

function VerifyEmailPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [status, setStatus] = useState<"loading" | "success" | "error">("loading");
  const [message, setMessage] = useState("");

  useEffect(() => {
    const token = searchParams.get("token");
    
    if (!token) {
      setStatus("error");
      setMessage("No verification token provided");
      return;
    }

    async function verifyEmail() {
      try {
        const response = await fetch("/api/auth/verify-email", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token }),
        });

        if (response.ok) {
          setStatus("success");
          setTimeout(() => {
            router.push("/overview");
          }, 3000);
        } else {
          const data = await response.json();
          setStatus("error");
          setMessage(data.message || "Verification failed");
        }
      } catch {
        setStatus("error");
        setMessage("Failed to verify email");
      }
    }

    verifyEmail();
  }, [searchParams, router]);

  return (
    <div className="glass-card rounded-xl p-8">
      <div className="text-center">
        {status === "loading" && (
          <>
            <Loader2 className="mx-auto h-12 w-12 animate-spin text-primary" />
            <h1 className="mt-6 font-display text-2xl font-bold uppercase tracking-wider text-primary">
              Verifying email...
            </h1>
          </>
        )}

        {status === "success" && (
          <>
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-green-500/20">
              <CheckCircle className="h-6 w-6 text-green-500" />
            </div>
            <h1 className="font-display text-2xl font-bold uppercase tracking-wider text-primary">
              Email verified!
            </h1>
            <p className="mt-2 text-sm text-foreground/60">
              Your email has been successfully verified.
            </p>
            <p className="mt-2 text-sm text-foreground/50">
              Redirecting to dashboard...
            </p>
          </>
        )}

        {status === "error" && (
          <>
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-red-500/20">
              <XCircle className="h-6 w-6 text-red-500" />
            </div>
            <h1 className="font-display text-2xl font-bold uppercase tracking-wider text-primary">
              Verification failed
            </h1>
            <p className="mt-2 text-sm text-foreground/60">
              {message || "The verification link is invalid or has expired."}
            </p>
            <a
              href="/login"
              className="mt-6 inline-block rounded-lg bg-primary px-4 py-2.5 text-sm font-semibold text-background transition-colors hover:bg-primary/90"
            >
              Go to login
            </a>
          </>
        )}
      </div>
    </div>
  );
}

export default function VerifyEmailPage() {
  return (
    <Suspense fallback={<div className="flex min-h-screen items-center justify-center"><div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" /></div>}>
      <VerifyEmailPageContent />
    </Suspense>
  );
}
