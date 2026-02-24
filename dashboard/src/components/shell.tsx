"use client";

import { type ReactNode } from "react";
import { ThemeProvider } from "./theme-provider";
import { AuthProvider } from "./auth-provider";
import { Sidebar } from "./sidebar";
import { Header } from "./header";
import { NamespaceProvider, useNamespace } from "./namespace-context";

function ShellInner({ children }: { children: ReactNode }) {
  const { namespace, setNamespace } = useNamespace();

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header namespace={namespace} onNamespaceChange={setNamespace} />
        <main className="flex-1 overflow-y-auto p-6">
          {children}
        </main>
      </div>
    </div>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider>
      <AuthProvider>
        <NamespaceProvider>
          <ShellInner>{children}</ShellInner>
        </NamespaceProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}
