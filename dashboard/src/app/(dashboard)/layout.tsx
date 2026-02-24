"use client";

import { type ReactNode } from "react";
import { ThemeProvider } from "@/components/theme-provider";
import { NamespaceProvider } from "@/components/namespace-context";
import { LocaleProvider } from "@/components/locale-provider";
import { Sidebar } from "@/components/sidebar";
import { Header } from "@/components/header";
import { useNamespace } from "@/components/namespace-context";

function DashboardLayoutInner({
  children,
}: {
  children: ReactNode;
}) {
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

export default function DashboardLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <ThemeProvider>
      <NamespaceProvider>
        <LocaleProvider>
          <DashboardLayoutInner>{children}</DashboardLayoutInner>
        </LocaleProvider>
      </NamespaceProvider>
    </ThemeProvider>
  );
}
