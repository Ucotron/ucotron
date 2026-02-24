"use client";

import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";

interface NamespaceContextValue {
  namespace: string;
  setNamespace: (ns: string) => void;
}

const NamespaceContext = createContext<NamespaceContextValue>({
  namespace: "default",
  setNamespace: () => {},
});

export function NamespaceProvider({ children }: { children: ReactNode }) {
  const [namespace, setNamespace] = useState("default");

  return (
    <NamespaceContext.Provider value={{ namespace, setNamespace }}>
      {children}
    </NamespaceContext.Provider>
  );
}

export function useNamespace(): NamespaceContextValue {
  return useContext(NamespaceContext);
}
