"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

type Locale = "en" | "es";

interface LocaleContextType {
  locale: Locale;
  setLocale: (locale: Locale) => void;
}

const LocaleContext = createContext<LocaleContextType | undefined>(undefined);

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en");
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("ucotron-language") as Locale;
    if (stored && (stored === "en" || stored === "es")) {
      setLocaleState(stored);
    } else {
      const browserLang = navigator.language.split("-")[0];
      if (browserLang === "es") {
        setLocaleState("es");
      }
    }
    setIsInitialized(true);
  }, []);

  const setLocale = (newLocale: Locale) => {
    setLocaleState(newLocale);
    localStorage.setItem("ucotron-language", newLocale);
  };

  if (!isInitialized) {
    return null;
  }

  return (
    <LocaleContext.Provider value={{ locale, setLocale }}>
      {children}
    </LocaleContext.Provider>
  );
}

export function useLocale() {
  const context = useContext(LocaleContext);
  if (context === undefined) {
    throw new Error("useLocale must be used within a LocaleProvider");
  }
  return context;
}
