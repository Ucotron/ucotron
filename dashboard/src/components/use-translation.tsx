"use client";

import en from "../../messages/en.json";
import es from "../../messages/es.json";
import { useLocale } from "./locale-provider";

const dictionaries = { en, es };

type Dictionary = typeof en;

function getNestedValue(obj: Dictionary, path: string): string {
  const keys = path.split(".");
  let result: unknown = obj;
  for (const key of keys) {
    if (result && typeof result === "object" && key in result) {
      result = (result as Record<string, unknown>)[key];
    } else {
      return path;
    }
  }
  return typeof result === "string" ? result : path;
}

export function useTranslation() {
  const { locale } = useLocale();

  const t = (key: string): string => {
    return getNestedValue(dictionaries[locale], key);
  };

  return { t, locale };
}
