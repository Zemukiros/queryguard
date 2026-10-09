import { useCallback, useEffect, useState } from "react";

type Theme = "light" | "dark";
const KEY = "queryguard-theme";

function stored(): Theme | null {
  try {
    const value = localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

/** Light until the visitor picks dark, whatever the OS setting; the pick is remembered when storage allows. */
export function useTheme() {
  const [choice, setChoice] = useState<Theme | null>(stored);
  const theme = choice ?? "light";

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  const toggle = useCallback(() => {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setChoice(next);
    try {
      localStorage.setItem(KEY, next);
    } catch {
      // private mode or blocked storage: the choice lasts for this page only
    }
  }, [theme]);

  return { theme, toggle };
}
