import { act, renderHook } from "@testing-library/react";

import { useTheme } from "./useTheme";

const KEY = "queryguard-theme";

/** An OS that prefers dark: the page must still open light. */
function preferDark() {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: query.includes("dark"), media: query, onchange: null,
    addEventListener: () => undefined, removeEventListener: () => undefined,
    addListener: () => undefined, removeListener: () => undefined, dispatchEvent: () => false,
  }));
}

beforeEach(() => {
  localStorage.clear();
  delete document.documentElement.dataset.theme;
  preferDark();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

test("opens light even when the OS prefers dark", () => {
  const { result } = renderHook(() => useTheme());
  expect(result.current.theme).toBe("light");
  expect(document.documentElement.dataset.theme).toBe("light");
});

test("the toggle switches to dark and remembers it", () => {
  const { result } = renderHook(() => useTheme());
  act(() => { result.current.toggle(); });
  expect(result.current.theme).toBe("dark");
  expect(document.documentElement.dataset.theme).toBe("dark");
  expect(localStorage.getItem(KEY)).toBe("dark");

  const again = renderHook(() => useTheme());
  expect(again.result.current.theme).toBe("dark");
});

test("a remembered light choice stays light", () => {
  localStorage.setItem(KEY, "light");
  const { result } = renderHook(() => useTheme());
  expect(result.current.theme).toBe("light");
});
