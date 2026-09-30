"use client";

import { useSyncExternalStore } from "react";
import { useTheme } from "next-themes";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";

const emptySubscribe = () => () => {};

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const mounted = useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false,
  );
  const isDark = mounted && resolvedTheme === "dark";

  return (
    <Button
      variant="ghost"
      aria-label="Toggle theme"
      disabled={!mounted}
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className="gap-2 px-3  border border-neutral-300"
    >
      {isDark ? (
        <>
          <Sun className="h-4 w-4" />
          <span>Light</span>
        </>
      ) : (
        <>
          <Moon className="h-4 w-4" />
          <span>Dark</span>
        </>
      )}
    </Button>
  );
}
