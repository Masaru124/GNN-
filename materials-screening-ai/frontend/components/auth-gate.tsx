"use client";

import { useEffect, useSyncExternalStore } from "react";
import { usePathname, useRouter } from "next/navigation";
import { isAuthenticated } from "@/lib/auth";

const emptySubscribe = () => () => {};

export function AuthGate({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const mounted = useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false,
  );

  useEffect(() => {
    if (pathname === "/login") return;
    if (!isAuthenticated()) {
      router.replace("/login");
    }
  }, [pathname, router]);

  if (pathname === "/login") return <>{children}</>;
  if (!mounted || !isAuthenticated()) return null;
  return <>{children}</>;
}
