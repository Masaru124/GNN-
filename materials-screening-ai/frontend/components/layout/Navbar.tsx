"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import {
  Atom,
  Menu,
  X,
  LayoutDashboard,
  Upload,
  Layers,
  Search,
  GitCompare,
  History,
  LogOut,
  LogIn,
  Sparkles,
  FlaskConical,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { isAuthenticated, setAuthenticated } from "@/lib/auth";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/discovery", label: "Discovery", icon: Sparkles },
  { href: "/simulation", label: "Virtual Lab", icon: FlaskConical },
  { href: "/predict", label: "Predict", icon: Upload },
  { href: "/batch", label: "Batch", icon: Layers },
  { href: "/search", label: "Search", icon: Search },
  { href: "/compare", label: "Compare", icon: GitCompare },
  { href: "/history", label: "History", icon: History },
];

export function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const authed = isAuthenticated();

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  const handleLogout = () => {
    setAuthenticated(false);
    setOpen(false);
    router.push("/login");
  };

  return (
    <header className="sticky top-0 z-40 border-b border-border/80 bg-background/85 backdrop-blur-md supports-[backdrop-filter]:bg-background/75">
      <div className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <Link
          href="/"
          className="group flex shrink-0 items-center gap-2.5 focus-ring rounded-lg"
          aria-label="MatScreen AI — Home"
        >
          <span className="leading-tight">
            <span className="block text-[15px] font-bold tracking-tight text-foreground">
              MatScreen AI
            </span>
          </span>
        </Link>

        <nav aria-label="Primary" className="hidden items-center gap-1 lg:flex">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "relative flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors duration-150 focus-ring",
                  active
                    ? "text-foreground"
                    : "text-muted-foreground hover:bg-muted/70 hover:text-foreground",
                )}
              >
                <Icon
                  className={cn(
                    "h-4 w-4 transition-colors",
                    active ? "text-primary" : "text-muted-foreground",
                  )}
                />
                {item.label}
                {active && (
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-3 -bottom-px h-0.5 rounded-full bg-primary"
                  />
                )}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-1.5 sm:gap-2">
          <ThemeToggle />
          {authed ? (
            <Button
              variant="destructive"
              size="sm"
              onClick={handleLogout}
              className="hidden sm:inline-flex"
            >
              Logout
            </Button>
          ) : (
            <Button
              variant="secondary"
              size="sm"
              onClick={() => router.push("/login")}
            >
              <LogIn className="h-4 w-4" />
              <span className="hidden sm:inline">Sign In</span>
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            aria-label="Toggle navigation menu"
            aria-expanded={open}
            aria-controls="mobile-nav"
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </div>
      </div>

      {open && (
        <nav
          id="mobile-nav"
          aria-label="Primary"
          className="animate-fade-in border-t border-border/80 px-4 py-3 lg:hidden"
        >
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-3 text-sm font-medium transition-colors focus-ring",
                  active
                    ? "bg-muted text-foreground"
                    : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                )}
              >
                <Icon
                  className={cn(
                    "h-4.5 w-4.5",
                    active ? "text-primary" : "text-muted-foreground",
                  )}
                />
                {item.label}
              </Link>
            );
          })}
          <div className="mt-2 border-t border-border/80 pt-3 lg:hidden">
            {authed ? (
              <Button
                variant="destructive"
                className="w-full"
                onClick={handleLogout}
              >
                Logout
              </Button>
            ) : (
              <Button
                variant="secondary"
                className="w-full"
                onClick={() => router.push("/login")}
              >
                <LogIn className="h-4 w-4" />
                Sign In
              </Button>
            )}
          </div>
        </nav>
      )}
    </header>
  );
}
