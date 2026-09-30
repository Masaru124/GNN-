"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import {
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
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Button } from "@/components/ui/button";
import { isAuthenticated, setAuthenticated } from "@/lib/auth";

type NavItem = {
  href: string;
  label: string;
  icon: LucideIcon;
};

type NavGroup = {
  label?: string;
  items: NavItem[];
};

const NAV_GROUPS: NavGroup[] = [
  { items: [{ href: "/", label: "Dashboard", icon: LayoutDashboard }] },
  {
    label: "Screening",
    items: [
      { href: "/predict", label: "Predict", icon: Upload },
      { href: "/batch", label: "Batch", icon: Layers },
      { href: "/search", label: "Search", icon: Search },
    ],
  },
  {
    label: "Analysis",
    items: [
      { href: "/discovery", label: "Discovery", icon: Sparkles },
      { href: "/simulation", label: "Virtual Lab", icon: FlaskConical },
      { href: "/compare", label: "Compare", icon: GitCompare },
    ],
  },
  {
    label: "Data",
    items: [{ href: "/history", label: "History", icon: History }],
  },
];

const isActive = (pathname: string, href: string) =>
  href === "/" ? pathname === "/" : pathname.startsWith(href);

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [prevPathname, setPrevPathname] = useState(pathname);
  const authed = isAuthenticated();

  if (prevPathname !== pathname) {
    setPrevPathname(pathname);
    setOpen(false);
  }

  const closeDrawer = () => setOpen(false);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open]);

  const handleLogout = () => {
    setAuthenticated(false);
    setOpen(false);
    router.push("/login");
  };

  const renderBrand = () => (
    <Link
      href="/"
      className="group flex shrink-0 items-center gap-2.5 focus-ring rounded-lg"
      aria-label="MatScreen AI — Home"
      onClick={closeDrawer}
    >
      <span className="leading-tight">
        <span className="block text-[15px] font-bold tracking-tight text-foreground">
          MatScreen AI
        </span>
      </span>
    </Link>
  );

  const renderNav = () => (
    <div className="space-y-5">
      {NAV_GROUPS.map((group) => (
        <div key={group.label ?? "__top__"}>
          {group.label && <p className="px-3 pb-1.5">{group.label}</p>}
          <ul className="space-y-0.5">
            {group.items.map((item) => {
              const Icon = item.icon;
              const active = isActive(pathname, item.href);
              return (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    onClick={closeDrawer}
                    className={cn(
                      "relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors duration-150 focus-ring",
                      active
                        ? "bg-muted text-foreground"
                        : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                    )}
                  >
                    {active && (
                      <span
                        aria-hidden="true"
                        className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-primary"
                      />
                    )}
                    <Icon
                      className={cn(
                        "h-4 w-4 transition-colors",
                        active ? "text-primary" : "text-muted-foreground",
                      )}
                    />
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </div>
  );

  return (
    <>
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col border-r border-border/80 bg-muted/30 lg:flex">
        <div className="flex h-16 shrink-0 items-center px-4">
          {renderBrand()}
        </div>
        <nav aria-label="Primary" className="flex-1 overflow-y-auto px-3 pb-4">
          {renderNav()}
        </nav>
        <div className="shrink-0 border-t border-border/80 p-3">
          <div className="flex flex-col gap-2">
            <ThemeToggle />

            {authed ? (
              <Button
                variant="destructive"
                size="lg"
                onClick={handleLogout}
                className="w-full justify-start gap-2"
              >
                <LogOut className="h-4 w-4" />
                <span>Logout</span>
              </Button>
            ) : (
              <Button
                variant="secondary"
                size="lg"
                onClick={() => router.push("/login")}
                className="w-full justify-start gap-2"
              >
                <LogIn className="h-4 w-4" />
                <span>Sign In</span>
              </Button>
            )}
          </div>
        </div>
      </aside>

      <header className="sticky top-0 z-40 flex h-16 shrink-0 items-center justify-between gap-4 border-b border-border/80 bg-background/85 px-4 backdrop-blur-md supports-[backdrop-filter]:bg-background/75 lg:hidden">
        {renderBrand()}
        <div className="flex items-center gap-1.5 sm:gap-2">
          <ThemeToggle />
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            aria-label="Toggle navigation menu"
            aria-expanded={open}
            aria-controls="sidebar-drawer"
            onClick={() => setOpen(true)}
          >
            {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </div>
      </header>

      <div
        className={cn(
          "fixed inset-0 z-50 lg:hidden",
          !open && "pointer-events-none",
        )}
        role="dialog"
        aria-modal="true"
        aria-label="Primary"
        inert={!open}
      >
        <div
          aria-hidden="true"
          onClick={closeDrawer}
          className={cn(
            "absolute inset-0 bg-foreground/40 backdrop-blur-sm transition-opacity duration-200 ease-out",
            open ? "opacity-100" : "opacity-0",
          )}
        />
        <aside
          id="sidebar-drawer"
          className={cn(
            "absolute inset-y-0 left-0 flex w-72 max-w-[85vw] flex-col border-r border-border bg-background shadow-xl transition-transform duration-200 ease-out",
            open ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <div className="flex h-16 shrink-0 items-center justify-between gap-2 border-b border-border/80 px-4">
            {renderBrand()}
            <Button
              variant="ghost"
              size="icon"
              aria-label="Close navigation menu"
              onClick={closeDrawer}
            >
              <X className="h-5 w-5" />
            </Button>
          </div>
          <nav
            aria-label="Primary"
            className="flex-1 overflow-y-auto px-3 py-4"
          >
            {renderNav()}
          </nav>
          <div className="shrink-0 border-t border-border/80 p-3 text-center">
            {authed ? (
              <Button
                variant="destructive"
                className="w-full text-center"
                onClick={handleLogout}
              >
                <LogOut className="h-4 w-4" />
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
        </aside>
      </div>
    </>
  );
}
