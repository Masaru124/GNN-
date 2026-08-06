# Design System

A premium, enterprise-grade visual system inspired by Cloudflare's design language: warm, technical, precise, and highly usable. Think software built by Cloudflare — not a clone, but the same level of refinement and craft.

---

## Stack

- **Framework:** Next.js App Router + React + TypeScript
- **Styling:** Tailwind CSS v4 via `@theme` / `@theme inline` in `app/globals.css`
- **Components:** shadcn-style primitives in `components/ui`
- **Icons:** Lucide React
- **Fonts:** IBM Plex Sans for UI + headings, IBM Plex Mono for numbers, labels, and code (loaded via `next/font/google` in `app/layout.tsx`)
- **Dark mode:** `next-themes`, class-based, dark default
- **Utilities:** `cn()` from `@/lib/utils`

---

## Visual Direction

The app is a modern enterprise AI platform — clean, confident, fast-feeling.

- Warm off-white canvas (`#FAF9F7`-style) in light mode; rich charcoal (never pure black) in dark mode.
- Warm Cloudflare-orange primary, neutral slate secondary, subtle blue accent used sparingly.
- Thin, low-contrast borders; slight card elevation; generous whitespace.
- Technical but approachable: mono labels, mono numerics, tight-tracked bold headings.
- **Avoid:** random gradients, neon colors, glassmorphism, heavy shadows, oversaturated palettes.

---

## Tokens

All semantic tokens live in `app/globals.css` and are bridged to Tailwind with `@theme inline`. Every color is defined in **OKLCH**.

| Token | Light (approx.) | Dark | Usage |
| --- | --- | --- | --- |
| `background` | `oklch(0.982 0.003 85)` | `oklch(0.16 0.006 85)` | Page canvas |
| `foreground` | `oklch(0.22 0.01 260)` | `oklch(0.95 0.005 85)` | Primary text |
| `card` | `oklch(1 0 0)` | `oklch(0.205 0.007 85)` | Card surfaces |
| `primary` | `oklch(0.723 0.172 54)` | `oklch(0.78 0.16 55)` | Warm orange — primary actions |
| `primary-foreground` | dark brown `oklch(0.2 0.03 45)` | dark `oklch(0.17 0.03 45)` | Text on orange (accessible) |
| `secondary` | `oklch(0.95 0.005 260)` | `oklch(0.255 0.008 260)` | Secondary surfaces |
| `muted` | `oklch(0.952 0.003 106)` | `oklch(0.245 0.007 85)` | Subtle surfaces |
| `muted-foreground` | `oklch(0.5 0.02 260)` | `oklch(0.72 0.01 260)` | Supporting copy |
| `accent` | `oklch(0.6 0.12 255)` | `oklch(0.76 0.1 245)` | Subtle blue, used sparingly |
| `border` | `oklch(0.91 0.008 260)` | `oklch(0.3 0.008 85)` | Thin, low-contrast borders |
| `ring` | `oklch(0.723 0.172 54)` | `oklch(0.78 0.16 55)` | Focus rings |
| `success` | `oklch(0.62 0.14 155)` | `oklch(0.72 0.14 155)` | Confident green |
| `warning` | `oklch(0.72 0.15 72)` | `oklch(0.82 0.14 75)` | Amber |
| `destructive` | `oklch(0.55 0.18 27)` | `oklch(0.64 0.17 27)` | Muted red |

Derived surface tints for badges/tags: `primary-muted`, `accent-muted`, `success-muted`, `warning-muted`, `destructive-muted` (the base color at ~10% alpha). Chart colors use `chart-1` (orange) through `chart-5`.

---

## Typography

| Token | Font | Usage |
| --- | --- | --- |
| `--font-sans` | IBM Plex Sans | Body, controls, forms, headings |
| `--font-display` | IBM Plex Sans | Aliased to sans (tight-tracked bold headings) |
| `--font-mono` | IBM Plex Mono | Numbers, technical values, labels, code |

Guidelines:

- Page headings: `text-2xl` → `text-4xl`, `font-bold tracking-tight`. The shared `PageHeader` component standardizes the eyebrow + title + description pattern.
- Eyebrows, table headers, and metadata use `.micro-label`: mono, uppercase, 11px, `tracking-[0.14em]`.
- **All numbers and technical values use `font-mono`.**
- Body copy: `text-[15px]`, `leading-relaxed`, `text-muted-foreground` for supporting text.

---

## Core Utilities

Defined in `app/globals.css`:

- `.micro-label` — mono uppercase technical eyebrow.
- `.focus-ring` — consistent visible focus ring (`ring-[3px] ring-ring/30`) for links/buttons.
- `.field-ring` — focus ring + border tint for inputs/textarea.
- `.card-hover` — gentle lift + shadow on interactive cards.
- `.skeleton` — shimmer placeholder (use via `Skeleton` in `components/ui/skeleton`).
- `.code-canvas` — monospace technical surface for raw data.

Motion: `animate-fade-in`, `animate-fade-up`, `animate-scale-in` (150–250 ms, ease-out). Subtle only — no decorative loops. `prefers-reduced-motion` is respected globally.

---

## Components

### Cards

`rounded-xl border border-border bg-card shadow-xs`. Do not stack heavy shadows or hover-elevate nested cards. Interactive cards add `card-hover`. Shared building blocks: `StatCard`, `EmptyState`, `Skeleton`.

### Buttons

Primary: orange fill, dark text (`bg-primary text-primary-foreground hover:bg-primary-hover`), `rounded-lg`, `shadow-xs`, `active:translate-y-px`. Variants: `primary | secondary | outline | ghost | destructive | success`. Sizes: `default | sm | lg | icon`.

### Inputs

`rounded-lg`, thin `border-input`, subtle shadow, hover border tint, orange focus ring via `field-ring`. Placeholder at `text-muted-foreground/70`.

### Badges

Tinted pill: `rounded-full border`, `~10%` background tint + matching text (`default | secondary | success | warning | destructive | outline`). Use mono for confidence/state badges.

### Tables

Minimal borders (`border-border/70`), mono uppercase headers, `hover:bg-muted/50` row hover. Numeric cells always `font-mono`.

### Navigation

Premium SaaS: sticky, `backdrop-blur` bar; active items show an orange icon + a 2px orange underline indicator; right-side theme toggle + auth actions; responsive slide-down mobile menu.

---

## Layout

Page canvas is centered at `max-w-7xl` in `app/layout.tsx` with `px-4 sm:px-6 lg:px-8` and generous vertical rhythm (`py-8 sm:py-10 lg:py-12`). Every page uses `space-y-8` and the shared `PageHeader`. Results and loading states animate in with `animate-fade-up`. Loading states use `Skeleton` shimmers instead of bare spinners.

---

## Interaction & Accessibility

- Hover lift: `card-hover` (`hover:-translate-y-0.5`, 200 ms ease-out).
- Focus: `focus-ring` on all interactive elements; never remove outlines.
- Disabled: `disabled:pointer-events-none disabled:opacity-50`.
- Contrast: dark text on the orange primary to satisfy WCAG; muted copy stays readable on all surfaces.
- ARIA: `role="alert"` on error banners, `role="progressbar"` with value on gauges, `aria-expanded`/`aria-controls` on the mobile menu, skip-to-content link in the layout.
- Touch targets: minimum `h-9`/`h-10` controls, padded mobile nav items.
