import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getErrorMessage(
  err: unknown,
  fallback = "Request failed.",
): string {
  if (typeof err === "object" && err !== null) {
    const e = err as {
      response?: { data?: { detail?: unknown } };
      message?: unknown;
    };
    if (typeof e.response?.data?.detail === "string" && e.response.data.detail) {
      return e.response.data.detail;
    }
    if (typeof e.message === "string" && e.message) return e.message;
  }
  if (typeof err === "string") return err;
  return fallback;
}
