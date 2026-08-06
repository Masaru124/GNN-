const AUTH_KEY = "matscreen-auth";

export function isAuthenticated(): boolean {
  if (typeof window === "undefined") return false;
  return window.localStorage.getItem(AUTH_KEY) === "1";
}

export function setAuthenticated(value: boolean): void {
  if (typeof window === "undefined") return;
  if (value) {
    window.localStorage.setItem(AUTH_KEY, "1");
  } else {
    window.localStorage.removeItem(AUTH_KEY);
  }
}
