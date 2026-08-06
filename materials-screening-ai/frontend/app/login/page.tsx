"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Atom, Lock, User as UserIcon, LogIn } from "lucide-react";
import { setAuthenticated } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";

export default function Page() {
  const router = useRouter();
  const [username, setUsername] = useState("researcher");
  const [password, setPassword] = useState("password");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setAuthenticated(true);
    router.push("/");
  };

  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
      <Card className="w-full max-w-md p-2">
        <CardHeader className="p-8 text-center">
          <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-primary-muted text-primary">
            <Atom className="h-7 w-7" />
          </div>
          <CardTitle className="text-2xl font-bold tracking-tight">Welcome back</CardTitle>
          <CardDescription>Sign in to access your materials screening workspace</CardDescription>
        </CardHeader>
        <CardContent className="px-8 pb-8">
          <form onSubmit={handleSubmit} className="space-y-5">
            <div className="space-y-1.5">
              <label htmlFor="username" className="text-[13px] font-medium text-foreground">
                Username
              </label>
              <div className="relative">
                <UserIcon className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label htmlFor="password" className="text-[13px] font-medium text-foreground">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>

            <Button type="submit" className="h-11 w-full">
              <LogIn className="h-4 w-4" />
              Sign In to Platform
            </Button>
          </form>

          <p className="mt-5 text-center text-xs leading-relaxed text-muted-foreground">
            Demo credentials pre-filled. Click{" "}
            <span className="font-medium text-foreground">Sign In</span> to proceed.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
