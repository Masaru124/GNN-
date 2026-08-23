import { Sidebar } from "@/components/layout/Sidebar";

export default function AppLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      <Sidebar />

      <main
        id="main-content"
        className="min-w-0 flex-1 px-4 py-8 sm:px-6 lg:px-8"
      >
        {children}
      </main>
    </div>
  );
}
