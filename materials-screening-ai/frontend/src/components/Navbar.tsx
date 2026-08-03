import React from 'react';
import { Atom, ShieldCheck, Activity, Search, UploadCloud } from 'lucide-react';

interface NavbarProps {
  onNavigate: (page: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ onNavigate }) => {
  return (
    <header className="sticky top-0 z-50 bg-dark-900/80 backdrop-blur-md border-b border-slate-800/80 px-6 py-3.5 flex items-center justify-between">
      {/* Brand Logo */}
      <div 
        onClick={() => onNavigate('dashboard')}
        className="flex items-center gap-3 cursor-pointer group"
      >
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 via-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-blue-500/20 group-hover:scale-105 transition-transform">
          <Atom className="w-6 h-6 text-white animate-spin-slow" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold font-sans tracking-tight bg-gradient-to-r from-white via-slate-200 to-blue-400 bg-clip-text text-transparent">
              MatScreen <span className="text-blue-500">AI</span>
            </h1>
            <span className="text-[10px] uppercase tracking-wider font-semibold px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
              v1.0 A7
            </span>
          </div>
          <p className="text-xs text-slate-400 hidden sm:block">Materials Property Screening & Uncertainty Quantification</p>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => onNavigate('search')}
          className="flex items-center gap-2 text-xs font-medium px-3.5 py-2 rounded-lg bg-slate-800/60 hover:bg-slate-700/60 text-slate-300 border border-slate-700/50 transition-colors"
        >
          <Search className="w-4 h-4 text-cyan-400" />
          <span>Search Materials</span>
        </button>

        <button
          onClick={() => onNavigate('upload')}
          className="flex items-center gap-2 text-xs font-medium px-3.5 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white shadow-md shadow-blue-600/25 transition-all"
        >
          <UploadCloud className="w-4 h-4" />
          <span>Predict CIF</span>
        </button>

        {/* System Status Pill */}
        <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>A7 GNN Online</span>
        </div>
      </div>
    </header>
  );
};
