import React from 'react';
import { 
  LayoutDashboard, 
  Upload, 
  Layers, 
  Search, 
  GitCompare, 
  History, 
  FileSpreadsheet,
  Settings
} from 'lucide-react';

interface SidebarProps {
  currentPage: string;
  onNavigate: (page: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentPage, onNavigate }) => {
  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'upload', label: 'Single Prediction', icon: Upload },
    { id: 'batch', label: 'Batch Screening', icon: Layers },
    { id: 'search', label: 'Search Database', icon: Search },
    { id: 'compare', label: 'Model Comparison', icon: GitCompare },
    { id: 'history', label: 'Prediction History', icon: History },
  ];

  return (
    <aside className="w-64 bg-dark-900/90 border-r border-slate-800/80 p-4 flex flex-col justify-between shrink-0 min-h-[calc(100vh-61px)]">
      <div className="space-y-6">
        <div>
          <p className="px-3 text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-3">
            Core Modules
          </p>
          <nav className="space-y-1">
            {menuItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentPage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onNavigate(item.id)}
                  className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-blue-600/15 text-blue-400 border border-blue-500/30 shadow-md shadow-blue-500/5'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-blue-400' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Model Spec Badge Footer */}
      <div className="p-3.5 rounded-xl glass-card border border-slate-800 text-xs text-slate-400 space-y-1.5">
        <div className="flex items-center justify-between text-slate-300 font-medium">
          <span>A7 Multi-Scale GNN</span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400">Ready</span>
        </div>
        <p className="text-[11px] text-slate-400">Radii: 4.0Å | 6.0Å | 8.0Å</p>
        <p className="text-[11px] text-slate-400">Conformal Coverage: 90% ($q = 0.4954$)</p>
      </div>
    </aside>
  );
};
