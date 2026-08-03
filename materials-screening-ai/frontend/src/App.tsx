import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './pages/Dashboard';
import { SinglePrediction } from './pages/SinglePrediction';
import { BatchScreening } from './pages/BatchScreening';
import { SearchMaterial } from './pages/SearchMaterial';
import { ComparePage } from './pages/ComparePage';
import { HistoryPage } from './pages/HistoryPage';

export const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<string>('dashboard');

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard onNavigate={setCurrentPage} />;
      case 'upload':
        return <SinglePrediction />;
      case 'batch':
        return <BatchScreening />;
      case 'search':
        return <SearchMaterial />;
      case 'compare':
        return <ComparePage />;
      case 'history':
        return <HistoryPage />;
      default:
        return <Dashboard onNavigate={setCurrentPage} />;
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-dark-900 text-slate-100 font-sans">
      <Navbar onNavigate={setCurrentPage} />

      <div className="flex flex-1">
        <Sidebar currentPage={currentPage} onNavigate={setCurrentPage} />

        <main className="flex-1 p-6 md:p-8 max-w-7xl mx-auto overflow-y-auto">
          {renderPage()}
        </main>
      </div>
    </div>
  );
};
