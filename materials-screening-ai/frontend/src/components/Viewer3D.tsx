import React, { useEffect, useRef } from 'react';
import { Box, RefreshCw } from 'lucide-react';

interface Viewer3DProps {
  cifString?: string;
  formula?: string;
}

declare global {
  interface Window {
    $3Dmol: any;
  }
}

export const Viewer3D: React.FC<Viewer3DProps> = ({ cifString, formula = 'Crystal Structure' }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);

  useEffect(() => {
    if (!containerRef.current || !cifString) return;

    // Check if 3Dmol is available on window
    if (typeof window !== 'undefined' && window.$3Dmol) {
      containerRef.current.innerHTML = '';
      const viewer = window.$3Dmol.createViewer(containerRef.current, {
        backgroundColor: '#0b0f17',
      });
      viewerRef.current = viewer;

      viewer.addModel(cifString, 'cif');
      viewer.setStyle({}, { sphere: { scale: 0.28 }, stick: { radius: 0.14 } });
      viewer.addUnitCell();
      viewer.zoomTo();
      viewer.render();
      viewer.spin('y', 0.5);
    }
  }, [cifString]);

  const handleResetView = () => {
    if (viewerRef.current) {
      viewerRef.current.zoomTo();
      viewerRef.current.render();
    }
  };

  return (
    <div className="relative glass-card rounded-2xl p-4 border border-slate-800 overflow-hidden group">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Box className="w-4 h-4 text-blue-400" />
          <h3 className="text-sm font-semibold text-slate-200">3D Crystal Lattice Viewer</h3>
        </div>
        <button
          onClick={handleResetView}
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
          title="Reset View"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>

      <div
        ref={containerRef}
        className="w-full h-[320px] rounded-xl overflow-hidden bg-dark-900 border border-slate-800/60 relative"
      >
        {!cifString && (
          <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-xs">
            No CIF structure loaded
          </div>
        )}
      </div>

      <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
        <span>Structure: <strong className="text-slate-200">{formula}</strong></span>
        <span className="text-[11px]">Click & Drag to Rotate | Scroll to Zoom</span>
      </div>
    </div>
  );
};
