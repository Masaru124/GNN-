"use client";

import React, { useEffect, useRef } from "react";
import { Box, RefreshCw } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface Viewer3DProps {
  cifString?: string;
  formula?: string;
}

interface MolViewer {
  addModel(data: string, format: string): unknown;
  setStyle(selector: unknown, style: unknown): unknown;
  addUnitCell(): unknown;
  zoomTo(): unknown;
  render(): unknown;
  spin(axis: string, speed: number): unknown;
}

interface MolGlobal {
  createViewer(
    element: HTMLElement,
    options: { backgroundColor: string },
  ): MolViewer;
}

declare global {
  interface Window {
    $3Dmol?: MolGlobal;
  }
}

let threeDmolPromise: Promise<MolGlobal | null> | null = null;

function load3Dmol(): Promise<MolGlobal | null> {
  if (typeof window === 'undefined') return Promise.resolve(null);
  if (window.$3Dmol) return Promise.resolve(window.$3Dmol);
  if (!threeDmolPromise) {
    threeDmolPromise = new Promise<MolGlobal | null>((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://3Dmol.org/build/3Dmol-min.js';
      script.async = true;
      script.onload = () => resolve(window.$3Dmol ?? null);
      script.onerror = () => {
        threeDmolPromise = null;
        reject(new Error('Failed to load 3Dmol.js'));
      };
      document.body.appendChild(script);
    });
  }
  return threeDmolPromise;
}

export const Viewer3D: React.FC<Viewer3DProps> = ({ cifString, formula = 'Crystal Structure' }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<MolViewer | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !cifString) return;

    let cancelled = false;

    load3Dmol()
      .then((mol) => {
        if (!mol || cancelled || !containerRef.current) return;

        containerRef.current.innerHTML = '';
        const viewer = mol.createViewer(containerRef.current, {
          backgroundColor: '#0f172a',
        });
        viewerRef.current = viewer;

        viewer.addModel(cifString, 'cif');
        viewer.setStyle({}, { sphere: { scale: 0.28 }, stick: { radius: 0.14 } });
        viewer.addUnitCell();
        viewer.zoomTo();
        viewer.render();
        viewer.spin('y', 0.5);
      })
      .catch((err) => {
        console.error('3Dmol load failed', err);
      });

    return () => {
      cancelled = true;
    };
  }, [cifString]);

  const handleResetView = () => {
    if (viewerRef.current) {
      viewerRef.current.zoomTo();
      viewerRef.current.render();
    }
  };

  return (
    <Card className="relative overflow-hidden p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Box className="h-4 w-4 text-accent" />
          <h3 className="font-display text-lg font-bold tracking-tight text-foreground">
            3D Crystal Lattice Viewer
          </h3>
        </div>
        <Button
          onClick={handleResetView}
          variant="ghost"
          size="icon"
          title="Reset View"
          aria-label="Reset View"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>
      </div>

      <div
        ref={containerRef}
        className="relative h-[320px] w-full overflow-hidden rounded-xl border border-border bg-[#0f172a]"
      >
        {!cifString && (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
            No CIF structure loaded
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-col gap-1 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
        <span>
          Structure: <strong className="font-medium text-foreground">{formula}</strong>
        </span>
        <span className="text-[11px]">Click & Drag to Rotate | Scroll to Zoom</span>
      </div>
    </Card>
  );
};
