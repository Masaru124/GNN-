"use client";

import React, { useEffect, useRef, useState } from "react";
import { Box, RefreshCw, Play, Pause, RotateCw, Layers, Tag } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface Viewer3DProps {
  cifString?: string;
  formula?: string;
  trajectoryFrames?: string[];
}

interface MolViewer {
  addModel(data: string, format: string, options?: any): unknown;
  getModel(): any;
  setStyle(selector: unknown, style: unknown): unknown;
  addUnitCell(model?: any): unknown;
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

const VIEWPORT_BG = "#ffffff";
const ELEMENT_NAMES: Record<string, { name: string; color: string }> = {
  H: { name: "Hydrogen", color: "#FFFFFF" },
  Li: { name: "Lithium", color: "#CC80FF" },
  Na: { name: "Sodium", color: "#AB5CF2" },
  K: { name: "Potassium", color: "#8F40D4" },
  Rb: { name: "Rubidium", color: "#702EB0" },
  Cs: { name: "Cesium", color: "#57178F" },
  Mg: { name: "Magnesium", color: "#8AF000" },
  Ca: { name: "Calcium", color: "#3DFF00" },
  Sr: { name: "Strontium", color: "#00D900" },
  Ba: { name: "Barium", color: "#00A500" },
  Ti: { name: "Titanium", color: "#BFA6A6" },
  V: { name: "Vanadium", color: "#A6A6AB" },
  Cr: { name: "Chromium", color: "#8A99C7" },
  Mn: { name: "Manganese", color: "#9C7AC7" },
  Fe: { name: "Iron", color: "#E06633" },
  Co: { name: "Cobalt", color: "#F090A0" },
  Ni: { name: "Nickel", color: "#50D050" },
  Cu: { name: "Copper", color: "#C88033" },
  Zn: { name: "Zinc", color: "#7D80B0" },
  Zr: { name: "Zirconium", color: "#94E0E0" },
  Nb: { name: "Niobium", color: "#73C2C9" },
  Mo: { name: "Molybdenum", color: "#54B5B5" },
  Sn: { name: "Tin", color: "#6E7B8B" },
  Ge: { name: "Germanium", color: "#668F8F" },
  Bi: { name: "Bismuth", color: "#9E4FB3" },
  O: { name: "Oxygen", color: "#FF0D0D" },
  F: { name: "Fluorine", color: "#90E050" },
  Cl: { name: "Chlorine", color: "#1FF01F" },
  Br: { name: "Bromine", color: "#A62929" },
  I: { name: "Iodine", color: "#9400D3" },
  P: { name: "Phosphorus", color: "#FF8000" },
  S: { name: "Sulfur", color: "#FFFF30" },
};

let threeDmolPromise: Promise<MolGlobal | null> | null = null;

function load3Dmol(): Promise<MolGlobal | null> {
  if (typeof window === "undefined") return Promise.resolve(null);
  if (window.$3Dmol) return Promise.resolve(window.$3Dmol);
  if (!threeDmolPromise) {
    threeDmolPromise = new Promise<MolGlobal | null>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://3Dmol.org/build/3Dmol-min.js";
      script.async = true;
      script.onload = () => resolve(window.$3Dmol ?? null);
      script.onerror = () => {
        threeDmolPromise = null;
        reject(new Error("Failed to load 3Dmol.js"));
      };
      document.body.appendChild(script);
    });
  }
  return threeDmolPromise;
}

export const Viewer3D: React.FC<Viewer3DProps> = ({
  cifString,
  formula = "Crystal Structure",
  trajectoryFrames = [],
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<MolViewer | null>(null);

  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isSpinning, setIsSpinning] = useState(false);

  const hasFrames = trajectoryFrames && trajectoryFrames.length > 0;
  const activeCif = hasFrames ? trajectoryFrames[frameIndex] : cifString;

  // Extract lattice parameters from CIF text via regex
  const parseLatticeParams = (cif?: string) => {
    if (!cif) return null;
    const aMatch = cif.match(/_cell_length_a\s+([\d.]+)/);
    const bMatch = cif.match(/_cell_length_b\s+([\d.]+)/);
    const cMatch = cif.match(/_cell_length_c\s+([\d.]+)/);
    const alphaMatch = cif.match(/_cell_angle_alpha\s+([\d.]+)/);
    const betaMatch = cif.match(/_cell_angle_beta\s+([\d.]+)/);
    const gammaMatch = cif.match(/_cell_angle_gamma\s+([\d.]+)/);

    return {
      a: aMatch ? aMatch[1] : "N/A",
      b: bMatch ? bMatch[1] : "N/A",
      c: cMatch ? cMatch[1] : "N/A",
      alpha: alphaMatch ? alphaMatch[1] : "90",
      beta: betaMatch ? betaMatch[1] : "90",
      gamma: gammaMatch ? gammaMatch[1] : "90",
    };
  };

  // Extract unique elements present in structure for color legend
  const extractElementLegend = (cif?: string) => {
    if (!cif) return [];
    const found = new Set<string>();
    
    // Regex for CIF atom_site lines
    const symbolMatches = cif.matchAll(/([A-Z][a-z]?)\s+[A-Z][a-z]?\d*\s+\d/g);
    for (const m of symbolMatches) {
      if (ELEMENT_NAMES[m[1]]) found.add(m[1]);
    }

    if (found.size === 0) {
      // Fallback matching from formula string
      for (const el in ELEMENT_NAMES) {
        if (formula.includes(el) || (cif && cif.includes(el))) {
          found.add(el);
        }
      }
    }

    return Array.from(found).map((el) => ({
      symbol: el,
      name: ELEMENT_NAMES[el]?.name || el,
      color: ELEMENT_NAMES[el]?.color || "#3b82f6",
    }));
  };

  const latticeParams = parseLatticeParams(activeCif);
  const elementLegend = extractElementLegend(activeCif);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !activeCif) return;

    let cancelled = false;

    load3Dmol()
      .then((mol) => {
        if (!mol || cancelled || !containerRef.current) return;

        containerRef.current.innerHTML = "";
        const isDark = typeof window !== "undefined" && (document.documentElement.classList.contains("dark") || window.matchMedia("(prefers-color-scheme: dark)").matches);
        const bgColor = isDark ? "#09090b" : "#ffffff";
        const viewer = mol.createViewer(containerRef.current, {
          backgroundColor: bgColor,
        });
        viewerRef.current = viewer;

        // Add 3D CIF Model with CPK Atomic Ball & Stick Coordination
        viewer.addModel(activeCif, "cif");
        viewer.setStyle(
          {},
          {
            sphere: { scale: 0.30, colorscheme: "Jmol" },
            stick: { radius: 0.15, colorscheme: "Jmol" },
          }
        );

        // Add unit cell bounding box
        viewer.addUnitCell();
        viewer.zoomTo();
        viewer.render();

        if (isSpinning) {
          viewer.spin("y", 0.8);
        }
      })
      .catch((err) => {
        console.error("3Dmol load failed", err);
      });

    return () => {
      cancelled = true;
    };
  }, [activeCif]);

  const toggleSpin = () => {
    if (viewerRef.current) {
      if (isSpinning) {
        viewerRef.current.spin("y", 0);
        setIsSpinning(false);
      } else {
        viewerRef.current.spin("y", 0.8);
        setIsSpinning(true);
      }
    }
  };

  useEffect(() => {
    if (!isPlaying || !hasFrames) return;

    const interval = setInterval(() => {
      setFrameIndex((prev) => (prev + 1) % trajectoryFrames.length);
    }, 250);

    return () => clearInterval(interval);
  }, [isPlaying, hasFrames, trajectoryFrames]);

  const handleResetView = () => {
    if (viewerRef.current) {
      viewerRef.current.zoomTo();
      viewerRef.current.render();
    }
  };

  return (
    <Card className="overflow-hidden p-5 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Box className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-bold tracking-tight text-foreground">
            3D Crystal Lattice Viewer — {formula}
          </h3>
        </div>

        <div className="flex items-center gap-1.5">
          <Button
            onClick={toggleSpin}
            variant="outline"
            size="sm"
            className={`h-8 px-2 text-xs gap-1 font-semibold ${isSpinning ? "border-primary text-primary bg-primary/10" : ""}`}
          >
            <RotateCw
              className={`h-3.5 w-3.5 ${isSpinning ? "animate-spin" : ""}`}
            />
            {isSpinning ? "Spinning" : "Auto Spin"}
          </Button>

          <Button
            onClick={handleResetView}
            variant="ghost"
            size="sm"
            title="Reset view"
            className="h-8 px-2 text-xs"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      <div
        ref={containerRef}
        className="relative h-[320px] w-full overflow-hidden rounded-xl border border-border bg-[#12141a]"
        style={{ background: VIEWPORT_BG }}
      >
        {!activeCif && (
          <div className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
            No CIF structure loaded
          </div>
        )}

        {latticeParams && (
          <div className="absolute top-3 left-3 bg-white backdrop-blur-md border border-white/10 rounded-lg p-2.5 text-[10px] font-mono text-white/90 space-y-1 z-10 pointer-events-none">
            <div className="font-bold text-primary flex items-center gap-1">
              <Layers className="h-3 w-3" /> Unit Cell Geometry ({formula})
            </div>
            <div>
              a:{" "}
              <span className="text-emerald-400 font-bold">
                {latticeParams.a} Å
              </span>{" "}
              | b:{" "}
              <span className="text-emerald-400 font-bold">
                {latticeParams.b} Å
              </span>{" "}
              | c:{" "}
              <span className="text-emerald-400 font-bold">
                {latticeParams.c} Å
              </span>
            </div>
            <div>
              α: {latticeParams.alpha}° | β: {latticeParams.beta}° | γ:{" "}
              {latticeParams.gamma}°
            </div>
          </div>
        )}

        {elementLegend.length > 0 && (
          <div className="absolute bottom-3 left-3 bg-white backdrop-blur-md border border-white/10 rounded-lg px-3 py-2 text-[10px] text-white/90 z-10 space-y-1">
            <span className="font-bold text-muted-foreground block text-[9px] uppercase tracking-wider flex items-center gap-1">
              <Tag className="h-2.5 w-2.5 text-primary" /> Elemental CPK Color
              Legend:
            </span>
            <div className="flex flex-wrap gap-2.5 items-center font-mono">
              {elementLegend.map((item) => (
                <div key={item.symbol} className="flex items-center gap-1.5">
                  <span
                    className="h-3 w-3 rounded-full border border-white/30 shadow-sm shrink-0"
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="font-bold text-white">{item.symbol}</span>
                  <span className="text-muted-foreground text-[9px]">
                    ({item.name})
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Trajectory Scrubber Controls if MD frames exist */}
      {hasFrames && (
        <div className="flex items-center justify-between gap-3 pt-1 text-xs">
          <Button
            size="sm"
            variant="outline"
            onClick={() => setIsPlaying(!isPlaying)}
            className="h-8 px-3 gap-1.5 font-bold"
          >
            {isPlaying ? (
              <Pause className="h-3.5 w-3.5 text-amber-500" />
            ) : (
              <Play className="h-3.5 w-3.5 text-emerald-500" />
            )}
            {isPlaying ? "Pause MD" : "Play MD Trajectory"}
          </Button>

          <div className="flex items-center gap-2 flex-1">
            <input
              type="range"
              min={0}
              max={trajectoryFrames.length - 1}
              value={frameIndex}
              onChange={(e) => {
                setIsPlaying(false);
                setFrameIndex(Number(e.target.value));
              }}
              className="flex-1 accent-primary"
            />
            <span className="font-mono text-[11px] text-muted-foreground min-w-[60px] text-right font-bold">
              Frame {frameIndex + 1}/{trajectoryFrames.length}
            </span>
          </div>
        </div>
      )}
    </Card>
  );
};
