"use client";

import React, { useState } from "react";
import { UploadCloud, FileText, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";

interface UploadBoxProps {
  onFileSelected: (file: File) => void;
  onSampleSelected: (cifText: string, formula: string) => void;
  isLoading?: boolean;
}

const SAMPLE_CIFS = {
  TiO2: `data_TiO2
_symmetry_space_group_name_H-M   'P 42/m n m'
_cell_length_a   4.593
_cell_length_b   4.593
_cell_length_c   2.959
_cell_angle_alpha   90.0
_cell_angle_beta    90.0
_cell_angle_gamma   90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ti1 Ti 0.0000 0.0000 0.0000
Ti2 Ti 0.5000 0.5000 0.5000
O1 O 0.3050 0.3050 0.0000
O2 O 0.6950 0.6950 0.0000
O3 O 0.8050 0.1950 0.5000
O4 O 0.1950 0.8050 0.5000`,

  LiFePO4: `data_LiFePO4
_symmetry_space_group_name_H-M   'P n m a'
_cell_length_a   10.330
_cell_length_b   6.010
_cell_length_c   4.690
_cell_angle_alpha   90.0
_cell_angle_beta    90.0
_cell_angle_gamma   90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 Li 0.0000 0.0000 0.0000
Fe1 Fe 0.2822 0.2500 0.9747
P1 P 0.0949 0.2500 0.4183
O1 O 0.0968 0.2500 0.7424
O2 O 0.4571 0.2500 0.2060
O3 O 0.1658 0.0466 0.2847`,

  BaTiO3: `data_BaTiO3
_symmetry_space_group_name_H-M   'P m -3 m'
_cell_length_a   4.000
_cell_length_b   4.000
_cell_length_c   4.000
_cell_angle_alpha   90.0
_cell_angle_beta    90.0
_cell_angle_gamma   90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Ba1 Ba 0.0000 0.0000 0.0000
Ti1 Ti 0.5000 0.5000 0.5000
O1 O 0.5000 0.5000 0.0000
O2 O 0.5000 0.0000 0.5000
O3 O 0.0000 0.5000 0.5000`,

  MoS2: `data_MoS2
_symmetry_space_group_name_H-M   'P 63/m m c'
_cell_length_a   3.160
_cell_length_b   3.160
_cell_length_c   12.290
_cell_angle_alpha   90.0
_cell_angle_beta    90.0
_cell_angle_gamma   120.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Mo1 Mo 0.3333 0.6667 0.2500
S1 S 0.3333 0.6667 0.6210
S2 S 0.3333 0.6667 0.8790`,
};

export const UploadBox: React.FC<UploadBoxProps> = ({
  onFileSelected,
  onSampleSelected,
  isLoading,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFileName(file.name);
      onFileSelected(file);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFileName(file.name);
      onFileSelected(file);
    }
  };

  return (
    <div className="space-y-5">
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={cn(
          "group relative cursor-pointer overflow-hidden rounded-xl border-2 border-dashed transition-all duration-200",
          dragActive
            ? "border-primary bg-primary-muted"
            : selectedFileName
              ? "border-success/40 bg-success-muted/40"
              : "border-border bg-card hover:border-primary/40 hover:bg-muted/40",
        )}
      >
        <input
          type="file"
          accept=".cif,.poscar,.txt"
          onChange={handleChange}
          className="absolute inset-0 z-10 h-full w-full cursor-pointer opacity-0"
          disabled={isLoading}
          aria-label="Upload a CIF structure file"
        />

        <div className="flex flex-col items-center justify-center gap-4 px-6 py-12 text-center sm:py-14">
          <div
            className={cn(
              "flex h-14 w-14 items-center justify-center rounded-2xl border transition-all duration-200 group-hover:scale-105",
              selectedFileName
                ? "border-success/25 bg-success-muted text-success"
                : dragActive
                  ? "border-primary/30 bg-primary-muted text-primary"
                  : "border-border bg-muted/70 text-primary",
            )}
          >
            {selectedFileName ? (
              <CheckCircle2 className="h-7 w-7" />
            ) : (
              <UploadCloud className="h-7 w-7" />
            )}
          </div>

          <div>
            <p className="text-[15px] font-semibold tracking-tight text-foreground">
              {selectedFileName
                ? `Ready: ${selectedFileName}`
                : "Drag & drop a CIF structure"}
            </p>
            <p className="mt-1 text-[13px] text-muted-foreground">
              {selectedFileName
                ? "File selected — inference will run automatically"
                : "or click to browse · .cif, .poscar supported"}
            </p>
          </div>
        </div>
      </div>

      <Card className="space-y-3 p-5">
        <div className="flex items-center justify-between gap-2">
          <p className="micro-label">Or load a sample structure</p>
          <FileText className="h-4 w-4 text-muted-foreground" />
        </div>
        <div className="flex flex-wrap gap-2">
          {Object.entries(SAMPLE_CIFS).map(([formula, cifText]) => (
            <button
              key={formula}
              onClick={() => {
                setSelectedFileName(`${formula}.cif`);
                onSampleSelected(cifText, formula);
              }}
              disabled={isLoading}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-muted/50 px-3 py-1.5 font-mono text-xs font-medium text-foreground transition-colors duration-150 hover:border-primary/30 hover:bg-primary-muted hover:text-primary focus-ring disabled:pointer-events-none disabled:opacity-50"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-primary/60" />
              {formula}
            </button>
          ))}
        </div>
      </Card>
    </div>
  );
};
