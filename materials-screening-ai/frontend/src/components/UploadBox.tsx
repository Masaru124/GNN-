import React, { useState } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertCircle } from 'lucide-react';

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
S2 S 0.3333 0.6667 0.8790`
};

export const UploadBox: React.FC<UploadBoxProps> = ({ onFileSelected, onSampleSelected, isLoading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
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
    <div className="space-y-4">
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`glass-card rounded-2xl p-8 border-2 border-dashed transition-all text-center cursor-pointer relative overflow-hidden group ${
          dragActive ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700/80 hover:border-slate-600 bg-slate-900/40'
        }`}
      >
        <input
          type="file"
          accept=".cif,.poscar,.txt"
          onChange={handleChange}
          className="absolute inset-0 opacity-0 cursor-pointer z-10"
          disabled={isLoading}
        />

        <div className="flex flex-col items-center justify-center space-y-3">
          <div className="w-14 h-14 rounded-2xl bg-blue-600/15 border border-blue-500/30 flex items-center justify-center text-blue-400 group-hover:scale-110 transition-transform">
            <UploadCloud className="w-7 h-7" />
          </div>

          <div>
            <p className="text-base font-semibold text-slate-200">
              {selectedFileName ? `Selected: ${selectedFileName}` : 'Drag & Drop CIF File Here'}
            </p>
            <p className="text-xs text-slate-400 mt-1">Supports Crystallographic Information Files (.cif) & POSCAR format</p>
          </div>

          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-lg shadow-blue-600/25 transition-all">
            Browse Computer
          </div>
        </div>
      </div>

      {/* Quick Sample Presets */}
      <div className="glass-card rounded-2xl p-4 border border-slate-800 space-y-2">
        <p className="text-xs font-medium text-slate-400">Or load a sample crystal structure:</p>
        <div className="flex flex-wrap gap-2">
          {Object.entries(SAMPLE_CIFS).map(([formula, cifText]) => (
            <button
              key={formula}
              onClick={() => {
                setSelectedFileName(`${formula}.cif`);
                onSampleSelected(cifText, formula);
              }}
              disabled={isLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700/80 text-slate-300 text-xs font-medium border border-slate-700/50 transition-colors"
            >
              <FileText className="w-3.5 h-3.5 text-cyan-400" />
              <span>{formula}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
