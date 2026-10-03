export interface MaterialSite {
  index: number;
  element: string;
  label: string;
  cartesian: [number, number, number];
  fractional: [number, number, number];
}

export interface LatticeParams {
  a: number;
  b: number;
  c: number;
  alpha: number;
  beta: number;
  gamma: number;
  matrix: number[][];
}

export interface MaterialInfo {
  material_id?: string;
  filename: string;
  formula: string;
  formula_pretty: string;
  num_atoms: number;
  density_g_cm3: number;
  volume_A3: number;
  elements: string[];
  lattice?: LatticeParams;
  sites?: MaterialSite[];
  cif_string?: string;
}

export interface ScaleAttention {
  "4A": number;
  "6A": number;
  "8A": number;
  raw_weights?: number[];
}

export interface PredictionResult {
  predicted_formation_energy_per_atom_eV: number;
  evidential_std_eV: number;
  aleatoric_std_eV: number;
  epistemic_std_eV: number;
  total_variance?: number;
  conformal_90_interval_eV: [number, number];
  conformal_width_eV: number;
  raw_der_95_interval_eV?: [number, number];
  confidence: "High" | "Medium" | "Low";
  confidence_score_pct: number;
  risk_level: string;
  recommendation: string;
  badge_color: "green" | "yellow" | "red";
  scale_attention: ScaleAttention;
}

export interface ConformalCoverageGate {
  chemistry_class: string;
  n_cal: number;
  min_n_cal: number;
  status: "calibrated" | "under_calibrated" | "no_calibration";
  claim_scope: string;
}

export interface PredictResponsePayload {
  material_info: MaterialInfo;
  conformal_coverage_gate?: ConformalCoverageGate;
  prediction: PredictionResult;
}

export interface CandidateItem extends PredictionResult {
  rank?: number;
  filename: string;
  formula: string;
  formula_pretty: string;
  num_atoms: number;
  density_g_cm3: number;
  volume_A3: number;
  cif_string: string;
}

export interface BatchScreenResponse {
  job_id: string;
  total_files: number;
  screened_count: number;
  high_confidence_count: number;
  runtime_seconds: number;
  candidates: CandidateItem[];
}

export interface SearchResultItem {
  material_id: string;
  formula: string;
  formula_pretty: string;
  name: string;
  crystal_system: string;
  spacegroup: string;
  volume_A3: number;
  density_g_cm3: number;
}

export interface HistoryLog {
  id: number;
  created_at: string;
  filename: string;
  formula: string;
  predicted_energy_eV: number;
  evidential_std_eV: number;
  confidence: string;
}

export interface ScaleEnergyResult {
  predicted_formation_energy_per_atom_eV: number;
  evidential_std_eV: number;
}

export interface ModelComparison {
  comparison: {
    multi_scale: ScaleEnergyResult;
    single_scale: ScaleEnergyResult;
    difference_eV: number;
    explanation: string;
  };
}
