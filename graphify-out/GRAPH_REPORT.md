# Graph Report - GNN  (2026-09-24)

## Corpus Check
- Large corpus: 218 files · ~2,500,764 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder.

## Summary
- 1400 nodes · 2658 edges · 128 communities (72 shown, 42 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 145 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- NNConv Graph Encoder
- Prediction & Batch Screening API
- Conformal UQ Calibration
- MultiScale Crystal Dataset
- MultiScale Crystal Dataset
- Weisfeiler-Lehman Expressivity Analysis
- Prediction & Batch Screening API
- Discovery Pipeline & Active Learning
- Dual Evidential Regression Head
- MultiScale Crystal Dataset
- Prediction & Batch Screening API
- Evaluation & Uncertainty Metrics
- Prediction & Batch Screening API
- Multi-Scale Attention Fusion
- Discovery Pipeline & Active Learning
- Virtual Lab Physics Simulation
- Crystal Gnn Scripts
- Prediction & Batch Screening API
- NNConv Graph Encoder
- Virtual Lab Physics Simulation
- Discovery Pipeline & Active Learning
- Prediction & Batch Screening API
- NNConv Graph Encoder
- Dual Evidential Regression Head
- Prediction & Batch Screening API
- MultiScale Crystal Dataset
- Prediction & Batch Screening API
- Weisfeiler-Lehman Expressivity Analysis
- Crystal Gnn Scripts
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- MultiScale Crystal Dataset
- MultiScale Crystal Dataset
- MultiScale Crystal Dataset
- Evaluation & Uncertainty Metrics
- Evaluation & Uncertainty Metrics
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Conformal UQ Calibration
- Discovery Pipeline & Active Learning
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Multi-Scale Attention Fusion
- Discovery Pipeline & Active Learning
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Dual Evidential Regression Head
- Prediction & Batch Screening API
- Virtual Lab Physics Simulation
- Virtual Lab Physics Simulation
- Dual Evidential Regression Head
- preprocessing Module
- MultiScale Crystal Dataset
- MultiScale Crystal Dataset
- Multi-Scale Attention Fusion
- run_managed Module
- Prediction & Batch Screening API
- MultiScale Crystal Dataset
- Conformal UQ Calibration
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- MultiScale Crystal Dataset
- evaluate_alignn_soap_loco Module
- run_der_lambda_sweep Module
- run_external_benchmark Module
- train Module
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- _analyze_collisions Module
- _run_external_models Module
- Evaluation & Uncertainty Metrics
- audit_collision_csv Module
- consolidate_cache Module
- _evaluate_chgnet Module
- export_colliding_cifs Module
- query_mp Module
- run_ablations Module
- _spotcheck_chgnet_5pairs Module
- _web_query_mp Module
- Prediction & Batch Screening API
- __init__ Module
- Frontend Dependencies & Config
- __init__ Module
- Frontend Dependencies & Config
- README Module
- README Module
- Evaluation & Uncertainty Metrics
- summary Module
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- ablation_A1 Module
- ablation_A2 Module
- ablation_A3 Module
- ablation_A4 Module
- ablation_A5 Module
- ablation_A6 Module
- ablation_A7 Module
- default Module
- requirements Module
- index Module
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- Prediction & Batch Screening API
- pyproject Module

## God Nodes (most connected - your core abstractions)
1. `cn()` - 42 edges
2. `MultiScaleGNN` - 39 edges
3. `MultiScaleDataset` - 30 edges
4. `load_data()` - 29 edges
5. `_build_model()` - 29 edges
6. `MultiScaleGNN` - 27 edges
7. `GNNPredictorService` - 26 edges
8. `SplitConformalPredictor` - 22 edges
9. `NNConvEncoder` - 21 edges
10. `SingleScaleGNN` - 21 edges

## Surprising Connections (you probably didn't know these)
- `GNNPredictorService` --uses--> `MultiScaleGNN`  [INFERRED]
  materials-screening-ai/backend/app/services/predictor.py → crystal_gnn/crystal_gnn/models/ms_gnn.py
- `GNNPredictorService` --uses--> `SingleScaleGNN`  [INFERRED]
  materials-screening-ai/backend/app/services/predictor.py → crystal_gnn/crystal_gnn/models/ms_gnn.py
- `main()` --uses--> `PhysicsValidationLayer`  [INFERRED]
  crystal_gnn/scripts/run_retroactive_177_ensemble.py → materials-screening-ai/backend/app/services/physics_validation.py
- `load_dataset_subsampled()` --calls--> `load_data()`  [INFERRED]
  crystal_gnn/collision_scan.py → crystal_gnn/scripts/train.py
- `main()` --uses--> `MultiScaleDataset`  [INFERRED]
  crystal_gnn/scripts/analyze_quartiles.py → crystal_gnn/crystal_gnn/data/dataset.py

## Import Cycles
- None detected.

## Communities (128 total, 42 thin omitted)

### Community 0 - "NNConv Graph Encoder"
Cohesion: 0.06
Nodes (40): NNConvEncoder, Tensor, NNConv encoder for single-radius crystal graphs., Stacked NNConv encoder with always-on MC dropout., Encode PyG batch into crystal-level embeddings [B, hidden_dim]., Split conformal prediction for uncertainty-calibrated intervals., Uncertainty estimation modules., mc_inference() (+32 more)

### Community 1 - "Prediction & Batch Screening API"
Cohesion: 0.04
Nodes (45): axios, clsx, eslint, eslint-config-next, lucide-react, dependencies, axios, clsx (+37 more)

### Community 2 - "Conformal UQ Calibration"
Cohesion: 0.07
Nodes (29): ndarray, Fit nonconformity quantiles on calibration data grouped by uncertainty…, Calibrate and evaluate split-conformal prediction intervals., Return conformal intervals using local quartile q_hats., Compute empirical coverage and interval statistics., Fit nonconformity quantile on calibration data., Return conformal intervals for supplied predictions/uncertainties., Compute empirical coverage and interval statistics. (+21 more)

### Community 3 - "MultiScale Crystal Dataset"
Cohesion: 0.10
Nodes (38): download_mp_dataset(), _load_existing(), _parse_crystal_system(), _parse_spacegroup_number(), Any, Path, Materials Project dataset download and filtering., Download and filter MP structures, writing gzipped structures and CSV labels. (+30 more)

### Community 4 - "MultiScale Crystal Dataset"
Cohesion: 0.09
Nodes (26): MultiScaleDataset, Batch, Data, Dataset, Path, Structure, Tensor, PyTorch dataset for multi-scale crystal graphs. (+18 more)

### Community 5 - "Weisfeiler-Lehman Expressivity Analysis"
Cohesion: 0.13
Nodes (28): Expressivity analysis tools., compute_crystal_graph(), find_resolved_pairs(), find_wl_collisions(), _h(), main(), Any, Structure (+20 more)

### Community 6 - "Prediction & Batch Screening API"
Cohesion: 0.14
Nodes (23): Page(), Page(), BatchTableProps, BadgeVariant, colorStyles, ConfidenceGauge(), ConfidenceGaugeProps, PredictionCard() (+15 more)

### Community 7 - "Discovery Pipeline & Active Learning"
Cohesion: 0.13
Nodes (26): Base, get_discovery_job_status(), get_discovery_report(), get_ensemble_disagreement_report(), get_retrain_history(), list_discovery_history(), get, Session (+18 more)

### Community 8 - "Dual Evidential Regression Head"
Cohesion: 0.09
Nodes (15): Any, Return attention weights from the last forward pass., DERHead, Tensor, Deep Evidential Regression output head., Predict NIG parameters for evidential regression., Return (mu, v, alpha, beta) each shaped [B, 1]., Return aleatoric uncertainty and clamp to finite range. (+7 more)

### Community 9 - "MultiScale Crystal Dataset"
Cohesion: 0.14
Nodes (26): download_mp_dataset(), _load_existing(), _parse_crystal_system(), _parse_spacegroup_number(), Any, Path, Materials Project dataset download and filtering., Download and filter MP structures, writing gzipped structures and CSV labels. (+18 more)

### Community 10 - "Prediction & Batch Screening API"
Cohesion: 0.07
Nodes (28): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+20 more)

### Community 11 - "Evaluation & Uncertainty Metrics"
Cohesion: 0.15
Nodes (24): compute_all_metrics(), compute_coverage(), compute_ece(), compute_ood_auroc(), compute_regression_metrics(), compute_spearman_uq_error(), ndarray, Evaluation metrics for regression, uncertainty and OOD detection. (+16 more)

### Community 12 - "Prediction & Batch Screening API"
Cohesion: 0.10
Nodes (19): BaseModel, Single Prediction REST API Router for MatScreen AI. Supports CIF file upload or…, SinglePredictRequest, Batch Screening REST API Router for MatScreen AI. Processes multiple CIF files,…, get_db(), init_db(), Database Connection & Initialization for MatScreen AI. Creates SQLite engine…, Initialize database tables and run automatic schema migration for SQLite. (+11 more)

### Community 13 - "Multi-Scale Attention Fusion"
Cohesion: 0.13
Nodes (18): Tensor, Fuse 3 scale embeddings using attention with h1 as anchor query., Return fused embeddings and mean-per-head attention weights [B, 3]., Return stored attention weights from the last forward pass., ScaleFusionAttention, Tensor, Cross-attention fusion across scale embeddings., Fuse 3 scale embeddings using attention with h1 as anchor query. (+10 more)

### Community 14 - "Discovery Pipeline & Active Learning"
Cohesion: 0.12
Nodes (16): ActiveLearningService, Active Learning retraining orchestrator for GNN fine-tuning., DiscoveryJobOrchestrator, Orchestrates multi-stage materials discovery pipelines and Tier 2 validation., MultiFidelityOrchestrator, Any, Multi-fidelity expected information-gain decision orchestrator., Evaluate candidate expected information gain and return routing decision.… (+8 more)

### Community 15 - "Virtual Lab Physics Simulation"
Cohesion: 0.12
Nodes (14): Any, Structure, Sweep lattice strains (-5% to +5%) along target axis, evaluate formation energy…, Compute Nudged Elastic Band (NEB) ion migration path and activation energy…, Run simulated NVT Langevin Molecular Dynamics thermal annealing trajectory at…, Service driving interactive virtual lab physics tests on crystal structures., Analyze all periodic table elements for site substitution compatibility using…, Push Virtual Lab simulation results back into discovery_candidates database… (+6 more)

### Community 16 - "Crystal Gnn Scripts"
Cohesion: 0.17
Nodes (17): main(), Audit script to diagnose the MAE discrepancy (0.0641 vs 0.8640) for A7 and…, main(), Fast & Accurate Audit for A7 on SOAP-LOCO Test Set. Constructs…, main(), evaluate_a7_test_set(), main(), per_bin_coverage() (+9 more)

### Community 17 - "Prediction & Batch Screening API"
Cohesion: 0.12
Nodes (16): Page(), metadata, plexMono, plexSans, AuthGate(), emptySubscribe(), isActive(), NAV_GROUPS (+8 more)

### Community 18 - "NNConv Graph Encoder"
Cohesion: 0.14
Nodes (19): MultiScaleCollate, Collate function for batching multi-scale graph tuples., MultiScaleCollate, Collate function for batching multi-scale graph tuples., Single-scale ablation model using only r1 encoder., Forward pass using only first-radius graph batch., SingleScaleGNN, _build_model() (+11 more)

### Community 19 - "Virtual Lab Physics Simulation"
Cohesion: 0.14
Nodes (23): ai_lab_assistant_chat(), AIChatRequest, NEBBarrierRequest, NVTMdRequest, PhononCheckRequest, Any, BaseModel, post (+15 more)

### Community 20 - "Discovery Pipeline & Active Learning"
Cohesion: 0.13
Nodes (16): CandidateTable(), Props, DiscoveryQueryBuilder(), DiscoveryQueryPayload, ParsedConstraint, Props, SAMPLE_PROMPTS, CandidateItem (+8 more)

### Community 21 - "Prediction & Batch Screening API"
Cohesion: 0.20
Nodes (19): confidenceVariant(), Page(), BatchTable(), confidenceBadge(), FILTERS, Badge, BadgeProps, BadgeVariant (+11 more)

### Community 22 - "NNConv Graph Encoder"
Cohesion: 0.11
Nodes (13): DERHead, Tensor, Deep Evidential Regression output head., Predict NIG parameters for evidential regression., Return (mu, v, alpha, beta) each shaped [B, 1]., Return aleatoric uncertainty and clamp to finite range., Return epistemic uncertainty and clamp to finite range., Cross-attention fusion across scale embeddings. (+5 more)

### Community 23 - "Dual Evidential Regression Head"
Cohesion: 0.18
Nodes (16): combined_loss(), evidential_loss(), Tensor, Evidential regression losses., Return total evidential loss and its NLL/regularization components., Return MSE during warm-up epochs., Switch from MSE warm-up to evidential loss after warm-up epochs., warm_up_loss() (+8 more)

### Community 24 - "Prediction & Batch Screening API"
Cohesion: 0.17
Nodes (21): _build_graph_for_radius(), default_checkpoint(), _discover_latest_checkpoint(), EdgePayload, health(), index(), NodePayload, _parse_structure() (+13 more)

### Community 25 - "MultiScale Crystal Dataset"
Cohesion: 0.16
Nodes (17): MultiScaleGNN, Three-scale model with attention/concat fusion and optional DER head., Run forward pass and return DER tuple or scalar prediction., MultiScaleGNN, Tensor, Three-scale model with attention/concat fusion and optional DER head., Run forward pass and return DER tuple or scalar prediction., Compute MC+DER uncertainty statistics over T stochastic passes. (+9 more)

### Community 26 - "Prediction & Batch Screening API"
Cohesion: 0.16
Nodes (14): Page(), Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle, Input (+6 more)

### Community 27 - "Weisfeiler-Lehman Expressivity Analysis"
Cohesion: 0.20
Nodes (17): Expressivity analysis tools., compute_crystal_graph(), find_resolved_pairs(), find_wl_collisions(), _h(), main(), Any, Structure (+9 more)

### Community 28 - "Crystal Gnn Scripts"
Cohesion: 0.16
Nodes (13): main(), Confirms A4 and A5 configs are truly identical after resolving OmegaConf…, _eval_mae(), main(), _make_loader(), _merge_cfg(), parse_radii_string(), Path (+5 more)

### Community 29 - "Prediction & Batch Screening API"
Cohesion: 0.16
Nodes (15): addLatticeLines(), checkpointEl, clearCrystalScene(), elementColor(), fmtEl, formatN(), initCrystalScene(), inputEl (+7 more)

### Community 30 - "Prediction & Batch Screening API"
Cohesion: 0.16
Nodes (13): passes_charge_neutrality(), Any, Structure, Generate candidate structures preserving target mobile ion (Li/Na) or halide…, Unified candidate generator dispatching to substitution or generative AI engine., Verify if crystal composition has physically valid, charge-balanced oxidation…, Generates candidate crystal structures via scaffold substitution & AI…, Create base pymatgen Structure for archetype scaffold. (+5 more)

### Community 31 - "MultiScale Crystal Dataset"
Cohesion: 0.18
Nodes (13): PyTorch dataset for multi-scale crystal graphs., build_node_features(), node_feature_dim(), Structure, Tensor, Preprocessing helpers for crystal graph construction., Build node feature tensor of shape [n_atoms, 123]., Encode distances with Gaussian RBF and normalize rows into [0, 1]. (+5 more)

### Community 32 - "MultiScale Crystal Dataset"
Cohesion: 0.18
Nodes (12): main(), main(), Comprehensive Batch CHGNet Benchmark on All 50k Colliding Pairs. Evaluates…, build_descriptors_for_all_radii(), load_dataset_subsampled(), main(), Collision scan on a statistically robust sample of 50,000 structures (matching…, Load raw structures, resolve material IDs, and subsample/convert to pymatgen… (+4 more)

### Community 33 - "MultiScale Crystal Dataset"
Cohesion: 0.17
Nodes (6): MultiScaleDataset, Any, Dataset, Path, Dataset that returns one graph per radius for each crystal., TestMultiScaleDataset

### Community 34 - "Evaluation & Uncertainty Metrics"
Cohesion: 0.21
Nodes (10): PhysicsValidationLayer, Any, Structure, Run ASE BFGS relaxation with the given calculator. Returns energy, relaxed…, Compute structural RMSD between two relaxed structures. Uses pymatgen…, Classify ensemble disagreement into actionable triage tiers. Thresholds are a…, Run CHGNet + MACE ensemble relaxation and compute disagreement metrics. If MACE…, Perform geometry optimization (relaxation) on candidate structure using MLIP.… (+2 more)

### Community 35 - "Evaluation & Uncertainty Metrics"
Cohesion: 0.21
Nodes (15): compute_all_metrics(), compute_coverage(), compute_ece(), compute_ood_auroc(), compute_regression_metrics(), compute_spearman_uq_error(), ndarray, Evaluation metrics for regression, uncertainty and OOD detection. (+7 more)

### Community 36 - "Prediction & Batch Screening API"
Cohesion: 0.19
Nodes (7): ELEMENT_NAMES, load3Dmol(), MolGlobal, MolViewer, Viewer3D(), Viewer3DProps, Window

### Community 37 - "Prediction & Batch Screening API"
Cohesion: 0.20
Nodes (11): get, Materials Search REST API Router for MatScreen AI. Integrates Materials Project…, Search crystal materials from database / Materials Project., Perform 1-click GNN property screening on a searched material., screen_searched_material(), search_materials(), MaterialsProjectService, Any (+3 more)

### Community 38 - "Conformal UQ Calibration"
Cohesion: 0.21
Nodes (8): BandGapEstimatorService, Any, Structure, Estimate band gap with disclosed uncertainty. Tries matgl ML model first (if…, Estimate band gap using pretrained matgl M3GNet model., Calibrated electronegativity/ionic-radius band gap heuristic. Uses halide anion…, Classify band gap into application-relevant categories., Tier B band gap estimator with calibrated uncertainty disclosure. Uses matgl…

### Community 39 - "Discovery Pipeline & Active Learning"
Cohesion: 0.14
Nodes (9): Any, Submit a discovery query job and launch background worker., Execute Tier 2 Physics Validation (CHGNet+MACE ensemble relaxation) on selected…, Execute full discovery pipeline in background worker., Any, Compute energy above the convex hull for a candidate material. Args:…, Compute S.U.N. and M.S.U.N. rates for a list of candidate dicts. Each candidate…, Fetch phase diagram reference entries for a chemical system. Returns: entries:… (+1 more)

### Community 40 - "Prediction & Batch Screening API"
Cohesion: 0.30
Nodes (13): create_access_token(), get_password_hash(), login(), BaseModel, post, Session, JWT Authentication Routes for MatScreen AI. Supports user registration, login,…, register() (+5 more)

### Community 41 - "Prediction & Batch Screening API"
Cohesion: 0.29
Nodes (9): GNNPredictorService, Any, Data, Structure, Construct PyTorch Geometric Data graph for a specific cutoff radius., Run GNN inference, DER uncertainty, Conformal 90% calibration, and scale…, Batched inference across multiple crystal structures. Constructs PyG Batch…, Multi-Task Band Gap Predictor Head (Eg in eV) delegated to… (+1 more)

### Community 42 - "Multi-Scale Attention Fusion"
Cohesion: 0.23
Nodes (11): plot_attention_heatmap(), plot_reliability_diagram(), plot_uncertainty_error_scatter(), ndarray, Visualization helpers for uncertainty and interpretability outputs., Plot uncertainty vs absolute error calibration diagram., Plot scale-attention heatmap for a batch., Scatter uncertainty against error. (+3 more)

### Community 43 - "Discovery Pipeline & Active Learning"
Cohesion: 0.18
Nodes (13): DiscoveryQueryRequest, parse_natural_language_prompt(), PromptParseRequest, BaseModel, post, Trigger Tier 2 MLIP structure relaxation on shortlisted candidates., Trigger Active Learning fine-tuning loop on newly Tier 2 validated candidates., Parse free-text natural language specification prompt into structured discovery… (+5 more)

### Community 44 - "Prediction & Batch Screening API"
Cohesion: 0.21
Nodes (10): download_csv_report(), ExportCSVRequest, BaseModel, post, Report Generator & CSV Export REST API Router for MatScreen AI., Generate and return CSV spreadsheet for candidate materials., ExportService, Any (+2 more)

### Community 45 - "Prediction & Batch Screening API"
Cohesion: 0.23
Nodes (8): CrystalPredictor, _find_default_ckpt(), main(), Path, Structure, Predict formation energy and uncertainty for a CIF file., Out-of-the-box predictor using pretrained A7 model weights., Predict formation energy and uncertainty for a PyMatGen Structure.

### Community 46 - "Prediction & Batch Screening API"
Cohesion: 0.20
Nodes (10): compare_models(), CompareRequest, BaseModel, post, UploadFile, Multi-Scale vs Single-Scale Model Comparison REST API Router for MatScreen AI.…, Compare Multi-Scale GNN (A7) vs Single-Scale GNN baseline for a given CIF…, CIFParserService (+2 more)

### Community 47 - "Dual Evidential Regression Head"
Cohesion: 0.17
Nodes (10): predict_single(), post, Session, UploadFile, Predict material formation energy per atom with evidential uncertainty &…, batch_screen_materials(), post, Session (+2 more)

### Community 48 - "Prediction & Batch Screening API"
Cohesion: 0.23
Nodes (7): ConstraintParser, Any, Structure, Parses user query constraints and applies hard/soft filters., Calculate weighted average raw material cost ($/kg) for the crystal composition., Calculate mass density in g/cm³., Evaluate a candidate crystal structure against hard filters. Returns:…

### Community 49 - "Virtual Lab Physics Simulation"
Cohesion: 0.20
Nodes (9): get_mutation_matrix(), load_structure(), get, Get charge-gated periodic table element substitution options., Load structure from Discovery candidate DB, Materials Project, or CIF text., Any, Structure, Parse CIF text into PyMatGen Structure and metadata payload. (+1 more)

### Community 50 - "Virtual Lab Physics Simulation"
Cohesion: 0.27
Nodes (4): PageHeader(), PageHeaderProps, Textarea, TextareaProps

### Community 51 - "Dual Evidential Regression Head"
Cohesion: 0.36
Nodes (8): combined_loss(), evidential_loss(), Tensor, Evidential regression losses., Return total evidential loss and its NLL/regularization components., Return MSE during warm-up epochs., Switch from MSE warm-up to evidential loss after warm-up epochs., warm_up_loss()

### Community 52 - "preprocessing Module"
Cohesion: 0.27
Nodes (10): atomic_number_onehot(), _cached_atomic_number_onehot(), _cached_element_scalar_features(), element_scalar_features(), Element, ndarray, Convert to float while replacing invalid values with a default., Return one-hot encoding for atomic number in [1, max_z]. (+2 more)

### Community 53 - "MultiScale Crystal Dataset"
Cohesion: 0.33
Nodes (9): batch_triple(), Shared pytest fixtures for Crystal GNN., small_model(), tiny_dataset(), tiny_labels(), tiny_structure_list(), tio2_pair_structures(), trained_predictions() (+1 more)

### Community 54 - "MultiScale Crystal Dataset"
Cohesion: 0.31
Nodes (8): build_graph_descriptor(), collision_rate_at_radius(), load_dataset_subsampled(), main(), Collision scan: quantifies how often distinct crystal structures produce near-…, Returns: collision_rate: fraction of structures involved in at least one…, Load raw structures, subsample them in raw dict format, then convert to…, Build a local environment descriptor matching the GNN's message passing: For…

### Community 55 - "Multi-Scale Attention Fusion"
Cohesion: 0.22
Nodes (8): plot_attention_heatmap(), plot_reliability_diagram(), plot_uncertainty_error_scatter(), ndarray, Visualization helpers for uncertainty and interpretability outputs., Plot uncertainty vs absolute error calibration diagram., Plot scale-attention heatmap for a batch., Scatter uncertainty against error.

### Community 56 - "run_managed Module"
Cohesion: 0.47
Nodes (8): load_state(), main(), print_summary_table(), Path, Orchestrator script to sequentially run and resume Crystal GNN ablations under…, Helper to load PyTorch checkpoint metadata using a light sub-process to avoid…, save_state(), torch_load_meta()

### Community 57 - "Prediction & Batch Screening API"
Cohesion: 0.28
Nodes (5): ParetoRanker, Any, Non-dominated sorting algorithm for multi-objective candidate ranking., Check if candidate p1 strictly dominates candidate p2. p1 dominates p2 iff p1…, Perform fast non-dominated sorting on a list of candidate dictionaries. Assigns…

### Community 58 - "MultiScale Crystal Dataset"
Cohesion: 0.39
Nodes (4): Batch, Data, Structure, Tensor

### Community 59 - "Conformal UQ Calibration"
Cohesion: 0.29
Nodes (4): LocallyAdaptiveConformalPredictor, Calibrate and evaluate locally adaptive split-conformal prediction intervals by…, Save calibrated predictor state to JSON., Load predictor state from JSON.

### Community 60 - "Prediction & Batch Screening API"
Cohesion: 0.43
Nodes (6): _build_model(), _load_data(), main(), Structure, Analyze relationship between prediction uncertainty/interval width and SOAP…, _species_union()

### Community 61 - "Prediction & Batch Screening API"
Cohesion: 0.33
Nodes (6): AttentionChart(), AttentionChartProps, chartColors, chartTints, ScaleKey, ScaleAttention

### Community 62 - "Prediction & Batch Screening API"
Cohesion: 0.40
Nodes (5): Composition, Verify real known cathodes pass charge neutrality gate., Verify unphysical charge-imbalanced formulas are rejected., test_known_cathodes_pass_charge_neutrality(), test_unphysical_formulas_rejected_by_charge_gate()

### Community 63 - "MultiScale Crystal Dataset"
Cohesion: 0.40
Nodes (3): Tensor, Compute MC+DER uncertainty statistics over T stochastic passes for…, Compute MC+DER uncertainty statistics over T stochastic passes.

### Community 64 - "evaluate_alignn_soap_loco Module"
Cohesion: 0.60
Nodes (4): evaluate_alignn(), load_soap_loco_test_set(), main(), Task 3: Fast Batched ALIGNN Zero-Shot Evaluation on SOAP-LOCO Test Set (Path A)…

### Community 65 - "run_der_lambda_sweep Module"
Cohesion: 0.70
Nodes (4): evaluate_checkpoint(), main(), Path, train_lambda()

### Community 66 - "run_external_benchmark Module"
Cohesion: 0.60
Nodes (4): evaluate_alignn(), evaluate_matgl(), main(), External SOTA Model Benchmark: ALIGNN & MatGL (MEGNet / M3GNet) Evaluates…

### Community 68 - "Prediction & Batch Screening API"
Cohesion: 0.40
Nodes (3): NaturalLanguageQueryParser, Any, Parses free-text natural language material specification prompts into…

### Community 69 - "Prediction & Batch Screening API"
Cohesion: 0.50
Nodes (3): Structure, Compute deterministic hash of structure formula and lattice parameters., Verify structural novelty of a candidate crystal. Returns: novelty_status…

### Community 70 - "_analyze_collisions Module"
Cohesion: 0.83
Nodes (3): get_structure_info(), main(), resolve_material_id()

### Community 71 - "_run_external_models Module"
Cohesion: 0.83
Nodes (3): main(), run_alignn(), run_matgl()

### Community 72 - "Evaluation & Uncertainty Metrics"
Cohesion: 0.50
Nodes (3): Any, Structure, Analyze structural void volume and channel bottleneck metrics. Returns: dict…

## Knowledge Gaps
- **123 isolated node(s):** `crystal_gnn`, `statusEl`, `inputEl`, `checkpointEl`, `fmtEl` (+118 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 559 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **42 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `GNNPredictorService` connect `Prediction & Batch Screening API` to `Prediction & Batch Screening API`, `Conformal UQ Calibration`, `Discovery Pipeline & Active Learning`, `Prediction & Batch Screening API`, `Prediction & Batch Screening API`, `Dual Evidential Regression Head`, `Discovery Pipeline & Active Learning`, `Prediction & Batch Screening API`, `Virtual Lab Physics Simulation`, `NNConv Graph Encoder`, `MultiScale Crystal Dataset`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `MultiScaleGNN` connect `MultiScale Crystal Dataset` to `NNConv Graph Encoder`, `Dual Evidential Regression Head`, `Prediction & Batch Screening API`, `Multi-Scale Attention Fusion`, `Crystal Gnn Scripts`, `Crystal Gnn Scripts`, `Prediction & Batch Screening API`, `NNConv Graph Encoder`, `MultiScale Crystal Dataset`, `NNConv Graph Encoder`, `Dual Evidential Regression Head`, `Prediction & Batch Screening API`, `Prediction & Batch Screening API`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Why does `MultiScaleGNN` connect `MultiScale Crystal Dataset` to `NNConv Graph Encoder`, `Dual Evidential Regression Head`, `Prediction & Batch Screening API`, `Multi-Scale Attention Fusion`, `Crystal Gnn Scripts`, `NNConv Graph Encoder`, `MultiScale Crystal Dataset`, `NNConv Graph Encoder`, `Prediction & Batch Screening API`, `MultiScale Crystal Dataset`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `MultiScaleGNN` (e.g. with `DERHead` and `NNConvEncoder`) actually correct?**
  _`MultiScaleGNN` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `_build_model()` (e.g. with `MultiScaleGNN` and `SingleScaleGNN`) actually correct?**
  _`_build_model()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `crystal_gnn`, `statusEl`, `inputEl` to the rest of the system?**
  _123 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `NNConv Graph Encoder` be split into smaller, more focused modules?**
  _Cohesion score 0.05952380952380952 - nodes in this community are weakly interconnected._