完整运行一次后得到的图不令人满意。 接下来我们从头review我们的流程，首先是这个stage taxonomy，你现在了解了我们做这个stage的目的是为了剪枝：通过只让不同stage之间的工具判断边的方式大大减少api的调用量。那么在分stage的时候就一定要有十足的把握保证同一stage的工具之间不会产生边。在这样的语境下，这个stage taxonomy是不是有更好更专业的设定方法？请你深度思考，给出更好版本的分发方案，并且分析为什么更好。

结论：**是的，现在这个 stage taxonomy 需要重做**。

当前版本更像是一个“药物发现功能大类 taxonomy”，但你现在用它的目的是 **API 调用剪枝**。这两个目标不一样。用于剪枝时，stage 的定义必须更严格：

> **同一个 pruning stage 内部的工具，原则上不应存在 `tool_A -> tool_B` 的直接相邻调用关系。**

当前 taxonomy 里已经显式承认了一个问题：`stage_pruning_policy` 里写到 previous `cross_stage_only` 简单但不够安全，因为有些合法关系天然发生在 same-stage 内部。 这说明当前 stage 设计本身不适合直接作为 hard pruning 依据。

---

# 1. 当前 taxonomy 的核心问题

当前 16 个 stage 太偏“领域功能大类”，例如：

```text
target_structure_preparation
ligand_generation_optimization
affinity_rescoring_virtual_screening
physics_based_simulation_free_energy
protein_design_engineering
reporting_visualization_io
```

这些 stage 在科学上是合理的，但用于剪枝不安全，因为它们内部本身就可能存在顺序关系。

例如：

```text
target_structure_preparation:
  convert_complex_cif_to_pdb -> fix_pdb
  fix_pdb -> convert_pdb_to_pdbqt_dock
  pulchura_rebuild -> pack_sidechains

physics_based_simulation_free_energy:
  prepare_complex -> run_mmpbsa
  run_mmpbsa -> analyze_mmpbsa
  protein_openmm_md -> openmm_extract_frames

ligand_generation_optimization:
  get_pepinvent_info -> pepinvent_peptide_sampling_by_template
  reinvent_denovo_sampling -> reinvent_mol2mol_sampling

affinity_rescoring_virtual_screening:
  equiscore_pocket -> equiscore_screen

protein_design_engineering:
  chroma_* -> proteinmpnn_tool
  proteinmpnn_tool -> foldx_tool
  evobind_tool -> chai1_predict / hdock_tool
```

如果采用 `cross_stage_only`，这些边全部会被剪掉。

所以问题不是 LLM 没判断好，而是 **stage 本身把可能相邻的工具放进了同一个 bucket**。

---

# 2. 正确原则：stage 应该是 “pruning stage”，不是 “domain stage”

我建议把 taxonomy 拆成两层：

```text
domain_stage:
  给人看的科学功能大类。
  例如 protein_design_engineering, docking, ADMET, MD simulation。

pruning_stage:
  给剪枝用的有向数据流层。
  同一个 pruning_stage 内尽量只放 parallel / alternative / descriptor tools，
  不放可能直接串联的工具。
```

后续 API 调用应该基于：

```text
primary_pruning_stage + allowed_stage_transition_matrix
```

而不是只用 broad `primary_stage`。

也就是说，真正的剪枝逻辑不应该是：

```text
if stage_A != stage_B:
    ask LLM
```

而应该是：

```text
if (pruning_stage_A, pruning_stage_B) in allowed_stage_transition_matrix:
    ask LLM
else:
    prune
```

这是更专业的做法。

原因是：只靠 “不同 stage” 剪枝不够强。当前 81 个工具的有向 pair 是：

```text
81 × 80 = 6480
```

如果只是去掉同 stage pair，当前 16-stage taxonomy 实际剪掉的数量有限，而且还会误剪合法 same-stage 边。更合理的是：

```text
细粒度 pruning_stage
+
有向 stage-transition matrix
```

这样既减少调用量，又降低漏边。

---

# 3. 还要单独处理 alternative_to

这里必须强调一个关键点。

你最终图允许 typed edges，包括：

```text
generates_input_for
preprocesses_for
validates
ranks_after
alternative_to
refines
```

其中 `alternative_to` 很多时候天然发生在同一类工具之间。

例如：

```text
pred_pocket_prank <-> fpocket_toolkit
pred_protein_structure_esmfold <-> chai1_predict
prolif_docking <-> analyze_protein_ligand_interactions
visualize_protein <-> visualize_molecule 不是 alternative，但属于同类 reporting tools
```

所以如果你说“同 stage 完全不判断任何边”，那 `alternative_to` 这类边会丢失。

更稳的设计是：

```text
1. pruning_stage 用于 transition edges：
   generates_input_for / preprocesses_for / validates / ranks_after / refines

2. alternative_to 不走 pairwise LLM 全量判断。
   它由 alternative_cluster_id 或 tool family deterministic rule 生成。
```

也就是说：

```text
same pruning_stage 内部默认不判断 direct transition edge；
但可以通过 cluster 机制生成 alternative_to。
```

这能同时满足：

```text
节点只保留工具；
边可以多类型；
LLM API 调用量减少；
同 stage transition 漏边风险降低。
```

---

# 4. 新版设计：Pruning-stage taxonomy v3

下面是我建议的新版本。它比当前 16 个 stage 更细，但不是为了“科学分类更细”，而是为了保证 **同 stage 内部尽量不产生 direct transition edge**。

## 4.1 Target / protein structure track

```yaml
target_identifier_acquisition:
  definition: Retrieve target identifiers, metadata, or raw sequence records.
  tools:
    - retrieve_protein_sequence

target_structure_acquisition:
  definition: Retrieve experimentally resolved or database structures.
  tools:
    - retrieve_protein_structure_by_pdb_id
    - retrieve_protein_structure_by_uniprot_id
    - retrieve_protein_structure_by_gene_name

protein_sequence_qc_descriptor:
  definition: Validate or characterize protein sequences before structure prediction/design.
  tools:
    - is_valid_protein_sequence
    - calculate_protein_sequence_properties

target_structure_prediction:
  definition: Predict protein or complex structures from sequence.
  tools:
    - pred_protein_structure_esmfold
    - chai1_predict

structure_format_conversion:
  definition: Convert predicted or complex structures into downstream structural file formats.
  tools:
    - convert_complex_cif_to_pdb

structure_chain_extraction:
  definition: Extract structural chains or chain sequences from structure files.
  tools:
    - extract_and_save_chains
    - extract_pdb_chains

structure_rebuild_repair:
  definition: Repair, rebuild, protonate, or complete protein structures.
  tools:
    - fix_pdb
    - pulchura_rebuild

structure_sidechain_completion:
  definition: Complete or repack side chains from backbone structures.
  tools:
    - pack_sidechains

receptor_docking_format_preparation:
  definition: Convert prepared receptor structures into docking-specific receptor formats.
  tools:
    - convert_pdb_to_pdbqt_dock

protein_structure_qc_descriptor:
  definition: Compute PDB-level geometry, quality, composition, or structural descriptors.
  tools:
    - calculate_pdb_basic_info
    - calculate_pdb_structural_geometry
    - calculate_pdb_quality_metrics
    - calculate_pdb_composition_info
```

这里把原来的 `target_structure_preparation` 拆开了。原因是它内部明显存在顺序关系。

例如：

```text
chai1_predict -> convert_complex_cif_to_pdb -> fix_pdb -> convert_pdb_to_pdbqt_dock
```

如果这些都放在 `target_structure_preparation`，cross-stage-only 会误剪。

---

## 4.2 Ligand track

```yaml
ligand_identity_acquisition:
  definition: Retrieve SMILES or compound identity from external chemistry databases.
  tools:
    - retrieve_smiles_by_compoundname

ligand_smiles_validation:
  definition: Validate SMILES strings before generation, property calculation, preparation, or docking.
  tools:
    - is_valid_smiles

ligand_file_preparation_conversion:
  definition: Convert valid SMILES or molecule lists into downstream molecular file formats.
  tools:
    - convert_smiles_to_format

ligand_descriptor_property_scoring:
  definition: Compute molecular descriptors, ADMET, drug-likeness, similarity, fragment, or phenotypic scores.
  tools:
    - pred_mol_admet
    - calculate_dleps_score
    - calculate_mol_basic_info
    - calculate_mol_hydrophobicity
    - calculate_mol_hbond
    - calculate_mol_structure_complexity
    - calculate_mol_topology
    - calculate_mol_drug_chemistry
    - calculate_mol_charge
    - calculate_mol_complexity
    - calculate_morgan_fingerprint_similarity
    - calculate_common_fragments

ligand_generation_template_acquisition:
  definition: Retrieve or prepare predefined peptide/template information for generators.
  tools:
    - get_pepinvent_info

ligand_de_novo_or_template_generation:
  definition: Generate new molecules, linkers, R-groups, scaffolds, or peptides from templates/warheads/scaffolds.
  tools:
    - reinvent_denovo_sampling
    - linkinvent_linker_sampling_by_warheads
    - linkinvent_linker_sampling_by_warhead_pair_name
    - libinvent_rgroup_sampling_by_scaffold
    - libinvent_rgroup_sampling_by_scaffold_name
    - pepinvent_peptide_sampling_by_template
    - pepinvent_peptide_sampling_by_peptide

ligand_analog_optimization:
  definition: Optimize or expand molecules from existing seed molecules.
  tools:
    - reinvent_mol2mol_sampling
```

这里把 `reinvent_mol2mol_sampling` 单独分出来，因为它很可能接在 de novo / template generation 之后做 analog expansion。如果放在同一个 `ligand_generation_optimization`，会误剪：

```text
reinvent_denovo_sampling -> reinvent_mol2mol_sampling
```

`ligand_descriptor_property_scoring` 里放 12 个 descriptor/property 工具是合理的，因为这些工具通常是 parallel calculators，而不是彼此直接相邻调用。

---

## 4.3 Binding-site / docking / scoring track

现代 docking / virtual screening workflow 通常会区分 binding-site identification、pose generation/conformational sampling、scoring/rescoring；例如近期 docking 框架 TriDS 就明确把 binding site identification、conformational sampling 和 scoring 作为可集成但可区分的环节。([arXiv][1]) 传统 docking baseline 对比中也区分 known binding-site condition 和 unknown binding-site condition，并在 unknown 情况下引入 automated pocket identification。([arXiv][2])

建议拆成：

```yaml
binding_site_detection:
  definition: Detect, rank, or define candidate binding pockets from protein structures.
  tools:
    - pred_pocket_prank
    - fpocket_toolkit

pose_generation_docking:
  definition: Generate protein-ligand, protein-protein, peptide-protein, or complex docking poses.
  tools:
    - molecule_docking_quickvina_fullprocess
    - hdock_tool

docking_pose_pocket_extraction_for_rescoring:
  definition: Extract pockets or prepare pocket-level inputs from docking results for rescoring.
  tools:
    - equiscore_pocket

affinity_prediction_rescoring_screening:
  definition: Predict affinity, rescore poses, rank candidates, or perform virtual screening.
  tools:
    - pred_binding_affinity_boltz2
    - equiscore_screen
    - karmadock_tool

end_to_end_docking_scoring_pipeline:
  definition: One-click or wrapper pipelines that internally combine several docking/scoring steps.
  tools:
    - equiscore_pipeline
```

为什么 `equiscore_pocket` 要单独出来？

因为它天然可以接：

```text
docking result -> equiscore_pocket -> equiscore_screen
```

如果 `equiscore_pocket` 和 `equiscore_screen` 同 stage，会误剪。

为什么 `equiscore_pipeline` 单独放？

因为它是 wrapper / pipeline 工具，不应该被强行放进某个普通 transition stage，否则会产生大量假边。它更像是：

```text
alternative_to: equiscore_pocket + equiscore_screen
```

---

## 4.4 Interaction analysis / validation track

```yaml
static_interaction_analysis_validation:
  definition: Analyze static protein-ligand, peptide-protein, or protein-protein complexes or docking poses.
  tools:
    - prolif_docking
    - prolif_pdb
    - analyze_protein_ligand_interactions

trajectory_interaction_analysis:
  definition: Analyze interactions from MD trajectories or dynamic complex ensembles.
  tools:
    - prolif_md
    - prolif_protein_protein
```

这里建议拆 static 和 trajectory。

原因是它们的 upstream 不一样：

```text
docking_pose / complex_pdb -> static_interaction_analysis_validation
trajectory / topology -> trajectory_interaction_analysis
```

它们不太会彼此直接相邻调用。

---

## 4.5 Physics simulation / free-energy track

```yaml
simulation_system_preparation:
  definition: Prepare protein-only or protein-ligand/protein-protein systems for MD or free-energy calculation.
  tools:
    - prepare_complex
    - prepare_protein_md

simulation_or_sampling_execution:
  definition: Run MD, coarse-grained simulation, conformational sampling, or relaxation workflows.
  tools:
    - protein_openmm_md
    - run_bioemu
    - openawsem_sim
    - goca_pipeline

simulation_ensemble_extraction:
  definition: Extract representative structures or frames from simulation/sampling outputs.
  tools:
    - openmm_extract_frames
    - extract_bioemu_structures
    - openawsem_traj_extract

free_energy_calculation:
  definition: Compute binding free energy using MM/PBSA, MM/GBSA, or related methods.
  tools:
    - run_mmpbsa
    - gmx_mmpbsa_propro

free_energy_result_analysis:
  definition: Analyze and summarize free-energy calculation outputs.
  tools:
    - analyze_mmpbsa

residue_numbering_mapping:
  definition: Map residue numbering between UniProt, PDB, predicted structures, and tool-internal schemes.
  tools:
    - residue_mapper
```

这里是当前 taxonomy 最需要修的地方之一。

原来的 `physics_based_simulation_free_energy` 把 preparation、simulation、free-energy calculation、analysis 全放在一起，这会误剪：

```text
prepare_complex -> run_mmpbsa
run_mmpbsa -> analyze_mmpbsa
prepare_protein_md -> protein_openmm_md
protein_openmm_md -> openmm_extract_frames
```

新版必须拆开。

---

## 4.6 Protein / peptide design track

```yaml
protein_backbone_or_complex_generation:
  definition: Generate protein backbones, complexes, symmetric assemblies, or binder candidates.
  tools:
    - chroma_monomer
    - chroma_complex
    - chroma_symmetry
    - evobind_tool

protein_sequence_design:
  definition: Design or score sequences conditioned on protein structures/backbones.
  tools:
    - proteinmpnn_tool

protein_design_repair_mutation_evaluation:
  definition: Repair, mutate, score, or evaluate designed proteins/interfaces.
  tools:
    - foldx_tool
```

原来的 `protein_design_engineering` 也太粗。这里面有明显链路：

```text
chroma_* -> proteinmpnn_tool -> foldx_tool
evobind_tool -> chai1_predict / hdock_tool / interaction_analysis
```

如果都放在同一 stage，会漏掉设计工作流里的关键边。

---

## 4.7 Reporting / visualization / I/O track

```yaml
file_import_decoding:
  definition: Decode external/base64 content into server-side files.
  tools:
    - base64_to_server_file

visualization_rendering:
  definition: Generate protein, molecule, or interaction visualization artifacts.
  tools:
    - visualize_protein
    - visualize_molecule
    - interaction_visualizer

file_export_encoding:
  definition: Encode server-side artifacts for transfer or reporting.
  tools:
    - server_file_to_base64
```

原来的 `reporting_visualization_io` 也应该拆。否则会误剪：

```text
base64_to_server_file -> visualize_protein
visualize_protein -> server_file_to_base64
interaction_visualizer -> server_file_to_base64
```

---

# 5. 新版完整 tool-to-pruning-stage 分配

下面是更适合剪枝的分配。

```yaml
tool_pruning_stage_map:

  retrieve_protein_sequence: target_identifier_acquisition
  retrieve_protein_structure_by_pdb_id: target_structure_acquisition
  retrieve_protein_structure_by_uniprot_id: target_structure_acquisition
  retrieve_protein_structure_by_gene_name: target_structure_acquisition

  is_valid_protein_sequence: protein_sequence_qc_descriptor
  calculate_protein_sequence_properties: protein_sequence_qc_descriptor

  pred_protein_structure_esmfold: target_structure_prediction
  chai1_predict: target_structure_prediction

  convert_complex_cif_to_pdb: structure_format_conversion
  extract_and_save_chains: structure_chain_extraction
  extract_pdb_chains: structure_chain_extraction

  fix_pdb: structure_rebuild_repair
  pulchura_rebuild: structure_rebuild_repair
  pack_sidechains: structure_sidechain_completion
  convert_pdb_to_pdbqt_dock: receptor_docking_format_preparation

  calculate_pdb_basic_info: protein_structure_qc_descriptor
  calculate_pdb_structural_geometry: protein_structure_qc_descriptor
  calculate_pdb_quality_metrics: protein_structure_qc_descriptor
  calculate_pdb_composition_info: protein_structure_qc_descriptor

  retrieve_smiles_by_compoundname: ligand_identity_acquisition
  is_valid_smiles: ligand_smiles_validation
  convert_smiles_to_format: ligand_file_preparation_conversion

  pred_mol_admet: ligand_descriptor_property_scoring
  calculate_dleps_score: ligand_descriptor_property_scoring
  calculate_mol_basic_info: ligand_descriptor_property_scoring
  calculate_mol_hydrophobicity: ligand_descriptor_property_scoring
  calculate_mol_hbond: ligand_descriptor_property_scoring
  calculate_mol_structure_complexity: ligand_descriptor_property_scoring
  calculate_mol_topology: ligand_descriptor_property_scoring
  calculate_mol_drug_chemistry: ligand_descriptor_property_scoring
  calculate_mol_charge: ligand_descriptor_property_scoring
  calculate_mol_complexity: ligand_descriptor_property_scoring
  calculate_morgan_fingerprint_similarity: ligand_descriptor_property_scoring
  calculate_common_fragments: ligand_descriptor_property_scoring

  get_pepinvent_info: ligand_generation_template_acquisition

  reinvent_denovo_sampling: ligand_de_novo_or_template_generation
  linkinvent_linker_sampling_by_warheads: ligand_de_novo_or_template_generation
  linkinvent_linker_sampling_by_warhead_pair_name: ligand_de_novo_or_template_generation
  libinvent_rgroup_sampling_by_scaffold: ligand_de_novo_or_template_generation
  libinvent_rgroup_sampling_by_scaffold_name: ligand_de_novo_or_template_generation
  pepinvent_peptide_sampling_by_template: ligand_de_novo_or_template_generation
  pepinvent_peptide_sampling_by_peptide: ligand_de_novo_or_template_generation

  reinvent_mol2mol_sampling: ligand_analog_optimization

  pred_pocket_prank: binding_site_detection
  fpocket_toolkit: binding_site_detection

  molecule_docking_quickvina_fullprocess: pose_generation_docking
  hdock_tool: pose_generation_docking

  equiscore_pocket: docking_pose_pocket_extraction_for_rescoring

  pred_binding_affinity_boltz2: affinity_prediction_rescoring_screening
  equiscore_screen: affinity_prediction_rescoring_screening
  karmadock_tool: affinity_prediction_rescoring_screening

  equiscore_pipeline: end_to_end_docking_scoring_pipeline

  prolif_docking: static_interaction_analysis_validation
  prolif_pdb: static_interaction_analysis_validation
  analyze_protein_ligand_interactions: static_interaction_analysis_validation

  prolif_md: trajectory_interaction_analysis
  prolif_protein_protein: trajectory_interaction_analysis

  prepare_complex: simulation_system_preparation
  prepare_protein_md: simulation_system_preparation

  protein_openmm_md: simulation_or_sampling_execution
  run_bioemu: simulation_or_sampling_execution
  openawsem_sim: simulation_or_sampling_execution
  goca_pipeline: simulation_or_sampling_execution

  openmm_extract_frames: simulation_ensemble_extraction
  extract_bioemu_structures: simulation_ensemble_extraction
  openawsem_traj_extract: simulation_ensemble_extraction

  run_mmpbsa: free_energy_calculation
  gmx_mmpbsa_propro: free_energy_calculation
  analyze_mmpbsa: free_energy_result_analysis

  residue_mapper: residue_numbering_mapping

  chroma_monomer: protein_backbone_or_complex_generation
  chroma_complex: protein_backbone_or_complex_generation
  chroma_symmetry: protein_backbone_or_complex_generation
  evobind_tool: protein_backbone_or_complex_generation

  proteinmpnn_tool: protein_sequence_design
  foldx_tool: protein_design_repair_mutation_evaluation

  base64_to_server_file: file_import_decoding
  visualize_protein: visualization_rendering
  visualize_molecule: visualization_rendering
  interaction_visualizer: visualization_rendering
  server_file_to_base64: file_export_encoding
```

---

# 6. 推荐的 stage transition matrix

只分 stage 还不够。必须再加一个 `allowed_stage_transitions`。

核心思想：

```text
只有 stage_A -> stage_B 在这个矩阵里，才进入 pairwise LLM adjudication。
```

推荐第一版矩阵如下。

```yaml
allowed_stage_transitions:

  target_identifier_acquisition:
    - protein_sequence_qc_descriptor
    - target_structure_prediction
    - protein_backbone_or_complex_generation

  target_structure_acquisition:
    - structure_format_conversion
    - structure_chain_extraction
    - structure_rebuild_repair
    - protein_structure_qc_descriptor
    - binding_site_detection
    - static_interaction_analysis_validation
    - visualization_rendering

  protein_sequence_qc_descriptor:
    - target_structure_prediction
    - protein_backbone_or_complex_generation

  target_structure_prediction:
    - structure_format_conversion
    - structure_chain_extraction
    - structure_rebuild_repair
    - protein_structure_qc_descriptor
    - binding_site_detection
    - pose_generation_docking
    - static_interaction_analysis_validation
    - protein_sequence_design
    - visualization_rendering

  structure_format_conversion:
    - structure_chain_extraction
    - structure_rebuild_repair
    - protein_structure_qc_descriptor
    - static_interaction_analysis_validation
    - visualization_rendering

  structure_chain_extraction:
    - structure_rebuild_repair
    - structure_sidechain_completion
    - protein_sequence_qc_descriptor
    - protein_sequence_design
    - protein_structure_qc_descriptor
    - binding_site_detection
    - visualization_rendering

  structure_rebuild_repair:
    - structure_sidechain_completion
    - receptor_docking_format_preparation
    - protein_structure_qc_descriptor
    - binding_site_detection
    - pose_generation_docking
    - simulation_system_preparation
    - protein_sequence_design
    - protein_design_repair_mutation_evaluation
    - visualization_rendering

  structure_sidechain_completion:
    - structure_rebuild_repair
    - receptor_docking_format_preparation
    - protein_structure_qc_descriptor
    - binding_site_detection
    - pose_generation_docking
    - simulation_system_preparation
    - visualization_rendering

  receptor_docking_format_preparation:
    - pose_generation_docking

  protein_structure_qc_descriptor:
    - binding_site_detection
    - pose_generation_docking
    - simulation_system_preparation
    - protein_design_repair_mutation_evaluation

  ligand_identity_acquisition:
    - ligand_smiles_validation
    - ligand_descriptor_property_scoring
    - ligand_file_preparation_conversion
    - ligand_analog_optimization

  ligand_smiles_validation:
    - ligand_file_preparation_conversion
    - ligand_descriptor_property_scoring
    - ligand_de_novo_or_template_generation
    - ligand_analog_optimization
    - pose_generation_docking
    - affinity_prediction_rescoring_screening

  ligand_file_preparation_conversion:
    - pose_generation_docking
    - affinity_prediction_rescoring_screening
    - visualization_rendering

  ligand_descriptor_property_scoring:
    - ligand_analog_optimization
    - affinity_prediction_rescoring_screening
    - visualization_rendering

  ligand_generation_template_acquisition:
    - ligand_de_novo_or_template_generation

  ligand_de_novo_or_template_generation:
    - ligand_smiles_validation
    - ligand_descriptor_property_scoring
    - ligand_file_preparation_conversion
    - ligand_analog_optimization
    - pose_generation_docking
    - affinity_prediction_rescoring_screening
    - visualization_rendering

  ligand_analog_optimization:
    - ligand_smiles_validation
    - ligand_descriptor_property_scoring
    - ligand_file_preparation_conversion
    - pose_generation_docking
    - affinity_prediction_rescoring_screening
    - visualization_rendering

  binding_site_detection:
    - pose_generation_docking
    - affinity_prediction_rescoring_screening
    - visualization_rendering

  pose_generation_docking:
    - docking_pose_pocket_extraction_for_rescoring
    - affinity_prediction_rescoring_screening
    - static_interaction_analysis_validation
    - simulation_system_preparation
    - visualization_rendering

  docking_pose_pocket_extraction_for_rescoring:
    - affinity_prediction_rescoring_screening

  affinity_prediction_rescoring_screening:
    - static_interaction_analysis_validation
    - ligand_analog_optimization
    - visualization_rendering
    - file_export_encoding

  end_to_end_docking_scoring_pipeline:
    - static_interaction_analysis_validation
    - visualization_rendering
    - file_export_encoding

  static_interaction_analysis_validation:
    - visualization_rendering
    - ligand_analog_optimization
    - protein_design_repair_mutation_evaluation
    - file_export_encoding

  simulation_system_preparation:
    - simulation_or_sampling_execution
    - free_energy_calculation

  simulation_or_sampling_execution:
    - simulation_ensemble_extraction
    - trajectory_interaction_analysis
    - free_energy_calculation
    - visualization_rendering

  simulation_ensemble_extraction:
    - protein_structure_qc_descriptor
    - static_interaction_analysis_validation
    - visualization_rendering
    - residue_numbering_mapping

  trajectory_interaction_analysis:
    - visualization_rendering
    - file_export_encoding

  free_energy_calculation:
    - free_energy_result_analysis
    - visualization_rendering
    - file_export_encoding

  free_energy_result_analysis:
    - visualization_rendering
    - file_export_encoding

  residue_numbering_mapping:
    - static_interaction_analysis_validation
    - visualization_rendering

  protein_backbone_or_complex_generation:
    - protein_sequence_design
    - target_structure_prediction
    - structure_rebuild_repair
    - protein_design_repair_mutation_evaluation
    - pose_generation_docking
    - static_interaction_analysis_validation
    - visualization_rendering

  protein_sequence_design:
    - target_structure_prediction
    - structure_sidechain_completion
    - protein_design_repair_mutation_evaluation
    - protein_structure_qc_descriptor

  protein_design_repair_mutation_evaluation:
    - target_structure_prediction
    - static_interaction_analysis_validation
    - visualization_rendering
    - file_export_encoding

  file_import_decoding:
    - target_structure_acquisition
    - ligand_smiles_validation
    - ligand_file_preparation_conversion
    - protein_structure_qc_descriptor
    - visualization_rendering

  visualization_rendering:
    - file_export_encoding
```

这套 matrix 比单纯 `cross_stage_only` 更重要。它会直接减少大量不可能 pair，比如：

```text
ligand_descriptor_property_scoring -> target_structure_prediction
visualization_rendering -> docking
free_energy_result_analysis -> ligand_file_preparation_conversion
report_export -> protein_design
```

这些不应该进入 LLM 判断。

---

# 7. 关于同 stage 的处理策略

建议明确写进配置：

```yaml
same_pruning_stage_transition_policy:
  transition_edges: forbid
  alternative_edges: generate_by_cluster
  cross_check_edges: generate_by_cluster_or_explicit_evidence_only
  validation_edges: only_if_allowed_by_relation_specific_rule
```

解释：

```text
same stage 不判断普通 direct transition edge；
same stage 的 alternative_to 不走普通 pairwise API；
same stage 的 cross_check / validates 只有显式证据时才允许。
```

否则会出现两种坏情况：

```text
1. 为了保留 alternative_to，不得不判断大量 same-stage pair。
2. 为了省 API，直接丢掉 alternative_to。
```

最好的方案是：**transition edge 和 alternative edge 分开建。**

---

# 8. 进一步建议：给每个 stage 加 alternative_cluster_id

例如：

```yaml
alternative_clusters:

  structure_prediction:
    tools:
      - pred_protein_structure_esmfold
      - chai1_predict
    relation: alternative_to
    condition_notes:
      - ESMFold is faster for single-chain proteins under certain settings.
      - Chai-1 supports multi-chain complex prediction.

  pocket_detection:
    tools:
      - pred_pocket_prank
      - fpocket_toolkit
    relation: alternative_to

  molecular_generation:
    tools:
      - reinvent_denovo_sampling
      - linkinvent_linker_sampling_by_warheads
      - libinvent_rgroup_sampling_by_scaffold
      - pepinvent_peptide_sampling_by_template
    relation: alternative_family

  interaction_analysis_static:
    tools:
      - prolif_docking
      - prolif_pdb
      - analyze_protein_ligand_interactions
    relation: alternative_or_complementary_analysis

  simulation_sampling:
    tools:
      - protein_openmm_md
      - run_bioemu
      - openawsem_sim
      - goca_pipeline
    relation: alternative_or_method_family
```

这样就不需要让 LLM 对这些 same-stage pair 逐个判断 `alternative_to`，而是通过 cluster 生成候选，再少量抽检。

---

# 9. 我建议的最终配置结构

下一版 `stage_taxonomy.json` 不要再只包含 `stage_order` 和 `tool_stage_map`。建议改成：

```yaml
schema_version: molclaw_pruning_taxonomy_v3

purpose:
  - stage-aware pair pruning
  - transition-edge candidate reduction
  - alternative-edge deterministic generation

policies:
  use_primary_pruning_stage_only: true
  do_not_use_secondary_stage_for_pair_expansion: true
  transition_edges_same_stage_allowed: false
  alternative_edges_same_stage_allowed: true
  alternative_edges_generated_by_cluster: true
  pairwise_llm_requires_allowed_stage_transition: true

pruning_stages:
  ...

tool_pruning_stage_map:
  ...

allowed_stage_transitions:
  ...

alternative_clusters:
  ...

edge_type_stage_policy:
  generates_input_for:
    requires_allowed_stage_transition: true
    allow_same_stage: false
  preprocesses_for:
    requires_allowed_stage_transition: true
    allow_same_stage: false
  converts_format_for:
    requires_allowed_stage_transition: true
    allow_same_stage: false
  validates:
    requires_allowed_stage_transition: false
    allow_same_stage_if_explicit_evidence: true
  ranks_after:
    requires_allowed_stage_transition: true
    allow_same_stage: false
  refines:
    requires_allowed_stage_transition: true
    allow_same_stage: false
  alternative_to:
    generated_by_cluster: true
    pairwise_llm_optional: true
```

---

# 10. 最核心修改建议

如果只改一件事，就是：

> **不要再用当前 16 个 domain stages 做 `cross_stage_only`。改成 fine-grained pruning stages + allowed stage-transition matrix。**

如果只改两件事，就是：

> **把原来可能内部串联的 stage 拆开：`target_structure_preparation`、`ligand_generation_optimization`、`affinity_rescoring_virtual_screening`、`physics_based_simulation_free_energy`、`protein_design_engineering`、`reporting_visualization_io`。**

如果只改三件事，就是：

> **把 `alternative_to` 从普通 pairwise edge adjudication 中拿出来，用 alternative cluster 单独生成。**

这个新版 taxonomy 会比当前版本更适合你的目标：减少 LLM API 调用，同时不因为粗粒度 stage 把真实工具链剪掉。

[1]: https://arxiv.org/abs/2510.24186?utm_source=chatgpt.com "TriDS: AI-native molecular docking framework unified with binding site identification, conformational sampling and scoring"
[2]: https://arxiv.org/abs/2412.02889?utm_source=chatgpt.com "Deep-Learning Based Docking Methods: Fair Comparisons to Conventional Docking Workflows"
