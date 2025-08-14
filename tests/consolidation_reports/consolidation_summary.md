# Script Consolidation Analysis

**Analysis Date:** N/A

## Summary

- **Total Scripts Analyzed:** 12
- **Duplicate Functions Found:** 4
- **Redundant Script Groups:** 0
- **Consolidation Opportunities:** 1

## Consolidation Actions

### 1. Extract duplicate functions into shared utility modules
**Priority:** 1
**Type:** create_shared_utilities
**Steps:**
- Create src/fine_tune_llm/scripts/utils.py
- Move common functions to utils module
- Update scripts to import from utils
- Remove duplicate function definitions
**Files Affected:** 12

## Duplicate Functions

- **main** (11 occurrences)
  - consolidation_analysis.py
  - infer_model.py
  - launch_risk_ui.py
  - merge_lora.py
  - prepare_data.py
  - risk_prediction_ui.py
  - run_dashboard.py
  - train_high_stakes.py
  - tune_hyperparams.py
  - train_lora_sft.py
  - main.py
- **__init__** (2 occurrences)
  - prepare_data.py
  - high_stakes_audit_facade.py
- **load_config** (2 occurrences)
  - prepare_data.py
  - main.py
- **parse_args** (2 occurrences)
  - train_high_stakes.py
  - train_lora_sft.py

## Redundant Script Groups

