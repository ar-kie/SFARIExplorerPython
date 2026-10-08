#!/usr/bin/env Rscript
# =============================================================================
# Within-Organism Batch Correction
# =============================================================================
#
# Since organism and dataset are completely confounded (each dataset has only
# one organism), we cannot correct for dataset while preserving organism.
#
# Solution: Correct for dataset WITHIN each organism separately.
# - Human: 9 datasets → batch correct against each other
# - Mouse: 3 datasets → batch correct against each other
# - Zebrafish: 1 dataset → no correction needed
# - Drosophila: 1 dataset → no correction needed
#
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(limma)
  library(edgeR)
  library(sva)
  library(variancePartition)
  library(BiocParallel)
})

# =============================================================================
# CONFIG
# =============================================================================

SFARI_ROOT <- Sys.getenv("SFARI_ROOT", "/sc/arion/projects/ad-omics/raphael/SFARI")  # override with $SFARI_ROOT
EXCHANGE_DIR <- Sys.getenv("SFARI_EXCHANGE_DIR", file.path(SFARI_ROOT, "data/r_exchange"))  # pseudobulk <-> R exchange
N_CORES <- 8

# Option: Skip batch correction for organisms where timepoint is confounded?
# If TRUE, Mouse will NOT be batch corrected (preserves timepoint but keeps batch effects)
# If FALSE, Mouse WILL be batch corrected (removes batch effects but loses timepoint signal)
SKIP_IF_TIMEPOINT_CONFOUNDED <- FALSE  # Set to TRUE to preserve timepoint for Mouse

# Organoid datasets (will be batch corrected separately from in-vivo)
ORGANOID_DATASETS <- c("He (2024)", "Wang (2022)")

# Mouse disease models to exclude (unknown age, not relevant to developmental study)
MOUSE_DISEASE_MODELS <- c("APOE4/TREM2", "5xFAD")

register(MulticoreParam(N_CORES))

cat("=", rep("=", 59), "\n", sep = "")
cat("Within-Organism Batch Correction\n")
cat("=", rep("=", 59), "\n\n", sep = "")

# =============================================================================
# LOAD DATA
# =============================================================================

cat("1. Loading data...\n")

counts_file <- file.path(EXCHANGE_DIR, "pseudobulk_counts.csv")
counts <- read.csv(counts_file, row.names = 1, check.names = FALSE)
cat("   Counts:", nrow(counts), "samples x", ncol(counts), "genes\n")

# Try to load metadata with numeric time first
meta_file_numeric <- file.path(EXCHANGE_DIR, "pseudobulk_meta_numeric_time.csv")
meta_file <- file.path(EXCHANGE_DIR, "pseudobulk_meta.csv")

if (file.exists(meta_file_numeric)) {
  meta <- read.csv(meta_file_numeric, stringsAsFactors = FALSE)
  cat("   Using metadata with numeric time\n")
} else {
  meta <- read.csv(meta_file, stringsAsFactors = FALSE)
  cat("   Using original metadata (no numeric time)\n")
}

rownames(meta) <- meta$sample_id
meta <- meta[rownames(counts), ]

# Ensure factors
meta$dataset <- as.factor(meta$dataset)
meta$cell_type <- as.factor(meta$cell_type)
meta$organism <- as.factor(meta$organism)

# Handle timepoint - check for numeric_time column
if ("numeric_time" %in% colnames(meta)) {
  has_numeric_time <- TRUE
  # numeric_time is continuous, keep as numeric
  meta$numeric_time <- as.numeric(meta$numeric_time)
  n_valid_time <- sum(!is.na(meta$numeric_time))
  cat("   Numeric time: ", n_valid_time, "/", nrow(meta), " valid values\n", sep = "")
} else {
  has_numeric_time <- FALSE
  cat("   Numeric time: not available\n")
}

# Also keep categorical timepoint for reference
if ("timepoint" %in% colnames(meta)) {
  meta$timepoint[meta$timepoint == "unknown"] <- NA
  meta$timepoint <- as.factor(meta$timepoint)
}

# Add sample_type column (in_vivo vs organoid)
if (!"sample_type" %in% colnames(meta)) {
  meta$sample_type <- ifelse(meta$dataset %in% ORGANOID_DATASETS, "organoid", "in_vivo")
}
meta$sample_type <- as.factor(meta$sample_type)

cat("   Sample types:\n")
print(table(meta$organism, meta$sample_type))

# Exclude disease models
if ("timepoint" %in% colnames(meta)) {
  disease_mask <- meta$timepoint %in% MOUSE_DISEASE_MODELS
  n_disease <- sum(disease_mask)
  if (n_disease > 0) {
    cat(sprintf("\n   Excluding %d samples (disease models: %s)\n", 
                n_disease, paste(MOUSE_DISEASE_MODELS, collapse = ", ")))
    meta <- meta[!disease_mask, ]
    counts <- counts[rownames(meta), ]
    cat(sprintf("   Remaining samples: %d\n", nrow(meta)))
  }
}

# Show organism-dataset structure
cat("\n   Dataset-Organism structure:\n")
org_ds <- table(meta$organism, meta$dataset)
for (org in rownames(org_ds)) {
  datasets <- colnames(org_ds)[org_ds[org, ] > 0]
  cat(sprintf("     %s: %d datasets (%s)\n", org, length(datasets), 
              paste(datasets, collapse = ", ")))
}

# Convert to matrix
counts_mat <- as.matrix(counts)
mode(counts_mat) <- "integer"

# =============================================================================
# FILTER GENES
# =============================================================================

cat("\n2. Filtering genes...\n")

min_count <- 10
min_samples <- max(3, ceiling(0.05 * nrow(counts_mat)))
gene_pass <- colSums(counts_mat >= min_count) >= min_samples
counts_filt <- counts_mat[, gene_pass]

cat("   Genes:", ncol(counts_mat), "->", ncol(counts_filt), "\n")

# =============================================================================
# NORMALIZE (GLOBAL)
# =============================================================================

cat("\n3. Global normalization...\n")

dge <- DGEList(counts = t(counts_filt))
dge <- calcNormFactors(dge)

# Simple voom for now
design_simple <- model.matrix(~ 1, data = meta)
vobj <- voom(dge, design_simple, plot = FALSE)

cat("   Voom transformation done\n")

# =============================================================================
# WITHIN-ORGANISM (AND SAMPLE TYPE) BATCH CORRECTION
# =============================================================================

cat("\n4. Within-organism batch correction...\n")
cat("   Note: Organoids and in-vivo samples corrected separately\n")

# Initialize corrected matrix (start with original)
corrected_expr <- vobj$E

organisms <- unique(meta$organism)

for (org in organisms) {
  
  # For Human, process in_vivo and organoid separately
  if (org == "Human") {
    sample_types <- c("in_vivo", "organoid")
  } else {
    sample_types <- c("all")  # Non-human: process all together
  }
  
  for (stype in sample_types) {
    
    # Create mask for this subset
    if (stype == "all") {
      subset_mask <- meta$organism == org
      subset_label <- org
    } else {
      subset_mask <- meta$organism == org & meta$sample_type == stype
      subset_label <- paste0(org, " (", stype, ")")
    }
    
    n_subset <- sum(subset_mask)
    if (n_subset == 0) next
    
    cat(sprintf("\n   Processing %s...\n", subset_label))
    
    # Get samples for this subset
    subset_samples <- rownames(meta)[subset_mask]
    subset_meta <- meta[subset_mask, ]
    subset_expr <- vobj$E[, subset_mask]
    
    n_datasets <- length(unique(subset_meta$dataset))
    n_samples <- length(subset_samples)
    
    cat(sprintf("     Samples: %d, Datasets: %d\n", n_samples, n_datasets))
    
    if (n_datasets <= 1) {
      cat("     Only 1 dataset - no batch correction needed\n")
      next
    }
    
    # Drop unused factor levels
    subset_meta$dataset <- droplevels(subset_meta$dataset)
    subset_meta$cell_type <- droplevels(subset_meta$cell_type)
    
    # Check if we have enough samples per batch
    batch_sizes <- table(subset_meta$dataset)
    if (any(batch_sizes < 2)) {
      cat("     Warning: Some batches have <2 samples, using simple correction\n")
    }
    
    # For the rest of the processing, use subset_meta and subset_mask
    org_meta <- subset_meta
    org_mask <- subset_mask
    org_expr <- subset_expr
  
  # Design matrix for biological variables to preserve (cell_type, numeric_time)
  # Note: organism is constant within this subset, so we don't include it
  
  # Check if numeric_time is available for this organism
  if (has_numeric_time) {
    org_numeric_time <- meta$numeric_time[org_mask]
    n_valid_time <- sum(!is.na(org_numeric_time))
    pct_valid <- n_valid_time / nrow(org_meta) * 100
    
    if (n_valid_time >= 10 && pct_valid >= 30) {
      # Enough samples with numeric time - include as continuous covariate
      time_range <- range(org_numeric_time, na.rm = TRUE)
      
      # Clarify time scale
      if (stype == "organoid") {
        time_unit <- "days differentiation"
      } else if (org == "Zebrafish") {
        time_unit <- "hours post-fertilization"
      } else if (org == "Drosophila") {
        time_unit <- "days post-eclosion"
      } else {
        time_unit <- "days"
      }
      
      cat(sprintf("     Numeric time: %d values (%.0f%%), range %.1f - %.1f %s\n", 
                  n_valid_time, pct_valid, time_range[1], time_range[2], time_unit))
      
      # Center numeric_time within organism for stability
      org_meta$numeric_time_centered <- org_numeric_time - mean(org_numeric_time, na.rm = TRUE)
      
      # For samples without numeric time, impute with mean (0 after centering)
      org_meta$numeric_time_centered[is.na(org_meta$numeric_time_centered)] <- 0
      
      tryCatch({
        design_preserve <- model.matrix(~ 0 + cell_type + numeric_time_centered, data = org_meta)
        cat("     Preserving: cell_type + numeric_time (continuous)\n")
        use_numeric_time <- TRUE
      }, error = function(e) {
        design_preserve <<- model.matrix(~ 0 + cell_type, data = org_meta)
        cat("     Preserving: cell_type (numeric_time model failed)\n")
        use_numeric_time <<- FALSE
      })
    } else {
      cat(sprintf("     Numeric time: only %d values (%.0f%%) - not using\n", 
                  n_valid_time, pct_valid))
      design_preserve <- model.matrix(~ 0 + cell_type, data = org_meta)
      cat("     Preserving: cell_type only\n")
      use_numeric_time <- FALSE
    }
  } else {
    # Fall back to categorical timepoint logic
    has_timepoint <- "timepoint" %in% colnames(org_meta) && 
                     sum(!is.na(org_meta$timepoint)) > 0 &&
                     length(unique(org_meta$timepoint[!is.na(org_meta$timepoint)])) > 1
    
    if (has_timepoint) {
      tp_valid <- !is.na(org_meta$timepoint) & org_meta$timepoint != "unknown"
      pct_valid <- sum(tp_valid) / nrow(org_meta) * 100
      n_timepoints <- length(unique(org_meta$timepoint[tp_valid]))
      
      cat(sprintf("     Categorical timepoint: %d unique values, %.0f%% coverage\n", n_timepoints, pct_valid))
      
      if (pct_valid >= 50) {
        org_meta$timepoint <- droplevels(org_meta$timepoint)
        tp_ds_table <- table(org_meta$dataset, org_meta$timepoint)
        tp_in_multiple_ds <- colSums(tp_ds_table > 0) > 1
        n_shared_tps <- sum(tp_in_multiple_ds)
        
        if (n_shared_tps > 0) {
          cat(sprintf("     %d timepoints shared across datasets - CAN preserve\n", n_shared_tps))
          tryCatch({
            design_preserve <- model.matrix(~ 0 + cell_type + timepoint, data = org_meta)
            cat("     Preserving: cell_type + timepoint (categorical)\n")
          }, error = function(e) {
            design_preserve <<- model.matrix(~ 0 + cell_type, data = org_meta)
            cat("     Preserving: cell_type (timepoint model failed)\n")
          })
        } else {
          cat("     WARNING: Timepoints FULLY CONFOUNDED with dataset!\n")
          if (SKIP_IF_TIMEPOINT_CONFOUNDED) {
            cat("     SKIPPING batch correction (SKIP_IF_TIMEPOINT_CONFOUNDED=TRUE)\n")
            next
          }
          design_preserve <- model.matrix(~ 0 + cell_type, data = org_meta)
          cat("     Preserving: cell_type only\n")
        }
      } else {
        design_preserve <- model.matrix(~ 0 + cell_type, data = org_meta)
        cat("     Preserving: cell_type (timepoint coverage too low)\n")
      }
    } else {
      design_preserve <- model.matrix(~ 0 + cell_type, data = org_meta)
      cat("     Preserving: cell_type (no timepoint variation)\n")
    }
  }
  
  # Check for confounding between batch and cell_type
  batch_ct_table <- table(org_meta$dataset, org_meta$cell_type)
  confounded <- any(colSums(batch_ct_table > 0) == 1)  # Any cell type in only 1 batch
  
  if (confounded) {
    cat("     Warning: Some cell types only in 1 batch - adjusting design\n")
    # Use simpler design or skip problematic cell types
    # For now, use intercept-only design for preservation
    design_preserve <- model.matrix(~ 1, data = org_meta)
  }
  
  # Try ComBat first
  tryCatch({
    # ComBat on expression matrix
    corrected_org <- ComBat(
      dat = org_expr,
      batch = org_meta$dataset,
      mod = design_preserve,
      par.prior = TRUE,
      prior.plots = FALSE
    )
    
    cat("     ComBat successful\n")
    corrected_expr[, org_mask] <- corrected_org
    
  }, error = function(e) {
    cat("     ComBat failed:", conditionMessage(e), "\n")
    cat("     Falling back to limma::removeBatchEffect\n")
    
    # Fallback to limma
    tryCatch({
      corrected_org <- removeBatchEffect(
        org_expr,
        batch = org_meta$dataset,
        design = design_preserve
      )
      corrected_expr[, org_mask] <<- corrected_org
      cat("     limma successful\n")
      
    }, error = function(e2) {
      cat("     limma also failed:", conditionMessage(e2), "\n")
      cat("     Keeping original values for this subset\n")
    })
  })
  
  }  # End sample_types loop
}  # End organisms loop

# =============================================================================
# SAVE CORRECTED DATA
# =============================================================================

cat("\n5. Saving corrected expression...\n")

corrected_df <- as.data.frame(t(corrected_expr))
write.csv(corrected_df, 
          file.path(EXCHANGE_DIR, "corrected_expression_within_organism.csv"),
          row.names = TRUE)
cat("   Saved: corrected_expression_within_organism.csv\n")

# Also save as main output
file.copy(
  file.path(EXCHANGE_DIR, "corrected_expression_within_organism.csv"),
  file.path(EXCHANGE_DIR, "corrected_expression.csv"),
  overwrite = TRUE
)
cat("   Copied to: corrected_expression.csv\n")

# =============================================================================
# VALIDATE
# =============================================================================

cat("\n6. Validating correction...\n")

# Variance partition - but note we can't include organism as random effect
# since we corrected within organism. Instead, compare dataset variance.

# For validation, we'll check within each multi-dataset organism (and sample type)

for (org in organisms) {
  
  # For Human, validate in_vivo and organoid separately
  if (org == "Human") {
    sample_types_val <- c("in_vivo", "organoid")
  } else {
    sample_types_val <- c("all")
  }
  
  for (stype_val in sample_types_val) {
    
    if (stype_val == "all") {
      val_mask <- meta$organism == org
      val_label <- org
    } else {
      val_mask <- meta$organism == org & meta$sample_type == stype_val
      val_label <- paste0(org, " (", stype_val, ")")
    }
    
    if (sum(val_mask) == 0) next
    
    val_meta <- meta[val_mask, ]
    n_datasets <- length(unique(val_meta$dataset))
    
    if (n_datasets <= 1) next
    
    cat(sprintf("\n   %s (before vs after):\n", val_label))
    
    val_meta$dataset <- droplevels(val_meta$dataset)
    val_meta$cell_type <- droplevels(val_meta$cell_type)
    
    # Check if numeric_time available for this subset (not for organoids)
    has_numeric <- has_numeric_time && stype_val != "organoid" &&
                   sum(!is.na(meta$numeric_time[val_mask])) > 10
    
    # Check if categorical timepoint available
    has_tp <- "timepoint" %in% colnames(val_meta) && 
              sum(!is.na(val_meta$timepoint) & val_meta$timepoint != "unknown") > 10
    
    if (has_numeric) {
      # Use numeric time - but variance partition needs factors, so bin it
      form_vp <- ~ (1|dataset) + (1|cell_type)
      vars_to_show <- c("dataset", "cell_type", "Residuals")
      cat("     (numeric_time used in correction but not in VP validation)\n")
    } else if (has_tp) {
      val_meta$timepoint <- droplevels(val_meta$timepoint)
      form_vp <- ~ (1|dataset) + (1|cell_type) + (1|timepoint)
      vars_to_show <- c("dataset", "cell_type", "timepoint", "Residuals")
    } else {
      form_vp <- ~ (1|dataset) + (1|cell_type)
      vars_to_show <- c("dataset", "cell_type", "Residuals")
    }
    
    # Sample genes
    set.seed(42)
    gene_subset <- sample(1:nrow(vobj$E), min(1000, nrow(vobj$E)))
    
    # Before
    vobj_before <- vobj
    vobj_before$E <- vobj$E[gene_subset, val_mask]
    
    tryCatch({
      vp_before <- fitExtractVarPartModel(
        vobj_before$E,
        form_vp,
        val_meta,
        BPPARAM = MulticoreParam(N_CORES)
      )
      vp_before_mean <- colMeans(vp_before, na.rm = TRUE) * 100
      
      # After
      vobj_after <- vobj
      vobj_after$E <- corrected_expr[gene_subset, val_mask]
      
      vp_after <- fitExtractVarPartModel(
        vobj_after$E,
        form_vp,
        val_meta,
        BPPARAM = MulticoreParam(N_CORES)
      )
      vp_after_mean <- colMeans(vp_after, na.rm = TRUE) * 100
      
      for (var in vars_to_show) {
        if (var %in% names(vp_before_mean)) {
          cat(sprintf("     %-10s %5.1f%% -> %5.1f%% (%+.1f%%)\n", 
                      paste0(var, ":"), 
                      vp_before_mean[var], vp_after_mean[var],
                      vp_after_mean[var] - vp_before_mean[var]))
        }
      }
    }, error = function(e) {
      cat("     Variance partition failed:", conditionMessage(e), "\n")
    })
  }
}

# =============================================================================
# GLOBAL SUMMARY
# =============================================================================

cat("\n", "=", rep("=", 59), "\n", sep = "")
cat("Global Summary\n")
cat("=", rep("=", 59), "\n\n", sep = "")

# Overall variance partition (can include organism now since it wasn't corrected away)
# Include timepoint if available
has_global_tp <- "timepoint" %in% colnames(meta) && 
                 sum(!is.na(meta$timepoint) & meta$timepoint != "unknown") > 50

if (has_global_tp) {
  form_global <- ~ (1|dataset) + (1|cell_type) + (1|organism) + (1|timepoint)
  vars_global <- c("dataset", "cell_type", "organism", "timepoint", "Residuals")
} else {
  form_global <- ~ (1|dataset) + (1|cell_type) + (1|organism)
  vars_global <- c("dataset", "cell_type", "organism", "Residuals")
}

set.seed(42)
gene_subset <- sample(1:nrow(vobj$E), min(2000, nrow(vobj$E)))

# Before
vobj_test <- vobj
vobj_test$E <- vobj$E[gene_subset, ]

vp_global_before <- fitExtractVarPartModel(
  vobj_test$E,
  form_global,
  meta,
  BPPARAM = MulticoreParam(N_CORES)
)
vp_global_before_mean <- colMeans(vp_global_before, na.rm = TRUE) * 100

# After
vobj_test$E <- corrected_expr[gene_subset, ]

vp_global_after <- fitExtractVarPartModel(
  vobj_test$E,
  form_global,
  meta,
  BPPARAM = MulticoreParam(N_CORES)
)
vp_global_after_mean <- colMeans(vp_global_after, na.rm = TRUE) * 100

cat("Variable        Before    After    Change\n")
cat(rep("-", 45), "\n", sep = "")
for (var in vars_global) {
  if (var %in% names(vp_global_before_mean)) {
    cat(sprintf("%-12s   %5.1f%%   %5.1f%%   %+5.1f%%\n",
                var,
                vp_global_before_mean[var],
                vp_global_after_mean[var],
                vp_global_after_mean[var] - vp_global_before_mean[var]))
  }
}

# Save comparison
comparison_df <- data.frame(
  variable = vars_global,
  before = vp_global_before_mean[vars_global],
  after = vp_global_after_mean[vars_global]
)
comparison_df$change <- comparison_df$after - comparison_df$before
write.csv(comparison_df, file.path(EXCHANGE_DIR, "variance_partition_comparison.csv"), row.names = FALSE)

cat("\n   Goal: dataset ↓ (within organism), cell_type, organism & timepoint preserved\n")

cat("\n", "=", rep("=", 59), "\n", sep = "")
cat("Done!\n")
cat("=", rep("=", 59), "\n", sep = "")
