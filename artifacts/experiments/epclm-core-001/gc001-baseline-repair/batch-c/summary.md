# GC-001 Baseline Repair



ANALYSIS_COMPLETE_REVIEW_PENDING



## A. Engineering

Fresh named arms; immutable historical GC references; strict checkpoint identity and verified return.

## B. Baseline comparison

See baseline-comparison.json; incomplete arms excluded. Flat matching is active-parameter, not compute matching.

## C. Communication

Primary comparison is C1-BROADCAST versus IC-ALL-SELF. Historical entry-only C1 and center-only NO-COMM are separate.

## D. State architecture

Fixed 2048-number cellular/independent/flat states versus T1 bounded token context; KV disabled for reference.

## E. Generation quality

Every fixed prompt/mode is retained. Low repetition/PPL alone is not coherence. Human review remains pending unless real scores are supplied.

## F. Scientific conclusions

Engineering checks and numerical observations are separate from hypotheses. One seed/development-only data establish no architectural superiority. H1–H5 NOT_EVALUATED.

## G. Research decision basis

{
  "scope": "POST_HOC_SINGLE_SEED_DEVELOPMENT_DECISION_SUPPORT",
  "questions": {
    "broadcast_reduces_nll_vs_independent": {
      "status": "INCONCLUSIVE_WITHIN_THIS_RUN",
      "evidence": "statistical_evidence.splits.development_holdout",
      "delta": -0.0006289480746130631,
      "interval_95": [
        -0.0015861831504744611,
        0.0002983633510862453
      ],
      "formal_confirmation": false
    },
    "generation_quality": {
      "status": "UNDETERMINED_PENDING_COMPLETE_BLIND_REVIEW",
      "evidence": "human_evaluation",
      "automatic_metrics_are_not_human_coherence": true
    },
    "state_retains_useful_history": {
      "status": "UNDETERMINED",
      "reason": "NLL, norms and similarity do not identify history retention; matched history deletion/state reset or intervention evidence is absent"
    },
    "state_evolution_degradation": {
      "status": "DESCRIPTIVE_ALERTS_ONLY",
      "evidence": "quality_diagnosis and state-diagnostics.json",
      "causal_attractor_claim": "NOT_ESTABLISHED"
    },
    "architecture_superiority": {
      "status": "UNDETERMINED",
      "reason": "one seed, no independent Test, no compute-matched Transformer comparison"
    }
  },
  "next_experiments": {
    "parameter_matching": {
      "status": "DECIDE_AFTER_REVIEW",
      "existing_flat_match": "active parameter matching already implemented; separate T1 matching requires a new protocol"
    },
    "compute_matching": {
      "status": "DEFERRED",
      "condition": "define measured training/inference compute and repeated-seed design first"
    },
    "longer_transformer_training": {
      "status": "DEFERRED_NO_BUDGET_INCREASE",
      "condition": "review convergence and blind quality first; use a separately approved matched protocol"
    }
  },
  "automatic_training": false,
  "additional_training_tokens": 0,
  "formal_hypotheses": "H1_H5_NOT_EVALUATED"
}

## H. Completion boundary

Analysis is complete; Batch C remains review-pending until at least two reviewers independently rate every blinded sample in every dimension. No training budget increase is authorized.
