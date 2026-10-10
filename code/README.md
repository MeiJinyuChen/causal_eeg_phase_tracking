# Where the code is

This directory is a leftover from an earlier handoff package; the five-layer layout it
used to describe no longer exists. The real code lives here:

  causal_phase_tracking\causal_cecHT\                causal streaming cecHT core,
                                                     pseudo-trigger, contracts, tests
  causal_phase_tracking\causal_simulation_evaluation\
                                                     offline replay, reference phase,
                                                     metrics, frozen goldens
  realtime_deployment\                               project-level contract, host plugin
                                                     and deployment evidence
  offline_benchmark\                                 isolated offline comparison; runs as
                                                     delivered, see its README
  experiment_evidence\                               runner and results per experiment

Start from the repository-root README.md, or from OVERVIEW_INDEX.txt for the full map.
