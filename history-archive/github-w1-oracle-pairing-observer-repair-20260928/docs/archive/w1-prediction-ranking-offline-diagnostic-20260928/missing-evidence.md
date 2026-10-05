# Missing evidence

The sealed run did not save candidate-level predictions from seed 7101, 7102,
or 7103, nor their ensemble mean. It saved only aggregate prediction metrics
and checkpoints.

Consequently, this zero-forward analysis cannot compute for the learned method:

- window-centered prediction error;
- pairwise utility-difference error or direction accuracy;
- predicted utility span;
- selected action or whether it selected NOOP;
- learned-versus-transparent same/different-action counts;
- acceptance state and true utility difference in learned disagreement windows;
- parent/repeat regret allocation;
- individual-seed ranking or changes caused by ensembling.

These quantities cannot be reconstructed uniquely from MAE, aggregate regret,
and aggregate Top-1. Loading the saved checkpoints would create new model
forwards and was explicitly excluded from this task.
