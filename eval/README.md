# Evaluation

The harness and the eval set. Method: [`../docs/EVALUATION.md`](../docs/EVALUATION.md).

```
eval/
├── queries/   ← committed: the hand-written eval queries + expected filters
├── labels/    ← GITIGNORED: relevance labels (listing IDs from one snapshot)
└── runs/      ← GITIGNORED: report output from each named run
```

**Why labels are not committed.** Relevance labels reference listing IDs from one
specific Inside Airbnb snapshot. They are not portable across snapshots, and they
are derived from data this repository may not republish. The queries themselves
*are* committed — they are original text.

**Why runs are not committed.** Reports are published deliberately, by summarizing
a named run into the README, rather than by dumping every experiment into git.

Nothing here is populated until M5 (T-050 onward).
