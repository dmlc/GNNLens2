## Graph structure

```bash
python graph.py
gnnvis --logdir tutorial_graph
```

## Ground truth and predicted node classes

```bash
python nlabel.py -d X
gnnvis --logdir tutorial_nlabel
```

where `X` is one of `Cora`, `Citeseer`, `Pubmed`.

## Edge weights and attention

```bash
python eweight.py -d X
gnnvis --logdir tutorial_eweight
```

where `X` is one of `Cora`, `Citeseer`, `Pubmed`.
