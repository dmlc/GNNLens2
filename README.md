# GNNLens2

GNNLens2 is an interactive visualization tool for graph neural networks (GNN). It allows seamless integration with [deep graph library (DGL)](https://github.com/dmlc/dgl) and can meet your various visualization requirements for presentation, analysis and model explanation. It is an open source version of [GNNLens](https://arxiv.org/abs/2011.11048) with simplification and extension.

## Installation

### Requirements

- [PyTorch](https://pytorch.org/)
- [DGL](https://www.dgl.ai/pages/start.html)
- Flask-CORS

You can install Flask-CORS with

```bash
pip install -U flask-cors
```

### Installation for the latest stable version

```bash
pip install gnnlens
```

### Installation from source

If you want to try experimental features, you can install from source as follows:

```bash
git clone https://github.com/dmlc/GNNLens2.git
cd GNNLens2/python
python setup.py install
```

### Verifying successful installation

Once you have installed the package, you can verify the success of installation with

```python
import gnnlens

print(gnnlens.__version__)
# 0.1.0
```

## Contributors

**HKUST VisLab**: 
