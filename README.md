# 🧪 Hypergraph Neural Networks for Molecular Property Prediction 🧬

This repository contains the implementation of two distinct hypergraph neural network approaches for molecular property prediction: a k-clique-based dense subgraph method and a functional group-oriented hypergraph construction technique. 🔬

## 📚 Overview

Both models leverage hypergraph structures to capture higher-order molecular interactions essential for accurate toxicity prediction. 🔍 Traditional Graph Neural Networks (GNNs) are limited to pairwise interactions, while our hypergraph approaches can model complex multi-atom interactions that are crucial for predicting molecular properties. 🧠

## ✨ Key Features

### 🔷 K-Clique-Based Model
- Uses dense subgraphs (k-cliques) within molecular structures as hyperedges
- Represents molecules as a heterogeneous hypergraph with three distinct node types: atoms, functional groups, and cliques
- Utilizes heterogeneous message passing layers for information propagation across different node and edge types
- Implements global attention pooling to generate fixed-size representations for entire molecules

### 🔶 Functional Group-Oriented Model
- Constructs hyperedges using recognized functional groups, ring systems, and pharmacophore patterns
- Identifies 18 common functional groups and structural motifs using SMARTS pattern matching
- Extracts ring information to identify both aromatic and aliphatic rings
- Uses a bipartite representation to connect atoms with their respective hyperedges

## 📊 Dataset

Both models were evaluated on the androgen receptor (NR-AR) pathway from the Tox21 dataset. The Tox21 dataset contains experimental activity data for approximately 10,000 compounds tested against the androgen receptor.

## 📈 Performance

The k-clique-based model achieved an ROC AUC of 0.894, outperforming the functional group-oriented approach which attained 0.825 AUC. The k-clique model showed superior performance across all evaluation metrics, including PR AUC, accuracy, sensitivity/recall, and precision.

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/username/hypergraph-molecular-prediction.git
cd hypergraph-molecular-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### 🔄 Data Preprocessing

```python
# Example of data preprocessing
from src.preprocessing import preprocess_tox21

# Preprocess the Tox21 NR-AR dataset
data = preprocess_tox21(target="NR-AR")
```

### K-Clique-Based Model

```python
# Example of using the k-clique-based model
from src.models import KCliqueHGNN
from src.hypergraph_construction import build_kclique_hypergraph

# Build a k-clique hypergraph from molecular data
hypergraph = build_kclique_hypergraph(molecule, k=3)

# Initialize and train the model
model = KCliqueHGNN()
model.train(train_data, val_data, epochs=100)

# Make predictions
predictions = model.predict(test_data)
```

### Functional Group-Oriented Model

```python
# Example of using the functional group-oriented model
from src.models import FunctionalGroupHGNN
from src.hypergraph_construction import build_fg_hypergraph

# Build a functional group hypergraph from molecular data
hypergraph = build_fg_hypergraph(molecule, num_patterns=18)

# Initialize and train the model
model = FunctionalGroupHGNN()
model.train(train_data, val_data, epochs=100)

# Make predictions
predictions = model.predict(test_data)
```

## 🔍 Findings

Our comparative analysis revealed that:

1. The method of hyperedge construction significantly impacts model effectiveness
2. The k-clique model better captures pharmacophore patterns and structural motifs relevant to receptor binding
3. The functional group-oriented approach offers computational advantages with a more straightforward implementation
4. The optimal clique size for the k-clique model was k=3, suggesting that triplet-based interactions capture an ideal level of molecular substructure

## 💪 Computational Requirements

The k-clique model requires more computational resources, particularly for preprocessing (k-clique detection), which takes 1.82x longer than the functional group model. The increased preprocessing cost stems from the computational complexity of k-clique detection, which scales exponentially with clique size in the worst case.

## 🚀 Future Directions

- Adaptive hyperedge construction that determines the appropriate clique size based on local molecular structure
- Learned hyperedge selection using attention mechanisms or graph neural networks
- Integration of 3D conformational information to capture spatial arrangements of atoms
- Multi-task learning to simultaneously predict activity across multiple related targets

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

1. Huang, R., Xia, M., Nguyen, D. T., Zhao, T., Sakamuru, S., Zhao, J., ... & Simeonov, A. (2016). Tox21Challenge to build predictive models of nuclear receptor and stress response pathways as mediated by exposure to environmental chemicals and drugs. Frontiers in Environmental Science, 3, 85.

2. Feng, Y., You, H., Zhang, Z., Ji, R., & Gao, Y. (2019). Hypergraph neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, No. 01, pp. 3558-3565).

3. Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., ... & Pande, V. (2018). MoleculeNet: A benchmark for molecular machine learning. Chemical Science, 9(2), 513-530.
