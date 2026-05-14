# NSK Stage 2 — Knowledge Graph Embedder

A graph autoencoder for compact, learnable representations of knowledge graphs in distributed multi-agent systems.

**Status:** Code release pending publication. See [Code release](#code-release) below.

---

## Overview

NSK Stage 2 is the neural-embedding component of the **Neuro-Symbolic Knowledge (NSK) pipeline** for distributed multi-agent knowledge sharing. It encodes a symbolic knowledge graph into a compact 32-dimensional vector that agents can transmit cheaply, while a paired decoder reconstructs structural and relational information sufficient for downstream merging.

The system is designed for bandwidth-constrained settings where agents must share knowledge without centralised infrastructure — e.g. swarm robotics, distributed sensor networks, and collaborative autonomous systems.

## Architecture

The embedder is implemented as a graph autoencoder:

```
Knowledge graph G  ──►  GATv2 encoder  ──►  z ∈ ℝ³²  ──►  Dual decoder  ──►  (edges, relations)
```

**Encoder** — relation-aware GATv2 with mean pooling over node representations.

**Decoder** — dual-objective:
- Edge existence (binary cross-entropy)
- Relation type classification over 237 relations (cross-entropy)

**Total parameters:** 51,661.

## Training results

Trained on FB15k-237 ego-graphs (4,530 valid 2-hop subgraphs, 70/15/15 split).

| Metric | Epoch 1 | Epoch 100 | Change |
|---|---|---|---|
| Train loss (total) | 2.858 | 0.715 | −75% |
| Validation loss | 1.461 | 0.691 | −53% |
| Mean off-diagonal cosine | — | 0.3308 | within target range |

Training completed in ~12.5 h on consumer-grade hardware (CPU fallback, GTX 1050).

## Reproducibility

All sampling and training use fixed seeds for deterministic results. Complete training and evaluation pipeline runs on consumer hardware with no GPU requirement.

## Code release

The full source code, trained model checkpoints, and reproduction scripts will be released alongside publication of the accompanying papers (see [Citation](#citation)). This repository currently serves as a public reference point for the project's architecture, training results, and licensing terms.

For early access or collaboration enquiries, please contact the authors directly (see [Contact](#contact)).

## Citation

If you reference this work, please cite:

```bibtex
@article{ali2026nsk,
  title  = {Neuro-Symbolic Knowledge Graph Compression and Fusion for
            Distributed Multi-Agent Systems},
  author = {Ali, A. and Viksnin, I. I.},
  year   = {2026},
  note   = {In preparation}
}

@article{ali2026sensitivity,
  title  = {Heuristic vs. Structural Mechanisms in Knowledge-Graph
            Compression: A Sensitivity Methodology and Its Application
            to a Neuro-Symbolic Pipeline},
  author = {Ali, A. and Viksnin, I. I.},
  year   = {2026},
  note   = {In preparation}
}
```

A `CITATION.cff` file is also provided for GitHub's "Cite this repository" button.

## License

MIT. See [LICENSE](LICENSE).

## Contact

**Alhasan Ali** — PhD candidate, Department of Computer Science
St. Petersburg Electrotechnical University "LETI"
Email: Aliyossefalhasan@gmail.com

**Supervisor: Viksnin I. I.** — Email: wixnin@mail.ru
