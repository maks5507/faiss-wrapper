# FAISS Wrapper

**FAISS Wrapper** is a high-level wrapper for Facebook similarity search system FAISS (https://github.com/facebookresearch/faiss) 

**Author:** Maksim Eremeev (me@maksimeremeev.com)

## Installation

```bash
conda install -c pytorch faiss-cpu

python setup.py build
pip install .
```

## Quick Start

Cosine distance-based similarity search with FW (Python version):

```python
import faiss_wrapper
import numpy as np

fw = faiss_wrapper.FaissWrapper(vec_dimension=3, 
                                transformation=lambda vec: vec / np.linalg.norm(vec), 
                                num_clusters=2, num_probe=10, metric='ip')

fw.train_index({'1': [1, 2, 3], '00': [2, 2, 1]})
fw.add_vectors_to_index({'1': [1, 2, 3], '00': [2, 2, 1]})

fw.search([[1, 0, 0], [0, 0, 1], [0, 1, 0]], 2)

>>> [
       {'00': 0.6666667, '1': 0.26726124},
       {'1': 0.80178374, '00': 0.33333334},
       {'00': 0.6666667, '1': 0.5345225}
    ]
```
