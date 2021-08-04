### Using this Repository:
1. Setup your virtual environment
```angular2html
$ conda env create -f environment.yaml
$ conda activate cs7643-raml
$ pip install -r requirements.txt
```
NOTE: If you are working on a separate branch and make any additional pip installations, be sure to **update the `requirements.txt` file**. 

2. [Sync your Jupyter Kernel up with your environment](https://janakiev.com/blog/jupyter-virtual-envs/)
```angular2html
$ pip install ipykernel
$ python -m ipykernel install --user --name=cs7643-raml
```
3. Launch Jupyter Notebook
```angular2html
$ jupyter notebook
```

### Repository Structure
```markdown
.
├── README.md
├── dataset
│   ├── __init__.py
│   ├── categories.json     <-- subset of super categories & subcategories
│   ├── coco_api_helper.py  <-- functions to help interact with pycocotools
│   ├── coco_data_prep.py   <-- Dataset and DataLoader functions + preprocess
│   ├── coco_labels.txt     <-- text file COCO categories
│   ├── config_dataset.py   
│   └── imgs_by_supercategory.json    <-- COCO image ids by super category
├── environment.yaml
├── notebooks
│   ├── COCO-Data-Exploration.ipynb
│   ├── COCO-Dataset-n-DataLoader.ipynb
│   ├── COCO-Subset-Data.ipynb
│   ├── embeddings
│   │   ├── DenseNet-all-validation-data.ipynb
│   │   ├── Densenet-inspired-embeddings.ipynb
│   │   └── DenseNet-inspired.ipynb
│   └── retrieval
│       ├── CNN_AutoEncoder-Embeddings-ANNOY.ipynb
│       ├── Densenet-Embedding-NearestNeighbors.ipynb
│       ├── DenseNet-Embeddings-ANNOY.ipynb
│       ├── Densenet-Embeddings-FAISS.ipynb
│       ├── Densenet-Inpsired-Embeddings-ANNOY.ipynb
│       ├── Densenet-Inspired-Embedding-NearestNeighbors.ipynb
│       ├── Densenet-Inspired-Embedding-Plotting-and-Eval.ipynb
│       └── ResNet-Embedding-NearestNeighbors.ipynb
├── requirements.txt
└── utils
    ├── __init__.py
    ├── aws_helper.py       <-- helper functions for S3, EC2
    ├── performance_metrics.py        <-- precision/recall functions
    └── plot_utils.py       <-- plotting function helpers
```
