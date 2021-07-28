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
├── data                    <-- hidden directory (.gitignore)
│   ├── interim
│   ├── processed
│   └── raw
├── dataset                 <-- get, prep & transform image data 
│   ├── __init__.py
│   ├── categories.json
│   ├── coco_api_helper.py
│   ├── coco_data_prep.py.  <-- create torch.utils.data Dataset object compatible with DataLoader
│   ├── config_dataset.py
├── docs                    <-- relevant academic papers
│   └── Deep-Image-Retrieval-ASurvey-2101.11282.pdf
├── environment.yaml
├── metrics                 <-- model metrics
│   ├── __init__.py
│   └── similarity_metrics.py
├── notebooks
│   ├── COCO-Data-Exploration.ipynb
│   ├── COCO-Dataset-n-DataLoader.ipynb
│   ├── COCO-Subset-Data.ipynb
│   └── embeddings
│       └── DenseNet.ipynb
├── requirements.txt
└── utils                   <-- misc utility modules
    ├── __init__.py
    └── aws_helper.py       <-- helper functions for S3, EC2
```
