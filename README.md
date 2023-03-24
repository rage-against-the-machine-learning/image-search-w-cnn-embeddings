### Project Abstract
Content based image retrieval (CBIR) helps to find similar images in a large dataset, given a query image. The similarity between the images is computed, which is then used to rank them and return the top matching images. Instead of relying on labels to conduct a search, we seek to use image features learned from deep models to retrieve image embeddings. Subsequently, the embeddings are searched using nearest neighbor algorithms. State of the art approaches augment deep models originally designed for image classification to learn embeddings that can be used to retrieve similar images. We measured the performance of such models on CBIR, using image labels to compute precision and recall as evaluation metrics. Among these approaches included ResNeXt-WSL, a model proposed by Facebook AI Research, and DenseNet.

Our goal is to build a simplified model to generate image embeddings that perform comparably well with these state of the art approaches. To compare to the transfer learning approaches, an Auto Encoder model, a simplified DenseNet model, and a ResNet variant were built and trained on the COCO 2014 dataset. Overall, our results demonstrate that effective CBIR can be accomplished using a variety of image embedding techniques that leverage deep learning, including models that are less complex than state of the art computer vision models.

Please see the report findings here: https://github.com/rage-against-the-machine-learning/image-search-w-cnn-embeddings/blob/main/CS_7643_Project.pdf

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

### Project Group 129 
- Michael Aldridge
- Madhuri Jamma
- Sylvai Tran
- Yufeng Wang


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
├── models
│   ├── densenet_inspired.py          <-- mini-DenseNet architecture
│   └── resnet_modules.py             <-- ResNet-18 architecture
├── notebooks
│   ├── COCO-Data-Exploration.ipynb
│   ├── COCO-Dataset-n-DataLoader.ipynb
│   ├── COCO-Subset-Data.ipynb
│   ├── embeddings           <-- ipynbs for generating embeddings
│   └── retrieval            <-- ipynbs for ANNOY, FAISS, and KNN
├── requirements.txt
└── utils
    ├── __init__.py
    ├── aws_helper.py       <-- helper functions for S3, EC2
    ├── performance_metrics.py        <-- precision/recall functions
    └── plot_utils.py       <-- plotting function helpers
```
