
# UrbanLaneGraph Dataset & Benchmark

<div align="center">

<img src="urbanlanegraph_dataset/img/teaser-cover.png">




**The UrbanLaneGraph dataset is a first-of-its-kind dataset for large-scale lane graph estimation from aerial images.**
______________________________________________________________________


<p align="center">
  <a href="http://urbanlanegraph.cs.uni-freiburg.de//">Website</a>
</p>

[![DOI](https://img.shields.io/badge/paper-arxiv-red)](https://arxiv.org/pdf/2302.06175.pdf)
[![DOI](https://img.shields.io/github/issues-closed/jzuern/lanegnn)](https://github.com/jzuern/lanegnn/issues)
[![DOI](https://img.shields.io/badge/lanegraph-awesome-green)]()

______________________________________________________________________

</div>



## UrbanLaneGraph _Dataset_ 

[![DOI](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


To view the download instructions, please see the file [DOWNLOAD.md](urbanlanegraph_dataset/DOWNLOAD.md)


### Dataset processing


Instructions on how to process the dataset into a representation suitable for model training may be found in 
[dataset_preparation/PROCESS_DATA.md](dataset_preparation/PROCESS_DATA.md).


### Model training

Based on a number of requests regarding pre-trained model checkpoints, we provide them [here](https://drive.google.com/drive/folders/1XMSt-wIeZs59-yLgzYPZiA9fzBoo_ZGs?usp=sharing).

The models defined in `methods/` can be trained using the scripts `methods/train_centerline_regression.py` and `methods/train_lanegnn.py`.
For more information, please see [methods/TRAINING.md](methods/TRAINING.md).






### Evaluator

[![DOI](https://img.shields.io/badge/license-MIT-lightgrey)]()

To install the `urbanlanegraph_evaluator` package, find the instructions in 
[urbanlanegraph_evaluator/INSTALL.md](urbanlanegraph_evaluator/INSTALL.md).

Instructions on how to evaluate your model locally using the package may be found in 
[urbanlanegraph_evaluator/EVALUATION_INSTRUCTIONS.md](urbanlanegraph_evaluator/EVALUATION_INSTRUCTIONS.md).


______________________________________________________________________
## UrbanLaneGraph _Benchmark_ on Eval.AI


Coming soon!




### Acknowledgements

We thank the following projects for making their code available:
- [LaneExtraction](https://github.com/songtaohe/LaneExtraction/)
- [apls](https://github.com/CosmiQ/apls)
