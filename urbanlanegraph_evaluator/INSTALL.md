Create a conda environment created from the `environment.yml` file. It contains all the dependencies needed to 
run the evaluation code.


```bash
conda env create -f environment.yml
conda activate urbanlanegraph
```

You can also u update the environment if you have already created it before.
```bash
conda env update -f environment.yml --name urbanlanegraph
```


To install the `urbanlanegraph_evaluator` package, please execute the following commands:


```bash
cd urbanlanegraph_evaluator
pip install -e .

# test import 
python -c "import urbanlanegraph_evaluator"
```


## Usage

To test all evaluation metrics, please run the following command:

```bash
cd urbanlanegraph_evaluator
python test_evaluator.py
```