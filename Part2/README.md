# Results and Observations

### Batching Efficiency
When running batch size = 1, each epoch takes approximately 25 seconds (mean=25.4, std=6.62, n=10) to run.
When running batch size = 8, each epoch takes approximately 3 seconds (mean=3, std=0.4, n=10) second to run.
When running batch size = 32, each epoch takes approximately 3 seconds (mean=1, std=0,n=10) second to run (granularity was at the second level, so it was rounded to the nearest second)


### Optimizer Performance
When using AdaGrad, the model accuracy couldn't surpass 70% on the validation set of the sst2 dataset. 
When using Adam, the model accuracy surpassed 70% and reached 78% on occasion on the validation set of the sst2 dataset.
The gap in the observed accuracy score was also present when testing with the newsgroups dataset.

### Activation Functions
We also tried using different activation functions (tanh, sigmoid, and ReLU). We noticed a slight degradation with sigmoid and tanh, and noticed that ReLU performed much better than the rest. This is likely due to the fact that ReLU limits the gradient to between 0 and 1, which can help prevent the vanishing gradient problem.

# Commands

## Virtual environment creation

It's highly recommended to use a virtual environment for this assignment.

Virtual environment creation (you may also use venv):

```{sh}
conda create -n cs288a1_310 python=3.10
conda activate cs288a1_310
python -m pip install -r requirements.txt
```

## Train and predict commands

Example command for the original code (subject to change, if additional arguments are added):

```{sh}
python perceptron.py -d newsgroups -f bow
python perceptron.py -d sst2 -f bow
python multilayer_perceptron.py -d newsgroups
```

## Commands to run unittests

It's recommended to ensure that your code passes the unittests before submitting it.
The commands can be run from the root directory of the project.

```{sh}
pytest
pytest tests/test_perceptron.py
pytest tests/test_multilayer_perceptron.py
```

Please do NOT commit any code that changes the following files and directories:

tests/
.github/
pytest.ini

Otherwise, your submission may be flagged by GitHub Classroom autograder.
