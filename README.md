# Perceptual Confidence Reflects Bayesian Causal Inference in Multisensory Integration

The repository accompanies a upcoming manuscript. It includes code used for model fitting and analysis/plotting.

## The repository
- model_fitting.py 
- scripts/modelClassGPU.py - for model simulation 
- scripts/experiments.py - define the preresquites for model fitting (loss function, handling data...)  
- scripts/computeQLL.py for computing the neg log liklihood 
- scripts/run_optimizers.py for paramter estimation
- scripts/evaluators.py for model comparison
- prepare_fit_data.ipynb - bin the behavioural data for quantitle maximum likelihood computation. 
- Model_evaluation_with_beh_data.ipynb - Plotting of the behavioral and model simulation data
- exp_config.csv and m_config.csv - define configuration for models. Each model can be uniquely identified by the combinition of exp_name and m_id.
- model_code_book.xlsx - explaination for the model configurations. 


## Usage
1. define the modeling experiment entry in exp_config.csv with a unique exp_name
2. run model_fitting.py for each subject and each model.