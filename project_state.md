#need restructure processing code into a module
->preproccessing code needs to be sepearated and made into a module  can be include in data_handling think so

#already created   

-> evalution module : all evalution methods seperated based on types and use like graphical ,mathematical,and abstracted based on use like for vae and tree based ...also saving is included in this which is to be modified later
-> model_src  : model definition

agy --conversation=49b4f532-73f2-44d1-8e7f-1c2f0127d2be




## Suggested Design

  ### 1. Config-Driven Experiments
  Instead of hardcoding hyperparameters inside each experiment script, each
  experiment gets a YAML config file. This way you can re-run the same
  experiment with different settings without touching the code.
    experiments/
    ├── configs/
    │   ├── experiment1.yaml      ← hyperparams, model choices, dataset paths
    │   ├── experiment2.yaml
    │   └── experiment3.yaml
    ├── experiment1.py            ← reads its config, runs the logic
    ├── experiment2.py
    └── run.py                    ← single entry point: python run.py --exp
  experiment1
  ### 2. Versioned Results with Timestamps

  Every time you run an experiment, results go into a timestamped subfolder.
  This way you never accidentally overwrite old results when you re-run with
  different configs.

    results/
    ├── experiment1/
    │   ├── run_2026-09-25_21-00/     ← first run
    │   │   ├── config.yaml           ← copy of config used (reproducibility)
    │   │   ├── vae_diagnostics/
    │   │   ├── vae_gradient_boosting/
    │   │   └── gradient_boosting/
    │   └── run_2026-09-26_10-00/     ← second run with different params
    │       ├── config.yaml
    │       └── ...
    ├── experiment2/
    └── ...

  ### 3. Single Entry Point (run.py)

    # python run.py --exp experiment1
    # python run.py --exp experiment2

  This calls the right experiment script and automatically creates the
  timestamped results folder. No need to manually manage paths inside each
  experiment.

  ### 4. Base Experiment Class

  Each experiment inherits from a BaseExperiment class that handles the
  boilerplate (loading config, creating result dirs, saving a copy of the
  config used):

    experiments/
    └── base_experiment.py    ← BaseExperiment class
  ──────
  ## How it would look in code

    # python run.py --exp experiment1
    
    BaseExperiment
        ├── loads configs/experiment1.yaml
        ├── creates results/experiment1/run_2026-09-25_21-00/
        ├── copies config.yaml into that folder
        └── calls experiment1.run()
  ──────
  ## What goes in a config file

    # configs/experiment1.yaml
    experiment: experiment1
    dataset:
      train: dataset/processed_dataset/processed_train.csv
      test:  dataset/processed_dataset/processed_test.csv
    
    vae:
      latent_dim: 8
      epochs: 100
      batch_size: 64
      lr: 0.001
      beta: 0.005
    
    gradient_boosting:
      n_estimators: 200
      max_depth: 4
      random_state: 42
