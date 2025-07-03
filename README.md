# stat5241_team_7
## Predictng Congressional Votes: 
### Modeling Party Alignment, Lobbying Influence, and Subject Ideology on Roll Call Behacior


Repository Structure:

data (module): stores raw data and model pipeline caches

data_loaders (modules):

    datasets.py: entry points to the individual datasets that we trained on

    rollcall.py: clean, aggregate, and merge raw datasets

models (module):

    evaluation.py: compute evaluation metrics per model

    models.py: registry of all models, hyper-parameters, and datasets tested

    nn.py: architecture of feed-forward neural network used

notebooks (module):

    EDA: Please consulte EDA_README.md

    Model evaluation: evaluation.ipynb

    exploration, member_info, rollcall_onboarding_v1,  rollcal_onboarding_v2: scratch notebooks used to initially understand and massage the datasets.

main.py: Entrypoint to run all models

