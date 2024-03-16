# PRSUM
Summarizes pull requests and generates pull requests descriptions using multiple artifacts  

We try to explore whether stacking data hierarchially can help in summarizing pull request descriptions. We use the following artifacts to summarize pull request descriptions:
- Source Code Comments
- Commit Messages
- Issue Titles
- Graphs based on Diffs (Based on FIRA Paper)

## Requirements
Create a virtual environment from the given environment.yml file using the command:
```
conda env create -f environment.yml
```
Environment named btp-env will be created. Activate the environment using the command:
```
conda activate btp-env
```

## Dataset
Download the dataset using the command:
```
gdown https://drive.google.com/uc?id=1CWgIaceU2TOXg-xcnT7Z1d5yxlUk4_pI
```
Place it in the Dataset folder.  
This dataset contains 333K Pull requests for Java Projects. The dataset is taken from the Automatic Generation of Pull Request Summarization Paper.

## Running the code
Refer to the readmes in the Dataset and the Code folder for more details.
