To train the model on the dataset follow th commands:
1. Go to the Code folder
2. Run the following command to train the model:
```
python Train.py
```
3. Run the following command to test the model:
```
python Test.py <model_name> <file_name>
```
The model_name should be from ['model_best_train', 'model_best_valid', 'model_final']
The file_name should be from ['train', 'valid', 'test']

4. The models will be stored in models folder and the predictions will be stored in the predictions folder
5. The results folder will contain the losses and accuracies
