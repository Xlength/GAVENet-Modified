Build Python environment before executing.
conda create -n gave python==3.10
conda activate gave
pip3 install -r requirement.txt
Then Code is executed in the following order:
1)download the GAVE dataset through GAVE Challenge . Put the dataset in the ./Data. 


2)
*Training
All training code can be found through the entrance of training script train.py, and the configuration file, with all the hyperparameters and command line arguments, is cfg.py.
*Get predictions
After the model trained, the predictions can be generated using get_pred.py. 
*Evaluation
Predictions with Groundtruth can be valuated using test.py, The evalutation result will be saved to json file under --results_dir.