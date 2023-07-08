### Code to save DSG_Detr outputs

1. clone their repo: git clone https://github.com/Shengyu-Feng/DSG-DETR.git.
2. In lib/sttran.py file, add global_output also to the entry dictionary.
3. In the DSG-DETR folder create a directory (predcls/sgcls as of my directory) to save the model at each epoch.
4. Edit the path in lines 148 and 202, where we save their output for train and test data.
5. Run the command commented at the last line of train_predcls.py with your dataset path.
6. Same thing for sgcls task.
