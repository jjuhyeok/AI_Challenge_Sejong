import pandas as pd
import numpy as np

import random
import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
submit = pd.read_csv('./submit_sample.csv')

print("====================read_Finish=================")

train.drop(['index', 'Unnamed: 0', 'customerID'], axis = 1, inplace = True)
test.drop(['index', 'Unnamed: 0', 'customerID'], axis = 1, inplace = True)

train.drop(['StreamingTV', 'StreamingMovies', 'InternetService'], axis = 1, inplace = True)
test.drop(['StreamingTV', 'StreamingMovies', 'InternetService'], axis = 1, inplace = True)


from pycaret.classification import *
model = setup(train, target = 'Churn' ,silent = True, session_id = 42, fold = 5)
#best_2 = compare_models(sort = 'acc', n_select=2)
print("====================setup_Finish=================")

gbc = create_model('gbc')


print("====================stacking_Finish=================")
tuned_gbc = tune_model(gbc, optimize = 'acc')
final_model = finalize_model(tuned_gbc)

print("====================finalize_Finish=================")

y_pred = predict_model(final_model, data = test)

print("====================predict_Finish=================")


submit['Churn'] = y_pred['Label']
submit['Churn'][submit['Churn'] == 'No'] = 0
submit['Churn'][submit['Churn'] == 'Yes'] = 1
submit['Churn'] = submit['Churn'].astype(int)
submit.to_csv('submit.csv',index = False)
