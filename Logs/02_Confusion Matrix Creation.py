# Confusion Matrix Creation

import math
import joblib
import pandas as pd
import step_02_ScoreNormalizer as step_02
from step_03a_ffnn import *


def one_hot_to_linear(y):
    cols = y.shape[1]
    if cols % 2 == 0:  
        colsBy2 = cols // 2        # even
    else:              
        colsBy2 = (cols + 1) // 2  # odd
    
    ydash = np.zeros((y.shape[0],))
    ydash = np.argmax(y, axis=1) - colsBy2

    # even
    if cols % 2 == 0: ydash[-0.5 < ydash] += 1
    return ydash

# Create the Model object
# modelCNN = NNBuilder.build("cnn-mg006-be01588-sn000-ep00024-weight-v001.h5")
# modelCNN = NNBuilder.build("cnn-mg007-be01588-sn000-ep00009-weight-v001.h5")
modelCNN = NNBuilder.build("cnn-mg007-be01588-sn000-ep00012-weight-v001.h5")

# Load actual dataset
# dataX, dataY = joblib.load("04_pkl_data/complete_kingbase_dataset.pkl")
dataX, dataY = joblib.load("../../Chess-Force-CNN-Dataset/04_pkl_data_combined/all_combined.pkl")

# Normalize the data the same was as it was done during training
# TO UPDATE THIS
dataY = step_02.ScoreNormalizer.normalize_007(dataY)

# Generate Predictions
predicted_y_probabilities = modelCNN.c_predict(dataX)
joblib.dump(predicted_y_probabilities, filename="TrainDataPredictions_cnn-mg007-be01588-sn000-ep00009-weight-v001.pkl", compress=1)

# Generate confusion matrix
# REFER: https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python
y_actu = pd.Series(one_hot_to_linear(dataY), name='Actual')
y_pred = pd.Series(one_hot_to_linear(predicted_y_probabilities), name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)

exit(0)

# In [27]: predictionY[predicted_y_probabilities[:,1] >= 0.5] = 1
#     ...: predictionY[predicted_y_probabilities[:,1] < 0.5] = -1
#     ...: 
# 
# In [28]: np.sum(predictionY == 1)
# Out[28]: 650871
# 
# In [29]: np.sum(predictionY == -1)
# Out[29]: 27386201
# 
# In [30]: 


# In [18]: np.sum(predicted_y_probabilities[:,1] > 0.5)
# Out[18]: 650826
#
# In [19]: np.sum(predicted_y_probabilities[:,1] < 0.5)
# Out[19]: 27386201
#
# In [20]: np.sum(predicted_y_probabilities[:,1] == 0.5)
# Out[20]: 45

# ----------------------------------------------------------------------------------------------------

# MODEL: cnn-mg006-be01588-sn005-ep00024-weight-v001.h5
         
# # REFER: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# from sklearn.metrics import confusion_matrix
# cf_mat = confusion_matrix(y_true=dataY.ravel(), y_pred=predictionY.ravel(), labels=[1, -1])

# In [38]: confusion_matrix(y_actu, y_pred, labels=[1, -1])
# Out[38]: 
# array([[  342395, 15365952],
#        [  227537, 10052627]])

# In [39]: confusion_matrix(y_actu, y_pred, labels=[1, -1], normalize='all')
# Out[39]: 
# array([[0.01317486, 0.59125942],
#        [0.00875529, 0.38681043]])

# MODEL: cnn-mg006-be01588-sn005-ep00024-weight-v001.h5

# Predicted        -ve       +ve
# Actual                                           
# -ve         0.386810  0.008755
# +ve         0.591259  0.013174

#                  Predicted
#              +------------+----------+
#              |     -1     |    1     |
#        +-----+------------+----------+
# Actual | -1  |  10052627  |  227537  |
#        |  1  |  15365952  |  342395  |
#        +-----+------------+----------+



# NOTE: This is less than the new dataset size (i.e. 29037071) because accidentally one file was left when creating the "all_combined.pkl" file previously

# ----------------------------------------------------------------------------------------------------

# NOTE: All the below models use dataset with New len = 2,90,37,071
#                                                      -------------

# ----------------------------------------------------------------------------------------------------

# MODEL: cnn-mg007-be01588-sn006-ep00009-weight-v001.h5

# Predicted        -2        -1         1         2
# Actual                                           
# -1         0.000044  0.000279  0.365916  0.000637
#  1         0.000063  0.000355  0.631633  0.001073

# Predicted        -ve       +ve
# Actual                                           
# -ve         0.000323  0.366553
# +ve         0.000418  0.632706

# ----------------------------------------------------------------------------------------------------

# MODEL: cnn-mg007-be01588-sn007-ep00012-weight-v001.h5

# Predicted       -2        -1    1       2
# Actual                                   
# -2         1838794   8248576  321  514701
# -1            5387     41050    0    4174
#  1          426580   1666858  105   84964
#  2         2393010  13088360  451  723740

# Predicted        -2        -1         1         2
# Actual                                   
# -2         0.063325  0.284071  0.000011  0.017726
# -1         0.000186  0.001414  0.000000  0.000144
#  1         0.014691  0.057404  0.000004  0.002926
#  2         0.082412  0.450747  0.000016  0.024925

# Predicted        -ve       +ve
# Actual                                   
# -ve         0.348996  0.017880
# +ve         0.605254  0.027871

"""
a=''
for i in list(map(int, a.split())):
    print(f"{i/29037071:.6f}", end='  ')
else: print()


"""
