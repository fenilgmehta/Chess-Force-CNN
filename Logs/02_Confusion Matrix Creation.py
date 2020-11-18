# Confusion Matrix Creation

from step_03a import *
modelCNN = NNBuilder.build("cnn-mg006-be01588-sn000-ep00024-weight-v001.h5")
dataX, dataY = joblib.load("04_pkl_data/complete_kingbase_dataset.pkl")
predicted_y_probabilities = modelCNN.c_predict(dataX)  # shape = (28037072, 2)
joblib.dump(predicted_y_probabilities, filename="TrainDataPredictions.pkl", compress=1)

predictionY = np.zeros((28037072,))
predictionY[predicted_y_probabilities[:,1] >= 0.5] = 1
predictionY[predicted_y_probabilities[:,1] < 0.5] = -1

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

# --------------------------------------------------

# REFER: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
from sklearn.metrics import confusion_matrix
cf_mat = confusion_matrix(y_true=dataY.ravel(), y_pred=predictionY.ravel(), labels=[1, -1])
# In [38]: confusion_matrix(y_actu, y_pred, labels=[1, -1])
# Out[38]: 
# array([[  342395, 15365952],
#        [  227537, 10052627]])

# In [39]: confusion_matrix(y_actu, y_pred, labels=[1, -1], normalize='all')
# Out[39]: 
# array([[0.01317486, 0.59125942],
#        [0.00875529, 0.38681043]])

#                  Predicted
#              +------------+----------+
#              |     -1     |    1     |
#        +-----+------------+----------+
# Acutal | -1  |  10052627  |  227537  |
#        |  1  |  15365952  |  342395  |
#        +-----+------------+----------+
