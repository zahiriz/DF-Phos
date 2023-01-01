import tensorflow as tf
from tensorflow.keras.models import Sequential


from tensorflow.keras.layers import Dense,Dropout,MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd


from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score

from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import xception

import random
import tensorflow as tf
from tensorflow.keras.applications import xception

print(tf.__version__)
#from plot_keras_history import plot_history


#*************************************************************************integerated Function define **************************************************************************************************
def get_gradients(img_input, top_pred_idx):
    """Computes the gradients of outputs w.r.t input image.
    Args:
        img_input: 4D image tensor
        top_pred_idx: Predicted label for the input image
    Returns:
        Gradients of the predictions w.r.t img_input
    """
    images = tf.cast(img_input, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        preds = model(images)
        #top_class = preds[:, top_pred_idx]
        top_class = top_pred_idx

    grads = tape.gradient(preds, images)
    return grads
def get_integrated_gradients(img_input, top_pred_idx,size, baseline=None, num_steps=50):

    if baseline is None:
        baseline = np.zeros(size).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    # 1. Do interpolation.
    img_input = img_input.astype(np.float32)
    interpolated_image = []
    for step in range(num_steps+1):
        temp = baseline+ (step / num_steps) * (img_input - baseline)
        interpolated_image.append(temp)

    interpolated_image = np.array(interpolated_image).astype(np.float32)

    # 2. Preprocess the interpolated images
    #interpolated_image = xception.preprocess_input(interpolated_image)

    # 3. Get the gradients
    grads = []
    for i, img in enumerate(interpolated_image):
        img = tf.expand_dims(img, axis=0)
        grad = get_gradients(img, top_pred_idx=top_pred_idx)
        shape = grad.numpy().shape
        grads.append(grad[0])
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)
    shape = grads.numpy().shape
    # 4. Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)
    shape = avg_grads.numpy().shape
    # 5. Calculate integrated gradients and return

    integrated_grads = (img_input - baseline) * avg_grads
    shape = integrated_grads.numpy().shape

    return integrated_grads
def random_baseline_integrated_gradients(img_input, top_pred_idx,size, num_steps=50, num_runs=2):
    """Generates a number of random baseline images.
    Args:
        img_input (ndarray): 3D image
        top_pred_idx: Predicted label for the input image
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
        num_runs: number of baseline images to generate
    Returns:
        Averaged integrated gradients for `num_runs` baseline images
    """
    # 1. List to keep track of Integrated Gradients (IG) for all the images
    integrated_grads = []

    # 2. Get the integrated gradients for all the baselines
    for run in range(num_runs):
        baseline = np.random.random(size) * 1
        igrads = get_integrated_gradients(
            img_input=img_input,
            top_pred_idx=top_pred_idx,
            baseline=baseline,
            num_steps=num_steps,
            size=size
        )
        integrated_grads.append(igrads)

    # 3. Return the average integrated gradients for the image
    integrated_grads = tf.convert_to_tensor(integrated_grads)
    return tf.reduce_mean(integrated_grads, axis=0)

def ReadExcelFile(filename,sheetname):

    dataframe = pd.read_csv(filename)

#       nrow = uncorrelated.values.shape[0]
#       colheader = list(uncorrelated.columns.values)
#       PID = rawdata[:, 0]

    ncol = dataframe.values.shape[1]
    rawdata = np.array(dataframe.to_numpy())

    X = rawdata[:, 1:ncol - 1]
    X = X.astype(np.float)
    y = rawdata[:, ncol - 1]
    ProteinID = rawdata[:,0]
    y = y.astype(np.float)
    X = RobustScaler().fit_transform(X)

    return X,y,ProteinID
def AccuracyPerLabel(ProteinID, RealY,TestY):
    missT = 0
    totalT = 0
    missY = 0
    totalY = 0
    missS =0
    totalS = 0

    for i in range(len(RealY)):
        temp = ProteinID[i].split("_")[1]
        if (temp == "T"):
            totalT+=1
            if (RealY[i]!=TestY[i]):
                missT+=1
        elif (temp == "Y"):
            totalY+=1
            if (RealY[i]!=TestY[i]):
                missY+=1
        elif (temp == "S"):
            totalS+=1
            if (RealY[i]!=TestY[i]):
                missS+=1
    ff= [missT/totalT , missS/totalS, missY/totalY]
    return ff

#**************************************************************************** Make 5 cross validation model  ********************************************************************************************

X,Y,ProteinID = ReadExcelFile("PreparedData/dbpaf/cdhit/exceldata/SGAAC.csv","Sheet1")
#Y = np_utils.to_categorical(Y)

n_spilit = 5
skf = StratifiedKFold(n_splits=n_spilit, random_state=random.randint(1,1000) , shuffle=True)
total=[]
accPerLabel = []
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    #y_train = np_utils.to_categorical(y_train)
    #y_test = np_utils.to_categorical(y_test)

    model = Sequential()
    model.add(Dense(20, activation='relu', input_shape=(24,),kernel_regularizer=tf.keras.regularizers.l1(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(30, activation='relu',kernel_regularizer=tf.keras.regularizers.l1(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.output_shape
    model.summary()

    # Model config
    #model.get_config()

    # List all weight tensors
    #model.get_weights()
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy,  metrics=[tf.keras.metrics.Recall(),'acc',tf.keras.metrics.Precision(),tf.keras.metrics.AUC()])
    #model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=40, batch_size=200, verbose=1,validation_data=(X_test,y_test) )

#    performance = model.predict(X_test, y_test)
#    print (performance)

    #pred = model.predict(X_test)
    pred = (model.predict(X_test) >0.5).astype("int32")

    #pred = np.argmax(pred, axis=1)
    #y_test = np.argmax(y_test,axis=1)
    precision =precision_score(y_test, pred)
    accuracy =accuracy_score(y_test, pred)
    recall =recall_score(y_test, pred)
    matt =matthews_corrcoef(y_test, pred)
    f1_score1 =f1_score(y_test, pred)
    temp = [accuracy, precision,recall,f1_score1,matt]
    total.append(temp)

    result = AccuracyPerLabel(ProteinID[test_index],Y[test_index],pred)
    accPerLabel.append(result)
#    keys = history.history.keys()
#    for item in keys :
#        print(item)

    #plot_history(history.history)
    #plt.show()
print("***********************************************Accuracy Per Label*******************************************************")
accPerLabel = np.array(accPerLabel)
mean = np.mean(accPerLabel,axis=0)
print(mean)
print("***********************************************Start feature Selection*******************************************************")
#**************************************************************************** Explainable Result ********************************************************************************************

#Train Model with all data
model.fit(X,Y)
mean = np.mean(total,axis=1)
print(mean)

igrads = []

img = X
for index,img_item in enumerate(img):
    print (X.shape[0]-index)
    orig_img = np.copy(img_item)
    img_processed = tf.cast(xception.preprocess_input(img_item), dtype=tf.float32)
    preds = model.predict(img[index:index+1,:])
    weights = random_baseline_integrated_gradients( orig_img, top_pred_idx=preds, num_steps=50 , num_runs=3,size=24)
    igrads.append(weights)
    #if (index>100):
    #    break
igrads=np.abs(np.array(igrads))
np.save("igradas",igrads)

#********************************************************************** find pvalues for best important features ****************************************************************
pvalues = []
from scipy import stats
for i in range(igrads.shape[1]):
    for j in range(igrads.shape[1]):
        if (i!=j):
            p=stats.ttest_ind(igrads[:,i],igrads[:,j]).pvalue
            temp = [i,j,p]
            pvalues.append(temp)
print(pvalues)
file = open("pvalues.txt","w")
for item in pvalues:
    file.write("{},{},{}\n".format(item[0],item[1],item[2]))
file.close()
#*********************************************************************** Plotting the result *************************************************************************************

labels = np.array(['N_1','N_2','N_3','N_4','N_5','N_6','N_7','N_8','M_1','M_2','M_3','M_4','M_5','M_6','M_7','M_8','C_1','C_2','C_3','C_4','C_5','C_6','C_7','C_8'])
plt.figure(figsize=(20,10),dpi=800)
#plt.subplot(221,)
plt.boxplot(igrads, showmeans=False,labels=labels,showfliers=False )
plt.savefig("barplot.png")

igrads = np.mean(igrads,axis=0)
igrads = np.abs(igrads)

print(igrads.shape)
print (np.max(igrads))
print (np.min(igrads))

#wfinal = [item for sublist in wfinal for item in sublist]
plt.bar(labels,igrads)
plt.savefig("gradients.png")
plt.show()

#import pydot
#import graphviz
#tf.keras.utils.plot_model(model=model,to_file="model.png",show_shapes=True,show_layer_names=True,dpi=800)