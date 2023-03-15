import warnings
warnings.filterwarnings("ignore")
import pprint
from sklearn.preprocessing import RobustScaler
import numpy as np
import pandas as pd
import os
import glob
import random
from collections import Counter
from sklearn.metrices import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.model_selection import  KFold, train_test_split,StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score, roc_auc_score,confusion_matrix

def countModel(labels):
    newlabels =[]
    for item in labels:
        model_name = item.split("_")[2]
        newlabels.append(model_name)

    model = np.unique(newlabels)
    count = Counter(model)
    print(count)
evaluationName = ['Precision', 'F1Score', 'Accuracy', 'Recall', 'Matt', 'Auc' ,"Specificity"]
def ErrorPerLabel(ProteinID, RealY,TestY):
    performanceDict = {}

    for i in range(len(RealY)):
        aminoAcid = ProteinID[i].split("_")[1]
        species = ProteinID[i].split("_")[2]
        species_amino = species+aminoAcid
        try:
            if (RealY[i] != TestY[i]):
                performanceDict["miss"+aminoAcid]+=1
                performanceDict["miss"+species]+=1
                performanceDict["miss"+species_amino]+=1
            performanceDict["total"+aminoAcid]+=1
            performanceDict["total"+species]+=1
            performanceDict["total"+species_amino]+=1
        except:
            if (RealY[i] != TestY[i]):
                performanceDict["miss"+aminoAcid] = 1
                performanceDict["miss"+species] = 1
                performanceDict["miss"+species_amino] = 1
            performanceDict["total"+aminoAcid]=1
            performanceDict["total"+species]=1
            performanceDict["total"+species_amino]=1
    result = {
                "errorS":performanceDict['missS']/performanceDict['totalS'],
                "errorT":performanceDict['missT']/performanceDict['totalT'],
                "errorY":performanceDict['missY']/performanceDict['totalY'],
                "errorMus":performanceDict['missMus']/performanceDict['totalMus'],
                "errorMusS":performanceDict['missMusS']/performanceDict['totalMusS'],
                "errorMusT":performanceDict['missMusT']/performanceDict['totalMusT'],
                "errorMusY":performanceDict['missMusY']/performanceDict['totalMusY'],
                "errorHomo":performanceDict['missHomo']/performanceDict['totalHomo'],
                "errorHomoS":performanceDict['missHomoS']/performanceDict['totalHomoS'],
                "errorHomoT":performanceDict['missHomoT']/performanceDict['totalHomoT'],
                "errorHomoY":performanceDict['missHomoY']/performanceDict['totalHomoY'],
             }
    #result.update(performanceDict)
    return result

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def RemoveCorrelated( dataframe):
    corr_matrix = dataframe.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = {}
    for i in range(upper.values.shape[0]):
        for j in range(i + 1, upper.values.shape[0]):
            if upper.values[i, j] >= 0.70:
                to_drop[upper.columns[j]] = 1

    uncorrelated_data = dataframe.drop(to_drop.keys(), axis=1)
    return uncorrelated_data
def ReadExcelFile(filename,sheetname=None):
    dataframe = pd.read_csv(filename)
    uncorrelated = RemoveCorrelated(dataframe)
#       nrow = uncorrelated.values.shape[0]
#       colheader = list(uncorrelated.columns.values)
#       PID = rawdata[:, 0]

    ncol = uncorrelated.values.shape[1]
    rawdata = np.array(uncorrelated.to_numpy())
    portein_ids = rawdata[:,0]
    X = rawdata[:, 1:ncol - 1]
    X = X.astype(np.float)
    y = rawdata[:, ncol - 1]

    y = y.astype(np.float)
    X = RobustScaler().fit_transform(X)
    return X,y,portein_ids
def doClassifyCrossValidation(X, y, classifier, proitein_ids = None, nfold=10 ,makeclassifier=False):
    Data = X
    evlP = [[0 for x in range(7)] for YY in range(nfold)]
    k = 0
    kf = KFold(n_splits=nfold, shuffle=False, random_state=2)
    result_per_label = []
    for train_index, test_index in kf.split(X):
        if (makeclassifier == True ):
            classifier = CascadeForestClassifier(random_state=1,verbose=0)
        classifier.fit(Data[train_index], y[train_index])

        y_pred = classifier.predict(Data[test_index])
        y_test = y[test_index]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)

        evlP[k][0] = (precision_score(y_test, y_pred, average='micro'))
        evlP[k][1] = (f1_score(y_test, y_pred, average='macro'))
        evlP[k][2] = (accuracy_score(y_test, y_pred))
        evlP[k][3] = (recall_score(y_test, y_pred, average="weighted"))
        evlP[k][4] = (matthews_corrcoef(y_test, y_pred))
        evlP[k][5] = multiclass_roc_auc_score(y_test, y_pred)
        evlP[k][6] = specificity

        k += 1
        label = ErrorPerLabel(proitein_ids[test_index], y_test, y_pred)
        result_per_label.append(label)
        #pprint.pprint(label)
    import pandas as pd
    df = pd.DataFrame(result_per_label)
    answer = dict(df.mean())
    print("Average :---------------------------------------------------------")
    pprint.pprint(label)


    average = np.matrix(evlP)
    average = average.mean(axis=0)
    average = np.squeeze(np.asarray(average))

    modelparams = pd.DataFrame({'Evaluating Function': evaluationName, 'Values': average})
    return modelparams


def doClassifyTrainAndTest(xTrain,yTrain,xTest,yTest, classifier):
    evlP = np.zeros(7)
    classifier.fit(xTrain, yTrain)
    y_pred = classifier.predict(xTest)
    y_test =yTest

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    evlP[0] = (precision_score(y_test, y_pred, average='micro'))
    evlP[1] = (f1_score(y_test, y_pred, average='macro'))

    evlP[2] = (accuracy_score(y_test, y_pred))
    evlP[3] = (recall_score(y_test, y_pred, average="weighted"))

    evlP[4] = (matthews_corrcoef(y_test, y_pred))
    evlP[5] = multiclass_roc_auc_score(y_test, y_pred)
    evlP[6] = specificity

    modelparams = pd.DataFrame({'Evaluating Function': evaluationName, 'Values': evlP})
    return modelparams,y_pred
windowlen = ['data21','data25','data29','data33','data37']
windowlen = ['data41']
#resultFile = open("result/result.txt")
for dataItem in windowlen:
    feature_xlsx_path = f"F:\Research\Phosphorylation\PreparedData\elm\model\{dataItem}\cdhit\CkSAApair.csv"
    X,Y,protein_Ids = ReadExcelFile(feature_xlsx_path)
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, Y, indices, test_size=0.1, random_state=40)

    classifier = CascadeForestClassifier(random_state=1)
    makeclassifier = True
    cross_result = doClassifyCrossValidation(X_train,y_train,classifier, protein_Ids[idx_train],makeclassifier =True)
    jack_result, y_pred = doClassifyTrainAndTest(X_train,y_train,X_test,y_test,classifier)
    label  = ErrorPerLabel(protein_Ids[idx_test],y_test,y_pred)

    print ("Cross validation result : ")
    print(cross_result)
    print ("Jack nife result : ")
    print(jack_result)
    print ("Error per each label :")
    pprint.pprint(label)
