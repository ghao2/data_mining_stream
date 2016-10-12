import numpy as np;
import scipy as sp;
import cvxopt as cv;
import copy;
from sklearn.svm import SVC;

#implement SVM
def SVM_function(dataSet, testData):

    inputX=dataSet[:,0:-1];
    inputY=dataSet[:,-1];

    inputX_test=testData[:,0:-1];
    inputY_test=testData[:,-1][np.newaxis].T;

    print 'input X', inputX;
    print 'input Y', inputY;

    print 'Shape X, Y', inputX.shape, inputY.shape;

    clf=SVC(kernel='linear');

    clf.fit(inputX, inputY);

    prediction=clf.predict(inputX_test);

    print inputY;
    print prediction;
    print 'W1',clf.coef_[0,0];
    print 'W2', clf.coef_[0,1];
    #print 'W2',clf.coef_[1];
    print clf.intercept_;

    print np.dot(clf.coef_, dataSet[0,0:-1])+clf.intercept_;

    prediction=prediction[np.newaxis].T;

    return_matrix=np.concatenate((inputX, prediction), axis=1);

    return return_matrix;



def unitest_SVM_function():
    dataSet=np.loadtxt('chunk38.txt');

    SVM_function(dataSet, dataSet);

    return;

def load_data():

    dataSet=np.loadtxt('chunk38.txt');

    return dataSet;


def solve_alpha(dataset):

    m=len(dataset);

    y=dataset[:, -1][np.newaxis].T;

    X=dataset[:, 0:-1];

    #calculate A
    A=y.T;
    A=cv.matrix(A);

    #calculate P
    P=np.dot(y, y.T)*np.dot(X, X.T);
    P=cv.matrix(P);

    #calculate b
    b=cv.matrix(0.0);

    #calculate q
    q=np.ones([m, 1])*-1.0;
    q=cv.matrix(q);

    #generate G matrix
    G=np.eye(m)*-1.0;
    G=cv.matrix(G);

    #calculate h
    h=np.zeros([m, 1])*0.0;
    h=cv.matrix(h);

    #calculate alpha
    alpha=cv.solvers.qp(P, q, G, h, A, b);

    #convert it back to np.array
    results=np.array(alpha['x']);

    #print results;

    return results;

def calc_W(dataSet, alpha):

    width=len(dataSet[0,:-1]);

    W=np.arange(width)*0.0;

    for index in range(0, len(dataSet)):

        W=W+alpha[index]*dataSet[index, -1]*dataSet[index, :-1];


    print 'W is', W;
    return W;


def calc_w0(dataSet, alpha, W):

    newSet=np.concatenate((dataSet, alpha), axis=1);

    SVM=newSet[newSet[:, -1]>0.0001];

    m=len(SVM);

    w0=0;

    for index in range(0, m):

        w0=w0+SVM[index, -2]-np.dot(W.T, SVM[index, 0:-2]);


    w0=w0/m*1.0;

    print 'This is w0', w0;

    return w0;


def classify(dataSet, W, w0):

    returnMatrix=copy.deepcopy(dataSet);

    for index in range(0, len(dataSet)):

        if np.dot(W, returnMatrix[index, 0:-1])+w0>0:

            returnMatrix[index, -1]=1;

        else:

            returnMatrix[index, -1]=-1;

    #print 'This is training data', dataSet;
    #print 'This is classified data',returnMatrix;


    return returnMatrix;


def confusionMatrix(before, after):

    tp=0;
    tn=0;
    fp=0;
    fn=0;

    length=len(before);

    for index in range(0, length):
        if(before[index, 2]==1):
            if(after[index, 2]==1):
                tp=tp+1;
            else:
                tn=tn+1;
        else:
            if(after[index, 2]==1):
                fp=fp+1;
            else:
                fn=fn+1;

    newArray=np.array([[tp, tn],
                       [fp, fn]]);

    return newArray;

def parameters(confusionMatrix):

    confusionMatrix=confusionMatrix.astype(float);

    precision=confusionMatrix[0,0]/(confusionMatrix[0,0]+confusionMatrix[0,1]);
    recall=confusionMatrix[0,0]/(confusionMatrix[0,0]+confusionMatrix[1,0]);
    accuracy=(confusionMatrix[0,0]+confusionMatrix[1,1])/(confusionMatrix.sum());

    F_measure=2*precision*recall/(precision+recall);

    print 'precision is ', precision;
    print 'recal is ', recall;
    print 'accuracy is ', accuracy;
    print 'F-measure is ', F_measure;


    return precision, recall, F_measure;

#
# data=load_data();
#
# #data=revise_class(data);
#
# result=solve_alpha(data);
#
# W=calc_W(data, result);
#
# w0=calc_w0(data, result, W);
#
# returnMatrix=classify(data, W, w0);
#
# confM=confusionMatrix(data,returnMatrix);

dataSet=np.loadtxt('chunk38.txt');

predictedMatrix=SVM_function(dataSet, dataSet);

confM=confusionMatrix(dataSet, predictedMatrix);

print confM;

parameters(confM);