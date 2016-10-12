import numpy as np;
import scipy as sp;
import cvxopt as cv;
import copy;
from sklearn.svm import SVC;


def load_data():

    dataSet=np.loadtxt('chunk0.txt');

    return dataSet;

#implement logistic regression:
def initialTheta():
    theta=np.arange(3)*0.0;

    for index in range(0, 3):
        theta[index]=np.random.random()*0.1;

    #print 'This is theta', theta;

    return theta;

def calcHtheta(theta, x):

    part1=np.exp(0-np.dot(theta, x));

    #print 'The exp part is: ',part1;
    hTheta=1/(1+part1);

    return hTheta;

def revise_data_set(dataSet):

    m=len(dataSet);

    newData=copy.deepcopy(dataSet);

    for index in range(0, m):

        if newData[index, -1]!=1:
            newData[index, -1]=0;

    return newData;


def calcGradian(theta, dataSet):

    dataSet=revise_data_set(dataSet);

    length=len(dataSet);

    #create a column of 1's
    column=np.ones([length, 1]);

    dataSet=np.concatenate((column, dataSet), axis=1);

    sum=0;

    for index in range(0, length):

        hTheta=calcHtheta(theta, dataSet[index, 0:-1]);
        y=dataSet[index, -1];
        x=dataSet[index, 0:-1];

        sum=sum+(hTheta-y)*x;

    #print 'The gradiant is', sum;

    return sum;

def findParamether(dataSet):

    dataSet=revise_data_set(dataSet);

    theta0=initialTheta();

    learningRate=1/1000.0;

    newTheta=theta0-learningRate*calcGradian(theta0, dataSet);

    #difference=pow(np.dot(theta0-newTheta, theta0-newTheta), 0.5);

    #oldDifference=1000000;

    flag=False;

    round=0
    #while (difference>0.001 and oldDifference!=difference):
    while (flag==False):

        oldTheta=copy.deepcopy(newTheta);

        newTheta=oldTheta-learningRate*calcGradian(oldTheta, dataSet);

        #difference=pow(np.dot(oldTheta-newTheta, oldTheta-newTheta), 0.5);
        difference=oldTheta-newTheta;

        maxDifference=np.amax(abs(difference));

        #print 'the difference vector is', difference;
        #print 'the max diffenrence is, ', maxDifference;

        if maxDifference<0.1:
            flag=True;

        round+=1;

        print round;
        print 'This is theta', newTheta;
        #print 'Difference is ',difference;

    return newTheta;


def classify(theta, dataSet):

    dataSet=revise_data_set(dataSet);

    newDataset=copy.deepcopy(dataSet);

    length=len(dataSet);

    #create a column of 1's
    column=np.ones([length, 1]);

    dataSet=np.concatenate((column, dataSet), axis=1);

    for index in range(0, len(newDataset)):

        inputX=dataSet[index, 0:-1];

        hTheta=calcHtheta(theta, inputX);

        #print 'This is hTheta:', hTheta;

        if (hTheta>0.5):

            newDataset[index, -1]=1;

        else:
            newDataset[index, -1]=0;

    return newDataset;

#calc MSE_i
#input is a data set and the classified data by a certain classifier
#ouput is MSE_i.
def calc_MSE_i(dataSet, classified_data):

    m=len(dataSet);

    #calc f_c_x
    f_c_X=np.zeros([m, 1]);

    for index in range(0, m):

        if dataSet[index,- 1]==classified_data[index,-1]:

            f_c_X[index, 0]=1.0;

        else:

            f_c_X[index, 0]=0.0;


    #calc MSE_i
    MSE_i=0;

    for index in range(0, m):

        MSE_i=MSE_i+(1-f_c_X[index,0])**2;


    MSE_i=MSE_i/m;

    #print 'This is MSE', MSE_i;

    return MSE_i;


def calc_error_rate(dataSet, classified_data):

    m=len(dataSet);

    #calc f_c_x
    f_c_X=np.zeros([m, 1]);

    for index in range(0, m):

        if dataSet[index,-1]==classified_data[index,-1]:

            f_c_X[index, 0]=0.0;

        else:

            f_c_X[index, 0]=1.0;

    #get the total error
    total_error=np.sum(f_c_X, axis=0);
    total_error=total_error[0];

    #calc rate
    error_tate=total_error/m;

    return error_tate;

#calc w_i
def calc_weight_i(MSE_i, MSE_r):

    w_i=MSE_r-MSE_i;

    return w_i;

#implement the ensemble classifier algorithm

#calc the classifer and weight for current chunk
#input is the current chunk data set, output is the W, w0 and weight_i in the form of a vector.
def calc_para_current_chunk(current_data):

    #calc classifier for current chunk
    theta=findParamether(current_data);

    #classify current chunk
    classified_data=classify(theta, current_data);

    #calc MSE_i for current chunk
    MSE_i=calc_MSE_i(current_data, classified_data);

    #here we assume MSE_r is calculated from uniform distribution, so it's value is 0.25
    weight_i=calc_weight_i(MSE_i, 0.25);

    #return result in the form of vector.
    W_w0_weight_vector=np.array([theta[0], theta[1], theta[2], weight_i]);

    return W_w0_weight_vector;

#input is the current data set and previous K-1 classifiers, the classifiers vector is in the form of
#[W, w0, weight_i], where th weight_i is the old weight.
#output is the updated classifier vector in the same form as the input.
def calc_weights_k_classifiers(classifier_vector, current_data):

    for index in range(0, len(classifier_vector)):

        #get the classifier
        theta=classifier_vector[index, 0:-1];

        #classify current chunk
        classified_data=classify(theta, current_data);

        #calc MSE_i for current chunk
        MSE_i=calc_MSE_i(current_data, classified_data);

        #print 'MSE for this one is', MSE_i;

        #here we assume MSE_r is calculated from uniform distribution, so it's value is 0.25
        weight_i=calc_weight_i(MSE_i, 0.25);

        #update the new weight to a certain classifier
        classifier_vector[index, -1]=weight_i;

    return classifier_vector;

#get the top t classifers.
def find_top_weight_classifiers(classifier_vector, t):

    #sort the vector by weight
    classifier_vector=classifier_vector[classifier_vector[:,-1].argsort()];

    #return the top 't' number of classifiers
    return classifier_vector[-t:, :];


#read one data set at a time stamp, record at most K classifiers.
#t is the number of top weights in the classifiers
def initialize_classifiers(K):

    classifier_vector=[];

    #initialize the process, read K-1 chunks, for example if K = 20, we only read 19 at first.
    for index in range(0, K-1):

        file_name='chunk'+str(index)+'.txt';

        current_data=np.loadtxt(file_name);

        theta_weight_current=calc_para_current_chunk(current_data);

        classifier_vector.append(theta_weight_current);

    theta_weight_vector=np.array(classifier_vector);

    return theta_weight_vector;

#calculate the ensemble classifier using the formula (4)
def calc_ensemble_classifiers(classifier_vector, current_data):

    #calculate f_c_x matrix
    f_c_x=np.zeros([len(current_data), len(classifier_vector)]);

    for index_classifier in range(0, len(classifier_vector)):

        classifier_current=classifier_vector[index_classifier];

        for index_data in range(0, len(current_data)):

            #this will return an np.array with 1-D.
            input_row=current_data[index_data, 0:-1];

            one=np.array([1.0]);

            #1-D array concatenate axis=0
            input_row=np.concatenate((one, input_row), axis=0);

            #find current theta
            current_theta=classifier_current[0:-1];

            discriminate_value=calcHtheta(current_theta, input_row);

            print 'This is the probability calculated by sigmoid function',discriminate_value;

            #update the f_c_x matrix
            f_c_x[index_data, index_classifier]=discriminate_value;

    #print 'The f_c_x matrix is',f_c_x;

    weight_vector=classifier_vector[:,-1];

    weight_sum=np.sum(weight_vector);

    prediction_vector=np.zeros([len(current_data), 1]);

    for index_f_c_x in range(0, len(f_c_x)):

        discriminate_value_fcx=np.dot(f_c_x[index_f_c_x], weight_vector)/weight_sum;

        prediction_vector[index_f_c_x]=discriminate_value_fcx;


    #calculate the return matrix
    current_data_without_weight=current_data[:, 0:-1];


    #print 'Final prediction vector is',prediction_vector;

    #print 'SAHPEs are    ',current_data_without_weight.shape, prediction_vector.shape;
    return_Matrix=np.concatenate((current_data_without_weight, prediction_vector), axis=1);

    return return_Matrix;

def training_by_ensembled_classifier(K, t):

    #initialize the K-1 classifiers
    theta_weight_vectors=initialize_classifiers(K);

    # MSEs=[];
    Error_rate=[];

    round=0;
    #start reading at K-1, for example, if K =20, we start training from the 20th chunk at index 19.
    for index in range(K-1, 100):

        print 'THis is round :',round;

        #read the current chunk
        file_name='chunk'+str(index)+'.txt';

        current_data=np.loadtxt(file_name);

        #calc the parameters for current chunk
        theta_weight_current=calc_para_current_chunk(current_data);
        theta_weight_current=theta_weight_current[np.newaxis];

        #update the weights for previous K-1 chunks

        #update the weights for each classifier with respect to the current data
        theta_weight_vectors_without_crt=calc_weights_k_classifiers(theta_weight_vectors, current_data);

        #combine previous classifiers with current classifier
        theta_weight_vectors_total=np.concatenate((theta_weight_vectors_without_crt, theta_weight_current), axis=0);

        for index_predict in range(0, 10):

            theta_total=theta_weight_vectors_total[index_predict, 0:-1];

            classified_data_x=classify(theta_total, current_data);

            MSE_i_x=calc_MSE_i(current_data, classified_data_x);

            print 'Every MSE for all chunks', MSE_i_x;

        #print 'The weights are', W_w0_weight_vectors_total[:, -1];

        #find the t vectors that have the largest weight.
        theta_weight_vectors_lgt_wt=find_top_weight_classifiers(theta_weight_vectors_total, t);


        for index_predict in range(0, 5):

            theta_new=theta_weight_vectors_lgt_wt[index_predict, 0:-1];

            classified_data=classify(theta_new, current_data);

            MSE_i=calc_MSE_i(current_data, classified_data);

            print 'Every MSE for the top 5 chunks', MSE_i;



        #print 'The largest weights are', W_w0_weight_vectors_lgt_wt[:, -1];

        #calculate the ensembled classifier, get the returned data
        classified_data=calc_ensemble_classifiers(theta_weight_vectors_lgt_wt, current_data);

        #print 'this is current data', current_data;
        #print 'this is classified data',classified_data;

        Error_rate_current_data_new_clsfr=calc_error_rate(current_data, classified_data);

        Error_rate.append(Error_rate_current_data_new_clsfr);

        theta_weight_vectors=np.concatenate((theta_weight_vectors, theta_weight_current), axis=0);

        theta_weight_vectors=theta_weight_vectors[1:,:];


        round+=1;

    MSEs_training_results=np.array(Error_rate);


    return MSEs_training_results;


def training_triditional(K):

    W_w0_weight_vectors=initialize_classifiers(K);

    Error=[];

    round=0;

    for index in range(K-1, 100):

        print 'THis is round :',round;
        #read history data

        #load the current data into the history data to initialize it
        file_name='chunk'+str(index)+'.txt';

        #current_data=np.loadtxt(file_name);
        history_data=np.loadtxt(file_name);

        #print 'This is shape of history data',history_data.shape;
        #print 'This is index ',index;

        print 'History data is initialized'

        for index_history in range(index-K+1, index):

            file_name='chunk'+str(index_history)+'.txt';

            old_data=np.loadtxt(file_name);

            history_data=np.concatenate((history_data, old_data), axis=0);

            print 'chunk', index_history, 'has been read'

        #print 'The first row of the first chunk is', history_data[100,:];

        #print 'This is shape of history data again',history_data.shape;
        #print 'This is index again',index;

        #calc parameters for history data
        theta=findParamether(history_data);

        print 'theta is found'

        #classify for history data
        classified_history_data=classify(theta, history_data);

        print 'history data is classified'

        Error_rate=calc_error_rate(history_data, classified_history_data);

        print 'Error rate is calculated'

        Error.append(Error_rate);

        round+=1;

    MSEs_training_results=np.array(Error);

    return MSEs_training_results;

def run_ensemble():

    MSEs_ensembled_classifier=training_by_ensembled_classifier(10, 5);
    print 'This is trained by ensembled_classifier',MSEs_ensembled_classifier;
    print 'This is average for ensembled classifier', np.average(MSEs_ensembled_classifier);

    return;

def run_triditional():

    MSEs_triditional_classifier=training_triditional(10);


    print 'This is trained by triditional classifier',MSEs_triditional_classifier;
    print 'This is average for triditional  classifier', np.average(MSEs_triditional_classifier);

    return;



run_ensemble();
#run_triditional();

#unitest_calc_MSE_i();


