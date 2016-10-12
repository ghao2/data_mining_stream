import numpy as np;
import scipy as sp;
import cvxopt as cv;
import copy;


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

    learningRate=0.1;

    newTheta=theta0-learningRate*calcGradian(theta0, dataSet);

    #difference=pow(np.dot(theta0-newTheta, theta0-newTheta), 0.5);

    #oldDifference=1000000;

    flag=False;

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

        if dataSet[index,- 1]==classified_data[index,-1]:

            f_c_X[index, 0]=1.0;

        else:

            f_c_X[index, 0]=0.0;

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
    theta_current=findParamether(current_data);

    #classify current chunk
    classified_data=classify(theta_current, current_data);

    #calc MSE_i for current chunk
    MSE_i=calc_MSE_i(current_data, classified_data);

    #here we assume MSE_r is calculated from uniform distribution, so it's value is 0.25
    weight_i=calc_weight_i(MSE_i, 0.25);

    #return result in the form of vector.
    theta0_theta1_theta2_vector=np.array([theta_current[0], theta_current[1], theta_current[2], weight_i]);

    return theta0_theta1_theta2_vector;

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

        print 'MSE for this one is', MSE_i;

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

        classifier_current=calc_para_current_chunk(current_data);

        classifier_vector.append(classifier_current);

    W_w0_weight_vector=np.array(classifier_vector);

    return W_w0_weight_vector;

#calculate the ensemble classifier using the formula (4)
def calc_ensemble_classifiers(classifier_vector):

    ensembled_classifier=np.arange(len(classifier_vector[0,:]))*0.0;

    for index in range(0, len(classifier_vector)):

        #[W1, W2, W0] times the weight
        #sum each vector
        ensembled_classifier=ensembled_classifier+classifier_vector[index]*classifier_vector[index, -1];

    #calculate the total weight, sum(axis=0) axis=0 means sum by column

    print 'This is the classifier vector',classifier_vector;

    total_weight=np.sum(classifier_vector, axis=0);
    total_weight=total_weight[-1];

    print 'This is the total weight', total_weight;

    #calculate the ensembled classifier
    ensembled_classifier=ensembled_classifier/total_weight;

    #update the weight to be 1
    ensembled_classifier[-1]=1.0;

    return ensembled_classifier;

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

        print 'The weights are', theta_weight_vectors_total[:, -1];

        #find the t vectors that have the largest weight.
        theta_weight_vectors_lgt_wt=find_top_weight_classifiers(theta_weight_vectors_total, t);

        print 'The largest weights are', theta_weight_vectors_lgt_wt[:, -1];

        #calculate the ensembled classifier
        ensembled_classifier=calc_ensemble_classifiers(theta_weight_vectors_lgt_wt);

        #use this new classifier to classify and calculate MSE
        theta_new=ensembled_classifier[0:-1,:];

        classified_current_data_new_clsfr=classify(theta_new, current_data);

        # MSE_current_data_new_clsfr=calc_MSE_i(current_data, classified_current_data_new_clsfr);
        #
        # MSEs.append(MSE_current_data_new_clsfr);

        Error_rate_current_data_new_clsfr=calc_error_rate(current_data, classified_current_data_new_clsfr);

        Error_rate.append(Error_rate_current_data_new_clsfr);

        round+=1;

    MSEs_training_results=np.array(Error_rate);


    return MSEs_training_results;

"""
def training_triditional(K):

    W_w0_weight_vectors=initialize_classifiers(K);

    Error=[];

    round=0;

    for index in range(K-1, 100):

        print 'THis is round :',round;
        #read history data

        #load the current data into the history data to initialize it
        file_name='chunk'+str(index)+'.txt';

        current_data=np.loadtxt(file_name);
        history_data=np.loadtxt(file_name);

        print 'History data is initialized'

        for index_history in range(index-K+1, index+1):

            file_name='chunk'+str(index_history)+'.txt';

            old_data=np.loadtxt(file_name);

            history_data=np.concatenate((history_data, old_data), axis=0);

            print 'chunk', index_history, 'has been read'


        #calc parameters for history data
        alpha=solve_alpha(history_data);

        print 'alpha is solved'

        W=calc_W(history_data, alpha);

        print 'W is solved'

        w0=calc_w0(history_data, alpha, W);

        print 'w0 is solved'

        #classify for history data
        classified_current_data=classify(current_data, W, w0);

        print 'history data is classified'

        Error_rate=calc_error_rate(current_data, classified_current_data);

        print 'Error rate is calculated'

        Error.append(Error_rate);

        round+=1;

    MSEs_training_results=np.array(Error);

    return MSEs_training_results;
"""

MSEs_ensembled_classifier=training_by_ensembled_classifier(20, 5);
#MSEs_triditional_classifier=training_triditional(20);

print 'This is trained by ensembled_classifier',MSEs_ensembled_classifier;

#print 'This is trained by triditional classifier',MSEs_triditional_classifier;




