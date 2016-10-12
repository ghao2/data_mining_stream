import numpy as np;
import scipy as sp;
import cvxopt as cv;
import copy;
from sklearn.svm import SVC;
from sklearn.linear_model import LogisticRegression;
import matplotlib.pyplot as pl;

#define a class to handle the mean vector for all data sets.
class Mean_vector:

    mean_matrix=np.zeros([100, 2]);


    def __init__(self):

        for index_data in range(0, 100):

            file_name='chunk'+str(index_data)+'.txt';

            data_chunk=np.loadtxt(file_name);

            mean_x1=np.mean(data_chunk[:,0]);

            mean_x2=np.mean(data_chunk[:,1]);

            self.mean_matrix[index_data, 0]=mean_x1;
            self.mean_matrix[index_data, 1]=mean_x2;

            # print 'The mean vector is',index_data, mean_x1, mean_x2;


    def return_mean_of_data_chunk(cls, index):
            # print 'The mean vector is',index_data, mean_x1, mean_x2;

        return cls.mean_matrix[index, :];


def load_data():

    dataSet=np.loadtxt('chunk0.txt');

    return dataSet;

#implement logistic regression:
#input dataset, output coefficients
def findParamether(dataSet):

    dataSet=revise_data_set(dataSet);

    inputX=dataSet[:,0:-1];
    inputY=dataSet[:,-1];

    # instantiate the model (using the default parameters)
    logreg = LogisticRegression();

    # fit the model with data
    logreg.fit(inputX, inputY);

    theta=np.concatenate((logreg.intercept_, logreg.coef_[0]), axis=0);

    return theta;


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

#don't need to pull data set.
def classify(theta, dataSet):

    dataSet=revise_data_set(dataSet);

    newDataset=copy.deepcopy(dataSet);

    length=len(dataSet);

    #create a column of 1's
    column1=np.ones([length, 1]);

    column0=np.zeros([length, 1]);

    newDataset=np.concatenate((column1, newDataset), axis=1);

    newDataset=np.concatenate((newDataset, column0), axis=1);

    for index in range(0, len(newDataset)):

        inputX=newDataset[index, 0:-2];

        hTheta=calcHtheta(theta, inputX);

        # print 'This is hTheta', hTheta;

        newDataset[index, -1]=hTheta;

        #print 'This is hTheta:', hTheta;

    print

    return newDataset;

#calc MSE_i
#input classified data has the hTheta as the last column, not 1 or 0, it is a probability.
def calc_MSE_i(classified_data):
    #print 'Classified data as input', classified_data;
    m=len(classified_data);

    MSE_i=0;

    for index in range(0, m):

        MSE_i=MSE_i+(classified_data[index, -2]-classified_data[index,-1])**2;


    MSE_i=MSE_i/m;

    #print 'This is MSE', MSE_i;

    return MSE_i;


def calc_error_rate(dataSet, classified_data):

    dataSet=revise_data_set(dataSet);

    m=len(dataSet);

    #calc f_c_x
    f_c_X=np.zeros([m, 1]);

    for index in range(0, m):

        if classified_data[index, -1]>0.5:
            data_class=1;
        else:
            data_class=0;

        if dataSet[index,- 1]==data_class:

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

    if w_i<0:

        w_i=0;

    return w_i;


def pull_to_origin(current_data, mean_vector):

    refined_data=copy.deepcopy(current_data);

    for index in range(0, 100):
        refined_data[index, 0:-1]=current_data[index, 0:-1]-mean_vector;

    return refined_data;


#implement the ensemble classifier algorithm

#calc the classifer and weight for current chunk
#input is the current chunk data set, output is the W, w0 and weight_i in the form of a vector.
def calc_para_current_chunk(current_data, chunk_index):

    current_data=copy.deepcopy(current_data);

    mean_vector=Mean_vector.return_mean_of_data_chunk(instance,chunk_index);

    current_data=pull_to_origin(current_data, mean_vector);

    #calc classifier for current chunk
    theta=findParamether(current_data);

    #classify current chunk
    classified_data=classify(theta, current_data);

    #calc error rate of this classifier
    print 'Error rate for current theta is',calc_error_rate(current_data, classified_data);


    #calculation for weight doesn't matter

    #calc MSE_i for current chunk
    MSE_i=calc_MSE_i(classified_data);

    #here we assume MSE_r is calculated from uniform distribution, so it's value is 0.25
    weight_i=calc_weight_i(MSE_i, 0.25);

    #return result in the form of vector.
    theta_weight_vector=np.array([theta[0], theta[1], theta[2], weight_i]);

    return theta_weight_vector;

#input is the current data set and previous K-1 classifiers, the classifiers vector is in the form of
#[W, w0, weight_i], where th weight_i is the old weight.
#output is the updated classifier vector in the same form as the input.
def calc_weights_k_classifiers(classifier_vector, current_data, index_mean):

    start_index_mean=index_mean-19;

    round=0;

    for index in range(0, len(classifier_vector)):

        print 'this is calc weights for k classifiers, in round', round;

        #get the classifier
        theta=classifier_vector[index, 0:-1];


        mean_vector=Mean_vector.return_mean_of_data_chunk(instance,start_index_mean);

        # print 'This is mean vetoer index ', start_index_mean;
        #
        # print 'This is current data before pulled', current_data;

        pulled_data=pull_to_origin(current_data, mean_vector);

        # print 'pulled data is changing? pulled for training classifier chunk ', start_index_mean;
        # print 'pulled by mean', mean_vector;
        # print 'current data', current_data;
        # print 'pulled data is', pulled_data;

        # print 'This is current data after pulled', pulled_data;
        # print 'This is theta used to calculate', theta;
        #classify current chunk
        classified_data=classify(theta, pulled_data);

        # print 'This is classified data',classified_data;

        #calc MSE_i for current chunk
        #the further the classifier, the smaller MSE_i is.
        MSE_i=calc_MSE_i(classified_data);
        #print 'This is calculated MSE_i', MSE_i;
        #print 'MSE for this one is', MSE_i;

        #here we assume MSE_r is calculated from uniform distribution, so it's value is 0.25
        weight_i=calc_weight_i(MSE_i, 0.25);
        #print 'This is calculated weight_i accordingly', weight_i;

        #update the new weight to a certain classifier
        classifier_vector[index, -1]=weight_i;

        start_index_mean+=1;

        round+=1;
    #print 'I want check out the classifier_vector for weight', classifier_vector;

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

    round=0;
    #initialize the process, read K-1 chunks, for example if K = 20, we only read 19 at first.
    for index in range(0, K-1):

        print "This is initialized round", round;

        file_name='chunk'+str(index)+'.txt';

        current_data=np.loadtxt(file_name);

        theta_weight_current=calc_para_current_chunk(current_data, index);

        classifier_vector.append(theta_weight_current);

        round+=1;



    theta_weight_vector=np.array(classifier_vector);

    return theta_weight_vector;


#calculate the ensemble classifier using the formula (4)
#input data is actually test data chunk
#output is classified test data
def calc_ensemble_classifiers(classifier_vector, test_data, index_input):

    test_data=revise_data_set(test_data);
    #print 'this is the classifier vector after input', classifier_vector;
    initial_data=test_data;
    #pull the input test data back to origin
    start_index=index_input-19;


    #print 'this is the test data after pulled', start_index, test_data;
    #calculate f_c_x matrix
    f_c_x=np.zeros([len(test_data), len(classifier_vector)]);

    for index_classifier in range(0, len(classifier_vector)):

        classifier_current=classifier_vector[index_classifier];

        #need to pull test data here
        mean_vector=Mean_vector.return_mean_of_data_chunk(instance,start_index);

        pulled_data=pull_to_origin(test_data, mean_vector);

        for index_data in range(0, len(test_data)):

            #this will return an np.array with 1-D.
            input_row=pulled_data[index_data, 0:-1];

            pulled_data_label=pulled_data[index_data, -1];

            one=np.array([1.0]);

            #1-D array concatenate axis=0
            input_row=np.concatenate((one, input_row), axis=0);

            #find current theta
            current_theta=classifier_current[0:-1];
            # print 'classifier for this f_c_x value is', current_theta;
            # print 'this label is:', pulled_data_label;
            discriminate_value=calcHtheta(current_theta, input_row);
            # print 'calculated discrimitive value is', discriminate_value;
            #discriminate_value=pulled_data_label-discriminate_value;
            # print 'final calculated f_c_x value is', discriminate_value;
            #print 'This is the probability calculated by sigmoid function',discriminate_value;

            #update the f_c_x matrix
            f_c_x[index_data, index_classifier]=discriminate_value;

        start_index+=1;

    #print 'This is f_c_x matrix', f_c_x;

    weight_vector=classifier_vector[:,-1];

    #print 'THis is the weight vector, should be increasing', weight_vector;


    weight_sum=np.sum(weight_vector);

    # print 'weight vector is', weight_vector;
    # print 'weight sum is', weight_sum;

    prediction_vector=np.zeros([len(test_data), 1])*0.0;

    for index_f_c_x in range(0, len(f_c_x)):

        discriminate_value_fcx=np.dot(f_c_x[index_f_c_x], weight_vector)/weight_sum;

        # print 'calculation process is', discriminate_value_fcx,'=', np.dot(f_c_x[index_f_c_x], weight_vector),'/',weight_sum;
        # print '########################';

        prediction_vector[index_f_c_x]=discriminate_value_fcx;


    print 'prediction_vector is', prediction_vector;

    initial_data_without_weight=initial_data[:, 0:-1];
    #print 'Final prediction vector is',prediction_vector;

    return_Matrix=np.concatenate((initial_data_without_weight, prediction_vector), axis=1);

    return return_Matrix;

def training_by_ensembled_classifier(K, t):

    #initialize the K-1 classifiers
    theta_weight_vectors=initialize_classifiers(K);

    # MSEs=[];
    Error_rate=[];

    round=0;

    MSE_variation_matrix=np.zeros([99-K+1, K]);

    #start reading at K-1, for example, if K =20, we start training from the 20th chunk at index 19.
    for index in range(K-1, 99):

        print 'THis is round :',round;

        #read the current chunk
        file_name='chunk'+str(index)+'.txt';


        current_data=np.loadtxt(file_name);


        #read the test data
        file_name='chunk'+str(index+1)+'.txt';

        test_data=np.loadtxt(file_name);


        #calc the parameters for current chunk
        theta_weight_current=calc_para_current_chunk(current_data, index);
        theta_weight_current=theta_weight_current[np.newaxis];

        # print "current data weight is calculated", theta_weight_current;
        # print '#########################################################';

        #update the weights for previous K-1 chunks

        #update the weights for each classifier with respect to the current data
        theta_weight_vectors_without_crt=calc_weights_k_classifiers(theta_weight_vectors, current_data, index);
        # print "All data weight is updated", theta_weight_current;
        # print '#########################################################';
        # print '#########################################################';

        #combine previous classifiers with current classifier
        theta_weight_vectors_total=np.concatenate((theta_weight_vectors_without_crt, theta_weight_current), axis=0);

        start_index=index-K+1;
        for index_predict in range(0, 20):

            theta_total=theta_weight_vectors_total[index_predict, 0:-1];

            # print 'This is theta to time classified data', theta_total;

            mean_vector=Mean_vector.return_mean_of_data_chunk(instance,start_index);

            # print 'This is mean vector', mean_vector;

            pulled_data=pull_to_origin(current_data, mean_vector);

            classified_data_x=classify(theta_total, pulled_data);

            # print "This is classified data", classified_data_x;

            MSE_i_x=calc_MSE_i(classified_data_x);

            MSE_variation_matrix[index-K+1, index_predict]=MSE_i_x;

            start_index+=1;

        #print 'The weights are', theta_weight_vectors_total[:, -1];

        #print 'The largest weights are', W_w0_weight_vectors_lgt_wt[:, -1];

        #calculate the ensembled classifier, get the returned data

        # print 'the input classifier vector before calculate f_c_x', theta_weight_vectors_total;
        # print 'THis is test data before last step', test_data;

        classified_data=calc_ensemble_classifiers(theta_weight_vectors_total, test_data, index);

        #print 'this is current data', current_data;
        #print 'this is classified data',classified_data;


        #print 'This is classified data before last step', classified_data;

        Error_rate_current_data_new_clsfr=calc_error_rate(test_data, classified_data);

        Error_rate.append(Error_rate_current_data_new_clsfr);

        theta_weight_vectors=np.concatenate((theta_weight_vectors, theta_weight_current), axis=0);

        theta_weight_vectors=theta_weight_vectors[1:,:];


        round+=1;

    MSEs_training_results=np.array(Error_rate);

    #output MSE_variation matrix
    #print MSE_variation_matrix;

    write_matrix(MSE_variation_matrix, 'MSE_matrix.txt');

    plot_matrix(MSE_variation_matrix, 'MSE_variation_matrix');

    #output Weights variation matrix
    weight_variation_matrix=0.5-MSE_variation_matrix;

    write_matrix(weight_variation_matrix, 'Weight_matrix.txt');
    plot_matrix(weight_variation_matrix, 'weight_variation_matrix');

    return MSEs_training_results;



def plot_matrix(matrix, file_name):

    for index in range(0, len(matrix)):

        input_y=matrix[index, :];

        input_x=np.arange(len(matrix[0,:]));

        pl.plot(input_x, input_y, 'r-');

    pl.axis([0,20,0,1.0])
    pl.title(file_name);
    pl.show();

    return;

def write_matrix(matrix, file_name):

    f=open(file_name, 'w');

    for index in range(0, len(matrix)):

        input_row=matrix[index, :];

        #to output a line that has more than 120 characters, have to use this function.
        input_row=np.array_str(input_row,max_line_width=1000);

        input_row=input_row.strip('][');

        input_row=input_row+'\n';

        f.write(input_row);

    f.close();

    return;




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

    MSEs_ensembled_classifier=training_by_ensembled_classifier(20, 5);
    print 'This is trained by ensembled_classifier',MSEs_ensembled_classifier;
    print 'This is average for ensembled classifier', np.average(MSEs_ensembled_classifier);

    input_x=np.arange(len(MSEs_ensembled_classifier))*0.1;

    pl.plot(input_x, MSEs_ensembled_classifier, 'r-', label='ensemble classifier');

    return;

def run_triditional():

    MSEs_triditional_classifier=training_triditional(20);


    print 'This is trained by triditional classifier',MSEs_triditional_classifier;
    print 'This is average for triditional  classifier', np.average(MSEs_triditional_classifier);

    input_x=np.arange(len(MSEs_triditional_classifier))*0.1;

    pl.plot(input_x, MSEs_triditional_classifier, 'b-', label='traditional classifier');

    return;

instance=Mean_vector();

# return_mean=instance.return_mean_of_data_chunk(99);
#
# print return_mean;

run_ensemble();
run_triditional();


pl.legend(loc='upper center');
pl.axis([0, 10, 0, 1.0]);
pl.title('Model Incrementation=3.0, k=20');
pl.show();
#unitest_calc_MSE_i();


