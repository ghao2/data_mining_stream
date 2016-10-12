import  numpy as np;
from sklearn.linear_model import LogisticRegression
import copy;


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



def calc_error_rate(oldY, newY):

    m=len(oldY);

    #calc f_c_x
    f_c_X=np.zeros([m, 1]);

    for index in range(0, m):

        if oldY[index]==newY[index]:

            f_c_X[index, 0]=0.0;

        else:

            f_c_X[index, 0]=1.0;

    #get the total error
    total_error=np.sum(f_c_X, axis=0);
    total_error=total_error[0];

    #calc rate
    error_tate=total_error/m;

    return error_tate;

for index in range(0, 100):

    file_name='chunk'+str(index)+'.txt';

    data=np.loadtxt(file_name);

    data=revise_data_set(data);

    inputX=data[:,0:-1];
    inputY=data[:,-1];

    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()

    # fit the model with data
    logreg.fit(inputX, inputY)

    print logreg.coef_, logreg.intercept_;

    # predict the response for new observations
    output=logreg.predict(inputX);

    #print data[:,-1];
    #print output;

    print calc_error_rate(data[:, -1], output);

#
# print logreg.coef_[0,0];
# print logreg.intercept_;
#
# theta=np.concatenate((logreg.intercept_, logreg.coef_[0]), axis=0);
#
# inputX=data[80, 0:-1];
#
# one=np.array([1.0]);
#
# inputX=np.concatenate((one, inputX));
#
# hTheta=calcHtheta(theta, inputX);
#
# print hTheta;
