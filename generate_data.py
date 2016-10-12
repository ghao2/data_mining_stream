import numpy as np;
import scipy as sp;
import math;
import matplotlib.pyplot as pl;


#create discriminate function, create a0, a1, and a2.
#the input varables are theta, t, and s, where theta represents the angle(degree) between the norm and the x cordinate
#and it is used to determine the direction of the norm, t is the magnitude of changes on a0, by which we control
#the drifting speed of the model, s is also the moving direcion of the line, which only sports two numbers, -1 or +1.
#initial value of a0 is (a1+a2)/2
#initial value of a1 and a2 are normalized.

def create_dis_func_para(theta, t, s):

    #transform the degree value into radian value
    input_angle=math.radians(theta);

    #calculate a1 and a2
    a2=math.sin(input_angle)**2;

    a1=math.cos(input_angle)**2;

    a0=0-(a1+a2)/2.0;

    #add on the parameters
    a0=a0+a0*t*s*1.0;

    return a0, a1, a2;


def unitest_create_dis_func_para():

    create_dis_func_para(4.0, 1, 1);

    return;

#create data chunks.
#the data for a specific discriminate function is distributed equally in the two sides of the function.
#each sides owns 100 data points.
#we also allow for a noise rate of the data, which is represented as p, it is of a certain percentage, for example 5%.
#d is the distance from the mean to the discriminate function
#variance is the variance of the gaussian distribution
#noise_rate is the percentage of the noise data point

def create_data_for_disc_func(a0, a1, a2, d, variance, noise_rate):

    #calc the mean and varance for the data sets

    #calc the line function for the line that pass origin and perpendicular to our discriminate function.
    #[a1_prime, a2_prime] is the norm of this line
    a1_prime=0-a2;
    a2_prime=a1;

    #we define the distance between the mean of the dataset and the discriminate function as d.

    #Then we define two parallel lines that respectively cross the means in two side of the discriminate function
    #and being parallel to the discriminate funciton.

    #after this, we can solve for the two means of our 2-D Gaussian Distribution.

    #calc the cross point position vector for dataset which is the area where a0+a1x1+a2x2>0;

    #function for first imaginary line:
    a1_positive=a1;
    a2_positive=a2;
    a0_positive=a0+d;

    A_positive=np.array([[a1_prime, a2_prime],
                [a1_positive, a2_positive]]);
    b_positive=np.array([[0],
                [0-a0_positive]]);

    mean_vector_positive=np.linalg.solve(A_positive,b_positive);

    #calc the cross point position vector for dataset which is the area where a0+a1x1+a2x2<0;

    #function for second imaginary line:
    a1_negative=a1;
    a2_negative=a2;
    a0_negative=a0-d;

    A_negative=np.array([[a1_prime, a2_prime],
                [a1_negative, a2_negative]]);
    b_negative=np.array([[0],
                [0-a0_negative]]);

    mean_vector_negative=np.linalg.solve(A_negative,b_negative);

    print mean_vector_negative, '\n',mean_vector_positive;

    #generate dataSet positive

    #calc sample size

    sample_size=int(50*(1-noise_rate));

    x_positive=np.random.normal(mean_vector_positive[0,0], variance, sample_size);
    y_positive=np.random.normal(mean_vector_positive[1,0], variance, sample_size);
    x_positive=x_positive[np.newaxis].T;
    y_positive=y_positive[np.newaxis].T;

    data_pos_normal=np.concatenate((x_positive, y_positive), axis=1);

    #calc noise size:
    noise_size=int(50*noise_rate);


    #generate noise whose true value is negative but in positive side
    x_noise_nega_in_pos=np.random.normal(mean_vector_negative[0,0], variance, noise_size);
    y_noise_nega_in_pos=np.random.normal(mean_vector_negative[1,0], variance, noise_size);
    x_noise_nega_in_pos=x_noise_nega_in_pos[np.newaxis].T;
    y_noise_nega_in_pos=y_noise_nega_in_pos[np.newaxis].T;

    data_pos_noise=np.concatenate((x_noise_nega_in_pos, y_noise_nega_in_pos), axis=1);

    dataSet_pos=np.concatenate((data_pos_normal, data_pos_noise), axis=0);

    #label dataSet pos
    m=len(dataSet_pos);

    one_pos=np.ones([m,1])*(-1.0);

    dataSet_pos=np.concatenate((dataSet_pos, one_pos), axis=1);


    #generate dataset negative
    x_negative=np.random.normal(mean_vector_negative[0,0], variance, sample_size);
    y_negative=np.random.normal(mean_vector_negative[1,0], variance, sample_size);
    x_negative=x_negative[np.newaxis].T;
    y_negative=y_negative[np.newaxis].T;

    data_nega_normal=np.concatenate((x_negative, y_negative), axis=1);

    #generate noise whose true value is positive but in negative side
    x_noise_pos_in_nega=np.random.normal(mean_vector_positive[0,0], variance, noise_size);
    y_noise_pos_in_nega=np.random.normal(mean_vector_positive[1,0], variance, noise_size);
    x_noise_pos_in_nega=x_noise_pos_in_nega[np.newaxis].T;
    y_noise_pos_in_nega=y_noise_pos_in_nega[np.newaxis].T;


    data_nega_noise=np.concatenate((x_noise_pos_in_nega, y_noise_pos_in_nega), axis=1);

    dataSet_neg=np.concatenate((data_nega_normal, data_nega_noise), axis=0);

    #label dataSet pos
    m=len(dataSet_pos);

    one_neg=np.ones([m,1])*1.0;

    dataSet_neg=np.concatenate((dataSet_neg, one_neg), axis=1);

    #combine all of the two sets
    whole_set=np.concatenate((dataSet_pos, dataSet_neg), axis=0);

    return whole_set;

def write_dataFile():

    for index in range(0, 100, 1):

        index=index/10.0;

        a0, a1, a2=create_dis_func_para(45.0, index*10*3.0, 1);

        data=create_data_for_disc_func(a0,a1,a2, 1.0, 0.5, 0.1);

        file_name='chunk'+str(int(index*10))+'.txt';

        f=open(file_name, 'w');

        for i in range(0, len(data)):

            input_row=str(data[i]);

            input_row=input_row.strip('][');

            input_row=input_row+'\n';

            f.write(input_row);

    return;


a0, a1, a2=create_dis_func_para(45.0, 10.0, 1);

data=create_data_for_disc_func(a0,a1,a2, 1.0, 0.5, 0.1);

pl.plot(data[:, 0], data[:, 1], 'ro');

#pl.show();

write_dataFile();
