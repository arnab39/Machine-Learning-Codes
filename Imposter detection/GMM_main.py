import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn.mixture import GaussianMixture,GMM
import Feature_extraction as fe
from sklearn.metrics import confusion_matrix




'''Function to draw a ellipse about the fitted data for visualisation'''
########################################################################

def draw_ellipse(position, covariance, alpha,clr, ax=None, **kwargs):
	ax = ax or plt.gca()

	'''Convert covariance to principal axes'''
	##########################################
	if covariance.shape == (2, 2):
		U, s, Vt = np.linalg.svd(covariance)
		angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
		#print angle
		width, height = 2 * np.sqrt(s)
	else:
		angle = 0
		width, height = 2 * np.sqrt(covariance)
	'''Drawing the Ellipse'''
	######################### 
	ell=mpl.patches.Ellipse(position, 2*width, 2*height,angle,color=clr, **kwargs)
	ax.add_patch(ell)
	ell.set_alpha(alpha)

##########################################################################





'''Function to train & test data using GMM classifier and returns the accuracy of testing'''
############################################################################################

def GMM_func(X_train, Y_train, X_test, Y_test, n_classes, show_results=False,
								fplt=False,colors='rgbym',select_classifier=2,cov_type='tied'):
	'''To find the mean of the data and pass it as initial mean in the GMM class'''
	###############################################################################
	temp1=X_train[:,0]
	temp2=X_train[:,1]
	temp1=np.reshape(temp1,(X_train.shape[0],1))
	temp2=np.reshape(temp2,(X_train.shape[0],1))
	#To compute the mean of each of the users 
	mean1=np.array([temp1[Y_train == i].mean()
	                              for i in range(n_classes)])
	mean2=np.array([temp2[Y_train == i].mean()
	                              for i in range(n_classes)])
	#To store the result in a mean vector
	mean_vector=np.zeros((n_classes,2))
	mean_vector[:,0]=mean1
	mean_vector[:,1]=mean2

	'''Try GMMs using different types of classifiers.'''
	####################################################

	'''Since we have class labels for the training data, we can
	    initialize the GMM parameters in a supervised manner.'''

	if select_classifier==1:
		#Creates an instance for the GMM class
		classifier1 = GMM(n_components=n_classes,covariance_type=cov_type,init_params='wc', n_iter=500)
		classifier1.means_=mean_vector
		#Fit the data using the GMM
		classifier1.fit(X_train)
		if fplt:
		    w_factor = 0.5 / classifier1.weights_.max()
		    for pos, covar, w, color in zip(classifier1.means_, classifier1.covars_, classifier1.weights_,colors):
		        draw_ellipse(pos, covar,alpha=w*w_factor, clr=color)
		#Predicts the results using the GMM 
		Y_pred=classifier1.predict(X_test)

	if select_classifier==2:
		#Creates an instance for the GMM class
	    classifier2 = GaussianMixture(n_components=n_classes,means_init=mean_vector,covariance_type=cov_type, max_iter=2000)
	    #Fit the data using the GMM
	    classifier2.fit(X_train)
	    if fplt:
			w_factor = 0.8 / classifier2.weights_.max()
			if cov_type=='tied':
				for pos, w, color in zip(classifier2.means_, classifier2.weights_, colors):
					draw_ellipse(pos, classifier2.covariances_,alpha=w*w_factor,clr=color)
			if cov_type=='full':
				for pos, covar, w, color in zip(classifier2.means_, classifier2.covariances_, classifier2.weights_, colors):
					draw_ellipse(pos,covar,alpha=w*w_factor,clr=color)
			#Predicts the results using the GMM 
	    Y_pred=classifier2.predict(X_test)

	'''Converts the prediction from multi-class to binary class, i.e., user and intruder.'''
	########################################################################################

	Y_pred=np.reshape(Y_pred,(Y_test.shape[0],1))
	Y_pred[Y_pred!=0]=1
	Y_test[Y_test!=0]=1
	confusion = confusion_matrix(Y_test, Y_pred)

	'''To compute the confusion matrix and the coressponding test parameters'''
	########################################################################################
	eps = 1e-9
	accuracy = 0
	if float(np.sum(confusion))!= 0:
	    accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
	specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1]+eps)
	sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0]+eps)
	precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1]+eps)

	#For printing the results
	if show_results:
	    print("\nGlobal Accuracy: " +str(accuracy))
	    print("Specificity: " +str(specificity))
	    print("Sensitivity: " +str(sensitivity))
	    print("Precision: " +str(precision))
	return accuracy

##############################################################################



'''To split the dataset for validation'''
#########################################
       
def split(arr,k,n,dim=1):
	if dim==1:
		arr1, arr2 = np.zeros((len(arr)/n)), np.zeros(((n-1)*int(len(arr)/n)))
	else:
		arr1, arr2 = np.zeros(((len(arr)/n),dim)), np.zeros(((n-1)*int(len(arr)/n),dim))
	for i in range(0,len(arr)/5):
		arr1[i] = arr[5*i+k]
		arr2[4*i] = arr[5*i+1+k]
		arr2[4*i+1]=arr[5*i+2+k]
		arr2[4*i+2]=arr[5*i+3+k]
		arr2[4*i+3]=arr[5*i+4+k]
	return arr1, arr2

##########################################


'''To validate after traing and send the best partition'''
##########################################################

def cross_validation(X_train, Y_train, n_classes, cov_type):
    m,show_results,n = 0.0,False,5
    acc = np.zeros(n)
    length_data = int(len(Y_train)/5)*5
    X_train = X_train[0:length_data-1]
    Y_train = Y_train[0:length_data-1]
    for k in range(0,n):
        X1, X2 = split(X_train,k,5,dim=2)
        Y1, Y2 = split(Y_train,k,5)
        acc[k] = GMM_func(X2, Y2, X1, Y1, n_classes, show_results, cov_type=cov_type)
    #Selecting the best data division
    max_value = max(acc)
    print "\nCross Validation Maximum Accuracy: " +str(max_value)
    k = np.argmax(acc)
    X1, X2 = split(X_train,k,5,dim=2)
    Y1, Y2 = split(Y_train,k,5)
    return X2, Y2

#########################################################


'''Preparing the training and test set for '''
##############################################
def data_division1(X_neutral,X_Happy,X_Sad,X_cont,Y_neutral,Y_Happy,Y_Sad,Y_cont):
    X_train = X_neutral
    Y_train = Y_neutral
    X_test = np.append(X_Happy,X_Sad,axis=0)
    Y_test = np.append(Y_Happy,Y_Sad)
    X_test = np.append(X_test,X_cont,axis=0)
    Y_test = np.append(Y_test,Y_cont)
    return X_train,X_test,Y_train,Y_test

def data_division2(X_neutral,X_Happy,X_Sad,X_cont,Y_neutral,Y_Happy,Y_Sad,Y_cont):
    X_train = np.append(X_Happy,X_Sad,axis=0)
    Y_train = np.append(Y_Happy,Y_Sad)
    X_train = np.append(X_train,X_neutral,axis=0)
    Y_train = np.append(Y_train,Y_neutral)
    X_test = X_cont
    Y_test = Y_cont
    return X_train,X_test,Y_train,Y_test
###############################################


'''Preparing the training and test set for '''
##############################################

        
if __name__ == '__main__':
	#Number of user for training
    n_classes = 5
    #Colors for plotting
    Colors = 'rgbym'
    #Covariance type
    cov_type1='full'
    cov_type2='full'

    plt.figure()
    
    #X_neutral is Neutral Data 
    print "\nGetting Neutral Data....\n"
    X_neutral,Y_neutral = fe.get_data(n_classes,neutral=True)
    print "--------------------------------------------------------"
    
    #X_Happy is Emotional Data = 'Happy'
    print "\nGetting Emotional Data(Happy)....\n"
    X_Happy,Y_Happy = fe.get_data(n_classes,'Happy',cont=False)
    print "--------------------------------------------------------"    

    #X_Sad is Emotional Data = 'Sad'
    print "\nGetting Emotional Data(Sad)....\n"
    X_Sad,Y_Sad = fe.get_data(n_classes,'Sad',cont=False)
    print "--------------------------------------------------------"

    #X_cont is Continuous Data
    print "\nGetting Continuous Data....\n"
    X_cont,Y_cont = fe.get_data(n_classes,cont=True)
    print "--------------------------------------------------------"

    #Preparing the Training data
    #Training on Neutral Data and Testing on Emotional and Continuous Data

    print "Training on Neutral Data and Testing on Emotional and Continuous Data"
    X_train,X_test,Y_train,Y_test=data_division1(X_neutral,X_Happy,X_Sad,X_cont,Y_neutral,Y_Happy,Y_Sad,Y_cont)

    #For Plotting the training data points
    for n, color in enumerate(Colors):
        datatemp1=X_train[:,0]
        datatemp2=X_train[:,1]
        datatemp1=np.reshape(datatemp1,(X_train.shape[0],1))
        datatemp2=np.reshape(datatemp2,(X_train.shape[0],1))
        data1 = datatemp1[Y_train == n]
        data2 = datatemp2[Y_train == n]
        plt.scatter(data1, data2, 0.8, color=color,label=n)
    
    #Five-Fold Cross Validation 
    X_train, Y_train = cross_validation(X_train, Y_train, n_classes,cov_type1)
    
    #Training on Neutral and Emotional Data and Testing on Continuous Data
    GMM_func(X_train, Y_train, X_test, Y_test, n_classes, True, True, Colors,2,cov_type1)

    #For plotting
    plt.title('Plot showing the training data and how the gaussians fit them')
    plt.legend(loc='lower right')
    plt.xlabel('Key hold time(ms)')
    plt.ylabel('Latency(ms)')
    plt.show()   
	

    print "\n--------------------------------------------------------\n"

    #Training on Neutral and emotional Data and Testing on Continuous Data
    print "Training on Neutral and Emotional Data and Testing on Continuous Data"
    X_train,X_test,Y_train,Y_test=data_division2(X_neutral,X_Happy,X_Sad,X_cont,Y_neutral,Y_Happy,Y_Sad,Y_cont)

    #For Plotting the training data points
    for n, color in enumerate(Colors):
        datatemp1=X_train[:,0]
        datatemp2=X_train[:,1]
        datatemp1=np.reshape(datatemp1,(X_train.shape[0],1))
        datatemp2=np.reshape(datatemp2,(X_train.shape[0],1))
        data1 = datatemp1[Y_train == n]
        data2 = datatemp2[Y_train == n]
        plt.scatter(data1, data2, 0.8, color=color,label=n)

    #Cross Validation 
    X_train, Y_train = cross_validation(X_train, Y_train, n_classes,cov_type2)
    
    #Training on Neutral and Emotional Data and Testing on Continuous Data
    GMM_func(X_train, Y_train, X_test, Y_test, n_classes, True, True, Colors, 2,cov_type2)

    #For plotting
    plt.title('Plot showing the training data and how the gaussians fit them')
    plt.legend(loc='lower right')
    plt.xlabel('Key hold time(ms)')
    plt.ylabel('Latency(ms)')
    plt.show()  