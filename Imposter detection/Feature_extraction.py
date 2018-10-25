import pandas as pd
from datetime import datetime
import os 
import numpy as np


#The path where the data is stored
datapath_begin='New Data Collection/'
datapath_end='/Keyboard Database/sentence/'


#The details of the user data we are considering 
user_details = {0:['Lalit','14EC10025'],2:['Arnab','14EC35031'],3:['Sandeep','14EC35033'],4:['Charu','14EC35003'],1:['Koruprolu Asish','14EC10024'] }


'''To read files and extract data from it '''
##############################################

def extract_data(username,roll_number,mood,cont=False):
	
	data2 = pd.DataFrame({'key' : [],
					 'press_time' : [],
					 'latency' : []})
	path = datapath_begin + username + '_' + roll_number + datapath_end + mood +'/'
	directory = os.path.join(path)
	print 'Collecting data of '+username+'...'
	for root,dirs,files in os.walk(directory):
		for file in files:
			if file.endswith(".txt"):
				with open(directory+file,"r") as file1:
					if cont:
						FMT = '%d:%H:%M:%S.%f'
					else:
						FMT = '%d:%m:%Y:%H:%M:%S.%f' 
					loop = 0
					queue = [[],[]]
					data1 = pd.DataFrame({'key' : [],
										'time' : [],
					 					'key_press_time' : []})
					for line in file1:
						loop += 1
						#refine data
						if cont:
							a = line.split()[0]
							b = line.split()[1]
							c = line.split()[7]
							
						else: 
							if len(line.split()) == 3:
								[a,b,c] = line.split()
								if b=='\b':
									b="backspace"
								c = c[:-4] + '.' + c[-3:]
								c = c[:6] + '20' + c[6:]

							elif len(line.split()) == 2:
								[a,c] = line.split()
								b = "space"
								c = c[:-4] + '.' + c[-3:]
								c = c[:6] + '20' + c[6:]

							else:
								print "Error in Data"+str(file)
						
						#record keypress
						if a == 'KeyDown':
							try:
								index_of_letter = queue[0].index(b)
							except:
								queue[0].append(b)
								queue[1].append(c)
							else:
								continue

						#compute data for key release
						if loop > 1 and a == 'KeyUp':
							#locate index

							try:
								index_of_letter = queue[0].index(b)
							
							except ValueError:
								index_of_letter = None
							else:
								c1 = queue[1][index_of_letter]

								#calculate time diff 
								tdelta = datetime.strptime(c, FMT) - datetime.strptime(c1, FMT)

								df1 = pd.DataFrame({'key' :[b],
													'time' :[c1],
													'key_press_time' :[tdelta.microseconds]})
								data1 = data1.append(df1,ignore_index=True)

								#remove key from queue
								del queue[0][index_of_letter]
								del queue[1][index_of_letter]
						if loop>5000:
							break

				for i in range(len(data1.index)):
					if i == 0:
						[a1,b1,c1] = data1.iloc[i]
						df1 = pd.DataFrame({'key' :[a1],
									 		'press_time' :[0],
									 		'latency' :[b1]})
						data2 = data2.append(df1,ignore_index=True)
					else:
						[a1,b1,c1] = data1.iloc[i]
						[a2,b2,c2] = data1.iloc[i-1]
						tdelta = datetime.strptime(c1, FMT) - datetime.strptime(c2, FMT)
						df1 = pd.DataFrame({'key' :[a1],
									 		'press_time' :[tdelta.microseconds],
									 		'latency' :[b1]})
						data2 = data2.append(df1,ignore_index=True)
	return data2

#####################################################




'''To remove outliers from the dataset '''
##############################################

def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

###############################################






'''To remove outliers from the dataset '''
##############################################

def get_data(n_classes,emotion='Happy',neutral=False,cont=False):
	data_list1=[]
	data_list2=[]
	label_list=[]
	data = pd.DataFrame({'key' : [],
					 'press_time' : [],
					 'latency' : []})
	data1 = pd.DataFrame({'key' : [],
					 'press_time' : [],
					 'latency' : []})
	for i in range(n_classes):
		#Selecting the type of data
		if neutral:
			fldr_name='Neutral'
		elif cont:
			fldr_name='Continuous'
		else:
			fldr_name='Emotional/'+emotion
		#Extracting raw data from the files 
		data1=extract_data(user_details[i][0],user_details[i][1],fldr_name,cont)
		data2 = data1.loc[(data1['press_time'] != 0)]
		#Removes the outliers from the data	with respect to both the attributes
		data=remove_outlier(remove_outlier(data2,'press_time'),'latency')
		for j in range(data.shape[0]):
			#Converted the data to ms
			data_list1.append(data.iloc[j,1]/1000)
			data_list2.append(data.iloc[j,2]/1000)
			label_list.append(i)
	#To store the data in a numpy array in a proper way
	X_data=np.zeros((len(data_list1)-1,2))
	Y_data=np.zeros((len(data_list1)-1))
	X_data[:,0]=np.array(data_list1[0:-1])
	X_data[:,1]=np.array(data_list2[0:-1])
	Y_data[:]=np.array(label_list[0:-1])
	Y_data=np.reshape(Y_data,(len(data_list1)-1,1))
	return X_data,Y_data

###############################################






'''Main function to verify the Feature extraction '''
#####################################################


if __name__ == '__main__':
	n_classes=5
	X_cont,Y_cont = get_data(n_classes,cont=True)
	print X_cont.shape
	print Y_cont.shape