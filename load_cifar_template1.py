import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import utils
import tensorflow as tf

#Step 1: define a function to load traing batch data from directory
def load_training_batch(folder_path,batch_id):
	"""
	Args:
		folder_path: the directory contains data files
		batch_id: training batch id (1,2,3,4,5)
	Return:
		features: numpy array that has shape (10000,3072)
		labels: a list that has length 10000
		"""
	file=folder_path+"\\data_batch_{}".format(batch_id)

	###load batch using pickle###
	with open(file,'rb') as fo:
		dict=pickle.load(fo, encoding='latin-1')
	###fetch features using the key ['data']###
	#features = dict['data'] #pass
	#features=features.reshape(10000,3072)
	###fetch labels using the key ['labels']###
	#labels = dict['labels'] #pass
	return dict#features,labels

#Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):
	"""
	Args:
		folder_path: the directory contains data files
	Return:
		features: numpy array that has shape (10000,3072)
		labels: a list that has length 10000
	"""
	file=folder_path+"\\test_batch"
	###load batch using pickle###
	with open(file,'rb') as fo:
		dict=pickle.load(fo, encoding='latin-1')
	###fetch features using the key ['data']###
	#features = dict['data'] #pass
	#features=features.reshape(10000,3072)
	###fetch labels using the key ['labels']###
	#labels = np.array(dict['labels']) #pass
	return dict #features,labels

#Step 3: define a function that returns a list that contains label names (order is matter)
"""
	airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""
def load_label_names():
	label_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    
	return label_names

#Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
	"""
	Args:
		features: a numpy array with shape (10000, 3072)
	Return:
		features: a numpy array with shape (10000,32,32,3)
	"""
	features=features.reshape(-1,32,32,3)
    #pass  
	return features


#Step 5 (Optional): A function to display the stats of specific batch data.
def display_data_stat(folder_path,batch_id,data_id):
	"""
	Args:
		folder_path: directory that contains data files
		batch_id: the specific number of batch you want to explore.
		data_id: the specific number of data example you want to visualize
	Return:
		None

	Descrption: 
		1)You can print out the number of images for every class. 
		2)Visualize the image
		3)Print out the minimum and maximum values of pixel 
	"""
	pass

#Step 6: define a function that does min-max normalization on input
def normalize(x):
	"""
	Args:
		x: features, a numpy array
	Return:
		x: normalized features
	"""
	x=x/(max(x)-min(x))
	return x

#Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x):
	"""
	Args:
		x: a list of labels
	Return:
		a numpy array that has shape (len(x), # of classes)
	"""
	x=np.array(x)
	x=x.reshape(-1,1)    
	enc=OneHotEncoder()
	enc.fit(x)
	x_ohe=enc.transform(x).toarray()
	return x_ohe

#Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
def preprocess_and_save(features,labels,filename):
	"""
	Args:
		features: numpy array
		labels: a list of labels
		filename: the file you want to save the preprocessed data
	"""
	features_clean=normalize(features)
	#labels=labels.reshape(-1,1)
	labels_clean=one_hot_encoding(labels)
    
	#a=np.concatenate((features_clean, labels_clean),axis=1)
	a=[features_clean, labels_clean]
	fileObject=open(filename,'wb')
   
   #writes obejct to file name
	with open(filename, 'wb') as fo:
		pickle.dump(a,fo)
	fileObject.close()
    
    
	#pass

#Step 9:define a function that preprocesss all training batch data and test data. 
#Use 10% of your total training data as your validation set
#In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path):
	"""
	Args:
		folder_path: the directory contains your data files
	"""
	for batch_id in range(1,6):
		data=load_training_batch(folder_path,batch_id)
		batch_features=data['data']
		batch_labels=data['labels']
		#batch_features, batch_labels=load_training_batch(folder_path,batch_id)
		X_train, X_val,y_train,y_val=train_test_split(batch_features, batch_labels,test_size=.1)
		filename='train_batch_{}'.format(batch_id)
		preprocess_and_save(X_train, y_train,filename)
		if batch_id==1:
			X_val_processed=X_val
			y_val_processed=y_val
		else:
			X_val_processed=np.vstack((X_val_processed,X_val))
			y_val_processed=np.vstack((y_val_processed, y_val))
            
	filename_val='validation_batch'
	preprocess_and_save(X_val_processed,y_val_processed,filename_val)
    
	test_data=load_testing_batch(folder_path)
	test_features=test_data['data']
	test_labels=test_data['labels']
	#test_features,test_labels=load_testing_batch(folder_path) 
	filename_test='test_batch_processed'
	preprocess_and_save(test_features, test_labels,filename_test)
        
        
	

#Step 10: define a function to yield mini_batch
def mini_batch(features,labels,mini_batch_size):
	"""
	Args:
		features: features for one batch
		labels: labels for one batch
		mini_batch_size: the mini-batch size you want to use.
	Hint: Use "yield" to generate mini-batch features and labels
	"""
    #batch size is number of samples in each iteration 
	features,labels=utils.shuffle(features,labels)
	data_size=np.shape(labels)[0]
	its=data_size//mini_batch_size
	i=0
	j=mini_batch_size
	k=0
	while k != its:
		mb_features=features[i:j,:]
		mb_labels=labels[i:j,:]
		i+=mini_batch_size
		j+=mini_batch_size
		k+=1
		yield mb_features, mb_labels
        

#Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
def load_preprocessed_training_batch(batch_id,mini_batch_size):
	"""
	Args:
		batch_id: the specific training batch you want to load
		mini_batch_size: the number of examples you want to process for one update
	Return:
		mini_batch(features,labels, mini_batch_size)
	"""
	filename='train_batch_{}'.format(batch_id)
	features, labels = pickle.load(open(filename,'rb'))
	return mini_batch(features,labels,mini_batch_size)

#Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch():
	file_name ='validation_batch' 
	features,labels = pickle.load(open('validation_batch','rb'))
	return features,labels

#Step 13: load preprocessed test batch
def load_preprocessed_test_batch():
	file_name = 'test_batch_processed'
	features,labels =pickle.load(open('test_batch_processed','rb'))
	return features, labels

