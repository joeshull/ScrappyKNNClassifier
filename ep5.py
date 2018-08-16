"""
From the Google Developers' series Machine Learning Recipes. 
Code from Episode #5. "Writing Our First Classifier"
link: https://www.youtube.com/watch?v=AoeEHqVSNOw

Commented by Joe Shull
"""

#Import the distance module from scipy.spatial
from scipy.spatial import distance

#Define a function euc that takes a and b parameters
def euc (a,b):
	#Return the euclidean distance between the two parameters called
        return distance.euclidean (a,b)

#Make a class ScrappyKNN.
class ScrappyKNN():

#ScrappyKNN has a function fit that takes self, X_train, and Y_train parameters
#Self variable so we can take existing variables and store them in the instatiated object.
    def fit(self, X_train, y_train):
#From self, get the X_train attribute and set it the X_train parameter with which the
#function was called.
#This stores the parameters in memory so methods can call it.
        self.X_train = X_train
        self.y_train = y_train
	
#ScrappyKNN has a function predict that takes self and X_Test parameters.
    def predict(self, X_test):
	#Set predictions to an empty list
        predictions = []
		#For each row in the X_Test data,
		#Python will iterate through each row the data with which it was called.
        for row in X_test:
			#Go through each row and kick off method self.closest by sending it the values.
			#Closest() will iterate through the training data and return a value.
			#At each iteration, set label to the data returned by function closest.
            label = self.closest(row)
            #Get list predictions and append to the contents the value returned by label.
            predictions.append(label)
        #At end of loop, return predictions list to function that called it.
        return predictions

	#Class ScrappyKNN has a function closest, which takes self and row parameters
    def closest (self, row):
    	#Set best_dist to function euc and call euc with row and self.X_train[0] parameters.
    	#Variables fed to it will be row of data from X_test and the X.train data starting at row 0
    	#Best_dist will be updated with our loop, but we start it with the formula to understand
    	#its logic.
        best_dist = euc(row, self.X_train[0])
        #Set best_index to 0. This will store the index (position) of our best fit.
        best_index = 0
        #Loop through all rows of the data from 1 through the length of self.X_train.
        for i in range (1, len(self.X_train)):
        	#Set dist to function euc and call it with row and self.X_train[i].
        	#dist will iterate through all rows of data and update its value
            dist = euc(row, self.X_train[i])
            #If dist iterates to a value less than the one currently stored as best_dist
            if dist < best_dist:
            	#store new dist and best_dist 
                best_dist = dist
                #Simultaneously store best_index as i
                best_index = i
		#when loop terminates, best_index will have stored the index where dist was
		#lowest out of all possible options.
		#Get self.y_train and pull the data stored in best_index. Return the value(s)
        return self.y_train[best_index]
    
    
#From sklearn library, import the datasets module
from sklearn import datasets
#From datasets get the class load_iris and set it to iris.
iris = datasets.load_iris()

#From iris, get the data attribute and set it to X
X = iris.data
#From iris, get the target attribute and set it to Y.
y = iris.target

#Import the train_test_split module from the sklearn.model_selection library
from sklearn.model_selection import train_test_split
#Set X_Train, X_Test, y_train, y_test to the train_test_split function and call it with
#X, y, and test_size = .5 parameters.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.5)

#Set my_classifier to an instance of ScrappyKNN(). This is where ScrappyKNN is instantiated.
my_classifier=ScrappyKNN()


#From my_classifier get the fit function and call it with 
#self, X_train, and y_train parameters.
my_classifier.fit(X_train, y_train)

#Global variable predictions set to 
#(from my_classifier get the predict function and call it with self and X_Test)
predictions = my_classifier.predict(X_test)

#From sklearn.metrics module, import the accuracy_score function
from sklearn.metrics import accuracy_score

#Get the accuracy_score function and call it with y_test and predictions parameters.
print accuracy_score(y_test, predictions)