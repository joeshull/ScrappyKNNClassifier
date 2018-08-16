"""
From the Google Developers' series Machine Learning Recipes. 
Code from Episode #5. "Writing Our First Classifier"
link: https://www.youtube.com/watch?v=AoeEHqVSNOw

Commented by Joe Shull
"""

#Import the distance module from scipy.spatial
from scipy.spatial import distance

#Define a method euc that takes a and b parameters
def euc (a,b):
	#Return the euclidean distance between the two parameters called
        return distance.euclidean (a,b)

#Make a class called ScrappyKNN.
class ScrappyKNN():

    #ScrappyKNN has a method fit that takes self, X_train, and Y_train parameters
    def fit(self, X_train, y_train):
        #From self, create the X_train attribute and set it the X_train parameter with which the method was called.
        #This stores the parameters in memory so methods can call it.
        self.X_train = X_train
        self.y_train = y_train
	
    #ScrappyKNN has a method predict that takes self and X_Test parameters.
    def predict(self, X_test):
	#Set local predictions to an empty list
        predictions = []
	#For each row in the X_Test data,
	#Python will iterate through each row the data with which it was called.
        for row in X_test:
	    #From self get the method called closest and call it with row parameter
	    #Set label to the data returned by self.closest.
            label = self.closest(row)
            #Get list called predictions and append the label to its contents.
            predictions.append(label)
        #At end of loop, return predictions list to method that called it.
        return predictions

    #Class ScrappyKNN has a method closest, which takes self and row parameters
    def closest (self, row):
    	#Set best_dist to method euc and call euc with row and self.X_train[0] parameters.
    	#Variables fed to it will be row of data from X_test and the X.train data starting at row 0
    	#Best_dist will be updated with our loop, but we start it with the formula to understand its logic and give it a value.
        best_dist = euc(row, self.X_train[0])
        #Set best_index to 0. This will store the index (position) of our best fit.
        best_index = 0
        #Loop through all rows of the data from 1 through the length of self.X_train.
        for i in range (1, len(self.X_train)):
            #Set dist to method euc and call it with row and self.X_train[i].
            #Dist will iterate through all rows of data and update its value
            dist = euc(row, self.X_train[i])
            #If dist iterates to a value less than the one currently stored as best_dist
            if dist < best_dist:
            	#store new dist and best_dist 
                best_dist = dist
                #Simultaneously store best_index as i
                best_index = i
	#When loop terminates, best_index will have stored the index where dist was
	#lowest out of all possible options.
	#From self find the y_train variable, call it with best_index and return that value
        return self.y_train[best_index]
    
    
#From sklearn library, import the datasets module
from sklearn import datasets
#Set iris to an instance of datasets.load_iris()
iris = datasets.load_iris()

#From iris, get the data attribute and set it to X
X = iris.data
#From iris, get the target attribute and set it to Y.
y = iris.target

#Import the train_test_split module from the sklearn.model_selection library
from sklearn.model_selection import train_test_split

#(Unpacking train_test_split)
#Set X_Train, X_Test, y_train, y_test to the train_test_split method and call it with
#X, y, and test_size = .5 parameters. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.5)

#Set my_classifier to an instance of ScrappyKNN(). This is where ScrappyKNN is instantiated.
my_classifier=ScrappyKNN()


#From my_classifier get the fit method and call it with 
#self, X_train, and y_train parameters.
my_classifier.fit(X_train, y_train)


#From my_classifier get the predict method and call it with self and X_Test,
#Set it to global predictions.
predictions = my_classifier.predict(X_test)

#From sklearn.metrics module, import the accuracy_score method
from sklearn.metrics import accuracy_score

#Get the accuracy_score method and call it with y_test and predictions parameters.
print accuracy_score(y_test, predictions)
