from pyspark.mllib.recommendation import ALS
from pyspark import SparkContext
# We will convert the amazon json file into integer

sc =SparkContext()
rank = 10
iterations = 10
lamda = 0.03 
# load training and test data into (user, product, rating) tuples
def parseRating(line):
	fields = line.split(',')
 	return (int(fields[1]), int(fields[2]), float(fields[3]))

def parseRatingTest(line):
 	fields = line.split(',')
  	return (int(fields[1]), int(fields[2]))   

training = sc.textFile("traindata.csv").map(parseRating).cache()
test = sc.textFile("testdata.csv").map(parseRatingTest)
 
# train a recommendation model
model = ALS.train(training,20,10,0.1)
 
# make predictions on (user, product) pairs from the test data
predictions = model.predictAll(test.map(lambda x: (x[0], x[1])))
predictions.coalesce(1).saveAsTextFile('predictions_amazon')
