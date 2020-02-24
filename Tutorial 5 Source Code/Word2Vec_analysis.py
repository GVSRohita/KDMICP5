from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec

# creating spark session
spark = SparkSession.builder .appName("Ngram Example").getOrCreate()

documentDF = spark.createDataFrame([
    ("input_1.txt".split(" "), ),
    ("input_2.txt".split(" "), ),
    ("input_3.txt".split(" "), ),
    ("input_4.txt".split(" "), ),
    ("input_5.txt".split(" "), )], ["text"])

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(documentDF)
print("The Word2Vec model identified based on the input data:")
print(model)
result = model.transform(documentDF)

for row in result.collect():
    text, vector = row
    #printing the results
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))

# showing the synonyms and cosine similarity of the word in input data
synonyms = model.findSynonyms("data", 5)   # its okay for certain words , real bad for others
synonyms.show(5)

# closing the spark session
spark.stop()