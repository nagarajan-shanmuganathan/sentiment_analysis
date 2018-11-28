import csv
import string
import re
import copy
import math
from collections import Counter
from random import shuffle

#Special characters that can be removed from the training set. Won't be considered even in the test set
exclude = '\ |\?|\.|\!|\/|\;|\:|\(|\)|[|]|,'

posFile = list(csv.reader(open('hotelPosT-train.txt', "rt", encoding='utf=8'), delimiter = '\t'))
negFile = list(csv.reader(open('hotelNegT-train.txt', "rt", encoding='utf=8'), delimiter = '\t'))

#Splitting the training and test file
splitPos = int(0.90 * len(posFile))
splitNeg = int(0.90 * len(negFile))

trainPosFile = copy.deepcopy(posFile)
trainNegFile = copy.deepcopy(negFile)

shuffle(trainPosFile)
shuffle(trainNegFile)

testPosFile = trainPosFile[splitPos:]
testNegFile = trainNegFile[splitNeg:]

posFile = trainPosFile[:splitPos]
negFile = trainNegFile[:splitNeg]

for row in testPosFile:
	row.append("POS")

for row in testNegFile:
	row.append("NEG")

testFile = testPosFile + testNegFile

shuffle(testFile)

goldFile = copy.deepcopy(testFile)


for row in testFile:
	del row[2]

#Uncomment this if the test file is read as from an external source. Please comment the gold file and accuracy calculations if the file is read externally.

#testFile = list(csv.reader(open('test.txt', "rt", encoding='utf=8'), delimiter = '\t'))

posWords = []
negWords = []
words = []

stopWords = ['a', 'an', 'am', 'after', 'again', 'are', 'as', 'at', 'be', 'because', 'being', 'but', 'by', 'did', 'do', 'from', 'had', 'has', 'have', 'he\'d', 'he\'ll', 'her', 'here', 'him', 'how', 'i', 'i\'d', 'i\'m', 'i\'ve', 'i\'ll', 'if', 'in', 'into', 'is', 'it', 'it\'s', 'its', 'let\'s', 'me', 'more', 'most', 'nor', 'of', 'only', 'once', 'or', 'our', 'ourselves', 'out', 'she', 'should', 'show', 'some', 'such', 'that', 'than', 'the', 'their', 'theirs', 'them', 'themselves', 'they', 'there', 'these', 'they\'d', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'what\'s', 'when', 'where', 'which', 'while', 'who', 'whose', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself']


########## Training ##############

for row in posFile:
	split = row[1].split()
	
	for word in split:
		word = word.lower()
		if word not in stopWords:
			word = re.sub(exclude,'',word) 
			posWords.append(word)
			words.append(word)

for row in negFile:
	split = row[1].split()
	for word in split:
		word = word.lower()
		if word not in stopWords:
			word = re.sub(exclude, '', word)
			negWords.append(word)
			words.append(word)


probPos = math.log10(len(posWords)/(len(posWords) + len(negWords)))
probNeg = math.log10(len(negWords)/(len(posWords) + len(negWords)))


print("Overall Probalities: ", probPos, probNeg)
uniqueWordCount = len(Counter(words))
uniqueWords = Counter(words).keys()
uniquePosWords = Counter(posWords).keys()
uniqueNegWords = Counter(negWords).keys()

numPosWords = len(posWords)
numNegWords = len(negWords)

print("Positive words: ", numPosWords, "Negative words: ", numNegWords, "Unique words: ", len(uniqueWords))

def computeProbabilities(word, uniqueClassWords, numClassWords, classWords):
	if(word not in uniqueClassWords):
		return math.log10(1/(numClassWords + uniqueWordCount))
	else:
		wordCount = Counter(classWords).get(word)
		return math.log10((wordCount + 1)/(numClassWords + uniqueWordCount))

def getProduct(probs): 
	prod = 1
	for prob in probs:
		prod += prob

	return prod

########### Classifying the sentences in the test file ##################
output = []
for row in testFile:
	split = row[1].split()
	positiveProbs = []
	negativeProbs = []	
	
	for word in split:
		word = word.lower()
		if word in stopWords:
			continue
		word = re.sub(exclude, '', word)
		if (word not in uniqueWords):
			positiveProbs.append(math.log10(1/uniqueWordCount))
			negativeProbs.append(math.log10(1/uniqueWordCount))
			continue
			
		positiveProbs.append(computeProbabilities(word, uniquePosWords, numPosWords, posWords))
		negativeProbs.append(computeProbabilities(word, uniqueNegWords, numNegWords, negWords))

	overallPosProb = probPos + getProduct(positiveProbs)
	overallNegProb = probPos + getProduct(negativeProbs)

	print("Positive: ", overallPosProb)
	print("Negative: ", overallNegProb)

	if(overallPosProb >= overallNegProb):
		print("POS")
		output.append([row[0], "POS"])
	else: 
		print("NEG")
		output.append([row[0], "NEG"])

#print(output)

correct = 0
for i in range (0, len(testFile)):
	if output[i][1] == goldFile[i][2]:
		correct += 1
	else:
		print(testFile[i],"\n\n")
		print("Should be: ", goldFile[i][2], " but wrongly classified")	


with open("output.txt", "w") as f:
	for row in output:
		f.write(row[0]+"\t"+row[1]+"\n")
f.close()
print("Correct: ", correct)
print("Length of test file: ", len(testFile))					
accuracy = correct/len(testFile) * 100;
print("Accuracy: ", accuracy)
