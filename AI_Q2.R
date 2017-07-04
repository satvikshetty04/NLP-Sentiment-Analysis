# ----------------------------------- Including necessary packages

#install.packages("RWeka")
library(RWeka)
library(stringr)


# ----------------------------------- Reading txt files

data_a = read.table("./sentiment labelled sentences/amazon_cells_labelled.txt", header = F, 
                    comment.char = "", col.names = c("Text", "Sentiment"), 
                    strip.white = T, sep = "\t", quote = "", as.is = T)
data_b = read.table("./sentiment labelled sentences/imdb_labelled.txt", header = F, as.is = T,
                    col.names = c("Text", "Sentiment"), strip.white = T, sep = "\t", quote = "")

data_c = read.table("./sentiment labelled sentences/yelp_labelled.txt", header = F, as.is = T,
                    col.names = c("Text", "Sentiment"), strip.white = T, sep="\t", quote = "")


# ----------------------------------- Merging all three data sets

data = rbind(data_a, data_b, data_c)
#data = data_a
#data = data_b
#data = data_c


# ----------------------------------- Pre-Processing/Cleaning

data[,1] = tolower(data[,1])
data[,1] = str_replace_all(data[,1], pattern ="[^[:alnum:][:space:]'/-]", replacement = " ")
data[,1] = trimws(data[,1])


# ----------------------------------- Sampling and creating train & test sets

sample = sample(nrow(data), size = 0.7 * nrow(data))
train_data = data[sample,]
test_data = data[-sample,]


# ----------------------------------- Obtaining ID3 Classifier

#Below statements need to be uncommented the first time in case the package is not available
#WPM("refresh-cache")
#WPM("install-package", "simpleEducationalLearningSchemes")
#WPM("load-package", "simpleEducationalLearningSchemes")

ID3 <- make_Weka_classifier("weka/classifiers/trees/Id3") 


# ----------------------------------- Creating unigrams , bigrams and trigrams

unigrams = RWeka::NGramTokenizer(train_data[,1], Weka_control(min = 1, max = 1))
unigrams = unique(unigrams)

bigrams = RWeka::NGramTokenizer(train_data[,1], Weka_control(min = 2, max = 2))
bigrams = unique(bigrams)

trigrams = RWeka::NGramTokenizer(train_data[,1], Weka_control(min = 3, max = 3))
trigrams = unique(trigrams)


# ----------------------------------- Creating train feature matrix & appending target

unigram_features_train = matrix(data = F, nrow = nrow(train_data), ncol = length(unigrams))
r = 1
for( row in train_data[,1]){
  row_ngrams = RWeka::NGramTokenizer(row, Weka_control(min = 1, max = 1))
  unigram_features_train[r,] = unigrams %in% row_ngrams
  r = r+1
}
unigram_features_train = data.frame(unigram_features_train, factor(train_data[,2]))
colnames(unigram_features_train) = c(unigrams, "Target_variable")


bigram_features_train = matrix(data = F, nrow = nrow(train_data), ncol = length(bigrams))
r = 1
for( row in train_data[,1]){
  row_ngrams = RWeka::NGramTokenizer(row, Weka_control(min = 2, max = 2))
  bigram_features_train[r,] = bigrams %in% row_ngrams
  r = r+1
}
bigram_features_train = data.frame(bigram_features_train, factor(train_data[,2]))
colnames(bigram_features_train) = c(bigrams, "Target_variable")


trigram_features_train = matrix(data = F, nrow = nrow(train_data), ncol = length(trigrams))
r = 1
for( row in train_data[,1]){
  row_ngrams = RWeka::NGramTokenizer(row, Weka_control(min = 3, max = 3))
  trigram_features_train[r,] = trigrams %in% row_ngrams
  r = r+1
}
trigram_features_train = data.frame(trigram_features_train, factor(train_data[,2]))
colnames(trigram_features_train) = c(trigrams, "Target_variable")



# ----------------------------------- Creating test feature matrix & appending target

unigram_features_test = matrix(data = F, nrow = nrow(test_data), ncol = length(unigrams))
r = 1
for( row in test_data[,1]){
  row_ngrams = RWeka::NGramTokenizer(row, Weka_control(min = 1, max = 1))
  unigram_features_test[r,] = unigrams %in% row_ngrams
  r = r+1
}
unigram_features_test = data.frame(unigram_features_test, factor(test_data[,2]))
colnames(unigram_features_test) = c(unigrams, "Target_variable")


bigram_features_test = matrix(data = F, nrow = nrow(test_data), ncol = length(bigrams))
r = 1
for( row in test_data[,1]){
  row_ngrams = RWeka::NGramTokenizer(row, Weka_control(min = 2, max = 2))
  bigram_features_test[r,] = bigrams %in% row_ngrams
  r = r+1
}
bigram_features_test = data.frame(bigram_features_test, factor(test_data[,2]))
colnames(bigram_features_test) = c(bigrams, "Target_variable")


trigram_features_test = matrix(data = F, nrow = nrow(test_data), ncol = length(trigrams))
r = 1
for( row in test_data[,1]){
  row_ngrams = RWeka::NGramTokenizer(row, Weka_control(min = 3, max = 3))
  trigram_features_test[r,] = trigrams %in% row_ngrams
  r = r+1
}
trigram_features_test = data.frame(trigram_features_test, factor(test_data[,2]))
colnames(trigram_features_test) = c(trigrams, "Target_variable")


# ----------------------------------- Creating ID3 models for unigram, bigram and trigram

model_unigram = ID3(formula = as.factor(unigram_features_train$Target_variable)~., data = unigram_features_train)
model_bigram = ID3(formula = as.factor(bigram_features_train$Target_variable)~., data = bigram_features_train)
model_trigram = ID3(formula = as.factor(trigram_features_train$Target_variable)~., data = trigram_features_train)


# ----------------------------------- Predicting for test data based on ID3 models

pred_unigram = predict(model_unigram, newdata = unigram_features_test[,1:(ncol(unigram_features_test)-1)])
pred_bigram = predict(model_bigram, newdata = bigram_features_test[,1:(ncol(bigram_features_test)-1)])
pred_trigram = predict(model_trigram, newdata = trigram_features_test[,1:(ncol(trigram_features_test)-1)])


# ----------------------------------- Calculating accuracy, precision, recall, f_measure

#Unigram Model
fp = 0
tp = 0
fn = 0
tn = 0
accuracy = 0
precision = 0
recall = 0
f_measure = 0
for(i in 1:length(pred_unigram)){
  if(pred_unigram[i] == unigram_features_test[i,ncol(unigram_features_test)]){
    if(pred_unigram[i] == 1){
      tp = tp + 1
    } else{
      tn = tn + 1
    }
  } else {
    if(pred_unigram[i] == 1){
      fp = fp + 1
    } else{
      fn = fn + 1
    }
  }
}
accuracy = (tp + tn)/(tp + tn + fp + fn) * 100
precision = tp/(tp + fp) * 100
recall = tp/(tp+fn) * 100
f_measure = (2 * recall * precision)/(recall + precision) 
cat("----------Unigram----------\nAccuracy:", accuracy, "\nPrecision:", precision, 
    "\nRecall:",recall,"\nF-Measure:",f_measure)
cat(tp, fn, fp, tn)


#Bigram Model
fp = 0
tp = 0
fn = 0
tn = 0
accuracy = 0
precision = 0
recall = 0
f_measure = 0
for(i in 1:length(pred_bigram)){
  if(pred_bigram[i] == bigram_features_test[i,ncol(bigram_features_test)]){
    if(pred_bigram[i] == 1){
      tp = tp + 1
    } else{
      tn = tn + 1
    }
  } else {
    if(pred_bigram[i] == 1){
      fp = fp + 1
    } else{
      fn = fn + 1
    }
  }
}
accuracy = (tp + tn)/(tp + tn + fp + fn) * 100
precision = tp/(tp + fp) * 100
recall = tp/(tp+fn) * 100
f_measure = (2 * recall * precision)/(recall + precision) 
cat("----------Bigram----------\nAccuracy:", accuracy, "\nPrecision:", precision, 
    "\nRecall:",recall,"\nF-Measure:",f_measure)
cat(tp, fn, fp, tn)


#Trigram Model
fp = 0
tp = 0
fn = 0
tn = 0
accuracy = 0
precision = 0
recall = 0
f_measure = 0
for(i in 1:length(pred_trigram)){
  if(pred_trigram[i] == trigram_features_test[i,ncol(trigram_features_test)]){
    if(pred_bigram[i] == 1){
      tp = tp + 1
    } else{
      tn = tn + 1
    }
  } else {
    if(pred_bigram[i] == 1){
      fp = fp + 1
    } else{
      fn = fn + 1
    }
  }
}
accuracy = (tp + tn)/(tp + tn + fp + fn) * 100
precision = tp/(tp + fp) * 100
recall = tp/(tp+fn) * 100
f_measure = (2 * recall * precision)/(recall + precision) 
cat("----------Trigram----------\nAccuracy:", accuracy, "\nPrecision:", precision, 
    "\nRecall:",recall,"\nF-Measure:",f_measure)
cat(tp, fn, fp, tn)
