
# Dane potrzebne do przeprowadzenia badania pochodzą m.in. z serwisu goodreads.com oraz steamboatbooks.com. Zgromadzona baza zawiera 51 000 recenzji dotyczących 363 różnych książek. Wśród nich jest 20 gatunków książek (thriller, science fiction itp.). Na podstawie opisów i tytułów książek wyszczególniono 9 różnych, najczęściej występujących tematyk (miłość, wojna, śmierć itp.). Dane dotyczące liczby sprzedanych egzemplarzy książki według serwisu amazon.com pochodzą z bazy danych serwisu kaggle.com. 
# Zmiennymi objaśniajacymi są również: płeć autora książki, płeć recenzenta, płeć głównego bahatera książki, gatunek, cena, liczba stron, średnia liczba przyznanych gwiazdek, rok wydania książki, liczba zgromadzonych przez książkę nagród, format wydania książki.
# Zmienną objaśnianą jest  "opinia recenzenta"  (pozytywna/ negatywna opinia).
# Celem pracy jest porównanie dokładności oszacowań modeli o różnych sposobach predykcji.
# Wykorzystano następujące modele i metody :
#   - walidacja krzyżowa modelu,
# - drzewo decyzyjne,
# - stopping criteria,
# - prunning,
# - bagging,
# - sub-bagging,
# - random forest,
# - gbm,
# - neural networks.

# Wczytywanie danych
library(readxl)
dane<- read_excel("/Users/alubis/Desktop/ML2/ML2projekt/model2.xlsx")

library(dplyr)
library(readr)
library(AER)
library(lmtest) # lrtest(), waldtest()
library(nnet)
library(caret)
library(verification)
library(tidyr)
library(foreign)
library(janitor) # tabyl()
library(class)
library(pROC)
library(DMwR) 
library(ROSE)
library(tidyverse)
library(MASS)
library(gplots)
library(tree)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(ranger)
library(gbm)
library(neuralnet)

# Przygotowanie danych
# Usunięcie brakow danych, zadeklarowanie zmiennych jakościowych:
  
dane = na.omit(dane)
dane$opinia=as.factor(dane$opinia)
levels(dane$opinia) <- c("neg", "pos")
dane$formaty <- as.factor(dane$formaty)
dane$plec_autora <- as.factor(dane$plec_autora)
dane$plec_recenzenta <- as.factor(dane$plec_recenzenta)
dane$rodzaj <- as.factor(dane$rodzaj)
dane$tematyka <- as.factor(dane$tematyka)

# Ogólna postać modelu

modelformula <- opinia ~  
  cena+ formaty + lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + rok + stars + tematyka +                 sprzedaz

# Podział danych na treningowe i testowe 

set.seed(123456)

which_train <- createDataPartition(dane$opinia, 
                                   p = 0.7, 
                                   list = FALSE) 
dane_train <- dane[which_train,]
dane_test <- dane[-which_train,]

# Dane są niezrównoważone, zatem nakładamy wagi:

set.seed(123456)

(freqs <- table(dane_train$opinia))

myWeights <- ifelse(dane_train$opinia == "pos",
                    0.5/freqs[1], 
                    0.5/freqs[2])


dane_train0= dane_train %>% filter(dane_train$opinia=="neg")
dane_train1= dane_train %>% filter(dane_train$opinia=="pos")
train_sample0 = dane_train0[sample(nrow(dane_train0), size= length(dane_train0$opinia)),]
train_sample1 = dane_train1[sample(nrow(dane_train1), size= length(dane_train0$opinia)),]
dane_train=rbind(train_sample0,train_sample1)
rm(dane_train0,dane_train1,train_sample0,train_sample1)


# MODEL I
# najprostszy model drzewa decyzyjnego

set.seed(123456)
dane.tree <- 
  rpart(modelformula,
        data = dane_train,
        method = "class") 

source("accuracy_ROC.R")
set.seed(123456)

# training set
accuracy_ROC(model = dane.tree,
             data = dane_train,
             target_variable = "opinia",
             predicted_class = "pos")

# testing set
tree_basic = accuracy_ROC(model = dane.tree,
                          data = dane_test,
                          target_variable = "opinia",
                          predicted_class = "pos")

table= as.data.frame(tree_basic)
table

# Tabela powyżej przedstawia miary dokładności oszacowań dla modelu drzewa decyzyjnego na danych testowych. Dokładność oszacowań wynosi 59%, wrażliwość 59%, specyficzność 54% natomiast wartość ROC 57%. W dalszej części pracy zostaną przedstawione metody i modele, które pomogą w polepszeniu wyników klasyfikacji.

# MODEL II
# cross-validation

tc <- trainControl(method = "cv",
                   number = 10, 
                   classProbs = TRUE,
                   summaryFunction = twoClassSummary)

cp.grid <- expand.grid(cp = seq(0, 0.03, 0.001))

set.seed(123456)
dane.tree2 <- train(modelformula,
                    data = dane_train, 
                    method = "rpart", 
                    metric = "ROC",
                    trControl = tc,
                    tuneGrid = cp.grid)
#weights=myWeights)

# let us examine the results
dane.tree2

set.seed(123456)
cp.grid <- expand.grid(cp = c(0))
dane.tree2.2 <- train(modelformula,
                      data = dane_train, 
                      method = "rpart", 
                      metric = "ROC",
                      trControl = tc,
                      tuneGrid = cp.grid)
#weights=myWeights)
dane.tree2.2


# Porównanie modeli

source("accuracy_ROC.R")

set.seed(123456)
# training set
accuracy_ROC(model = dane.tree2.2,
             data = dane_train,
             target_variable = "opinia",
             predicted_class = "pos")

# testing set
tree_cv = accuracy_ROC(model = dane.tree2.2,
                       data = dane_test,
                       target_variable = "opinia",
                       predicted_class = "pos")

table2= as.data.frame(tree_cv)
table = cbind(table, table2)
table = table[, order(desc(table["ROC",])), drop = FALSE]
table

# Wzrosła wartość krzywej ROC.

# MODEL III
# STOPPING CRITERIA

set.seed(123456)
dane.tree3 <- 
  rpart(modelformula,
        data = dane_train,
        method = "class",
        minsplit = 100,
        minbucket = 50, 
        maxdepth = 15,
        cp=0)

source("accuracy_ROC.R")
set.seed(123456)

# training set
accuracy_ROC(model = dane.tree3,
             data = dane_train,
             target_variable = "opinia",
             predicted_class = "pos")

# testing set
tree_stop_crit = accuracy_ROC(model = dane.tree3,
                              data = dane_test,
                              target_variable = "opinia",
                              predicted_class = "pos")
tree_stop_crit


table2= as.data.frame(tree_stop_crit)
table = cbind(table, table2)
table= table[, order(desc(table["ROC",])), drop = FALSE]
table

# Zastosowanie kryteriów stopujących zwiększyło wartość ROC.

# MODEL IV
# TREE PRUNNING

set.seed(123456)
dane.tree4 <- rpart(modelformula,
                    data = dane_train,
                    method = "class",
                    minsplit = 85, # ~ 2% of training set
                    minbucket = 42, # ~ 1% of training set
                    maxdepth = 10, # default
                    cp = 0)

# training set
set.seed(123456)
accuracy_ROC(model = dane.tree4,
             data = dane_train,
             target_variable = "opinia",
             predicted_class = "pos")

set.seed(123456)
# testing set
tree_prunning = accuracy_ROC(model = dane.tree4,
                             data = dane_test,
                             target_variable = "opinia",
                             predicted_class = "pos")

tree_prunning

source("accuracy_ROC.R")

table2= as.data.frame(tree_prunning)
table = cbind(table, table2)
table = round(table, 3)
table = table[, order(desc(table["ROC",])), drop = FALSE]
table

# Wartość ROC wyższa niż w podstawowym drzewie, wzrósł również wskaźnik Accurancy.


# MODEL V 
# bagging

n <- nrow(dane_train)
# we create an empty list to collect results 
results_logit <- list()

set.seed(123456)

for (sample in 1:20) {
  message(sample)
  # we draw n-element sample (with replacement) 
  data_sample <- 
    dane_train[sample(1:n, 
                      size = n,
                      replace = TRUE),]
  # paste as the next element of the list 
  results_logit[[sample]] <- glm(modelformula,
                                 data_sample, 
                                 family = binomial(link = "logit"))
}

set.seed(123456)
forecasts_p = predict(object = results_logit[[1]],
                      newdata = dane_test,
                      type = "response")


# now we make predictions for all models
set.seed(123456)

forecasts_bag <- 
  sapply(results_logit,
         function(x) 
           predict(object = x,
                   newdata = dane_test,
                   type = "response")) %>% 
  data.frame()

set.seed(123456)
forecast_bag_final <-
  ifelse(rowSums(forecasts_bag < 0.5) > 20/2,
         "neg", "pos") %>% 
  factor(., levels = c("neg", "pos"))

set.seed(123456)
# accuracy of that model
a =confusionMatrix(data = forecast_bag_final,
                   reference = dane_test$opinia,
                   positive = "pos")

sensitivity = as.table(as.matrix(a, what = "classes")[1])
specificity = as.table(as.matrix(a, what = "classes")[2])
accuracy = as.table(as.matrix(a, what = "overall")[1])
bagging= data.frame()
bagging= rbind(bagging,sensitivity)
bagging= rbind(bagging,specificity)
bagging= rbind(bagging,accuracy)
library(pROC)
AUC <- roc(predictor = forecasts_p,
           response = dane_test$opinia)
ROC =AUC$auc
bagging= rbind(bagging,ROC)
colnames(bagging)[1]<-"bagging"
table = cbind(table, bagging)
table = round(table, 3)
table = table[, order(desc(table["ROC",])), drop = FALSE]
table


# MODEL VI 
# random forest


set.seed(123456)
dane.random.forest <- randomForest(modelformula,
                                   data = dane_train)
print(dane.random.forest)
plot(dane.random.forest)

# estimate of out-of-bag error: 40.61%
# ntree= 150-190


set.seed(123456)
dane.rf2 <- randomForest(modelformula,
                         data = dane_train,
                         ntree = 185,
                         sampsize = 1500,
                         mtry = 3,
                         importance = TRUE)
print(dane.rf2)
plot(dane.rf2)


# estimate of out-of-bag error: 40.56%
# ntree= 185


parameters_rf <- expand.grid(mtry = 1:10)

set.seed(123456)
dane.rf3 <- 
  train(modelformula, 
        data = dane_train, 
        method = "rf", 
        ntree = 185,
        tuneGrid = parameters_rf, 
        trControl = tc,
        importance = TRUE)


dane.rf3

# optimal mtry = 4


set.seed(123456)
dane.rf3 <- randomForest(modelformula,
                         data = dane_train,
                         ntree = 185,
                         sampsize = 1500,
                         mtry = 4,
                         importance = TRUE)
dane.rf3

# OOB estimate of  error rate: 40.08%

set.seed(123456)

parameters_ranger <- 
  expand.grid(mtry = 4,
              # split rule
              splitrule = "gini",
              # minium size of the terminal node
              min.node.size = c(50, 80,100,250,300))

set.seed(123456)
dane.rf3a <- 
  train(modelformula, 
        data = dane_train, 
        method = "ranger", 
        num.trees = 185, # default = 500
        # number of threads to use for computations
        num.threads = 4,
        # meausure of importance
        importance = "impurity",
        # parameters
        tuneGrid = parameters_ranger, 
        trControl = tc)

dane.rf3a

# min.node.size = 100


set.seed(123456)
parameters_ranger <- 
  expand.grid(mtry = 4,
              # split rule
              splitrule = "gini",
              # minium size of the terminal node
              min.node.size = 100)

set.seed(123456)
dane.rf3a <- 
  train(modelformula, 
        data = dane_train, 
        method = "ranger", 
        num.trees = 185, # default = 500
        # number of threads to use for computations
        num.threads = 4,
        # meausure of importance
        importance = "impurity",
        # parameters
        tuneGrid = parameters_ranger, 
        trControl = tc)

dane.rf3a

set.seed(123456)
rf_basic = accuracy_ROC(model = dane.random.forest,
                        data = dane_test,
                        target_variable = "opinia",
                        predicted_class = "pos")

rf_tunned = accuracy_ROC(model = dane.rf2,
                         data = dane_test,
                         target_variable = "opinia",
                         predicted_class = "pos")

rf_optimized =accuracy_ROC(model = dane.rf3,
                           data = dane_test,
                           target_variable = "opinia",
                           predicted_class = "pos")

rf_computations = accuracy_ROC(model = dane.rf3a,
                               data = dane_test,
                               target_variable = "opinia",
                               predicted_class = "pos")

rf=cbind(rf_basic,rf_tunned)
rf=cbind(rf,rf_optimized)
rf=cbind(rf,rf_computations)
rf=as.data.frame(rf)
rf[, order(desc(rf["ROC",])), drop = FALSE]

random_forest = rf_computations
table = cbind(table, random_forest)
table = round(table, 3)
table[, order(desc(table["ROC",])), drop = FALSE]

# MODEL VII 
# boosting - GBM

set.seed(123456)
dane_train2 <-
  dane_train[sample(1:nrow(dane_train),
                    size = 3000,
                    replace = FALSE),]

# for distribution = "bernoulli"
# the dependent variable must be coded as 0/1
set.seed(123456)
dane_train$opinia_1 <-
  (dane_train$opinia == "pos") * 1

dane_train2$opinia_1 <-
  (dane_train2$opinia == "pos") * 1

modelformula_01 <- opinia_1 ~  
  cena+ formaty + lnagrod + lstron + plec_autora +plec_recenzenta + rodzaj + rok + stars + tematyka +                 sprzedaz

set.seed(123456)
dane.gbm <- 
  gbm(modelformula_01,
      data = dane_train,
      distribution = "bernoulli",
      # total number of trees
      n.trees = 1000,
      # number of variable interactions - actually depth of the trees
      interaction.depth = 5,
      # shrinkage parameter - speed (pace) of learning
      shrinkage = 0.01,
      verbose = FALSE)

set.seed(123456)
# we generate prediction on the traing data set
dane.gbm_fitted <- predict(dane.gbm,
                           dane_train, 
                           # type = "response" gives in this case 
                           # probability of success
                           type = "response",
                           # n.trees paramters sets the number of iteration
                           # which is used to generate the prediction
                           n.trees = 1000)

set.seed(123456)
# and on the testing set
dane.gbm_prediction <- predict(dane.gbm,
                               dane_test, 
                               type = "response",
                               n.trees = 1000)
source("accuracy_ROC2.R")
set.seed(123456)
# testing data
gbm= accuracy_ROC2(data = data.frame(opinia = dane_test$opinia,
                                     pred = dane.gbm_prediction),
                   predicted_probs = "pred",
                   target_variable = "opinia") 
gbm


# tuning of parameters to find the optimal set of their values. 

# sprawdzone parametry:
#parameters_gbm <- expand.grid(interaction.depth = c(4, 5, 6, 7, 8, 9),
#                              n.trees = c(800, 1000, 1500, 2000, 2500, 3000),
#                              shrinkage = c(0.01, 0.1), 
#                              n.minobsinnode = c(10,30,50,80, 100, 250))


# n.trees = 1000, interaction.depth = 8, shrinkage = 0.01 and n.minobsinnode = 30
# n.trees = 2500, interaction.depth = 5, shrinkage = 0.01 and n.minobsinnode = 50

set.seed(123456)
# Użyte parametry:
parameters_gbm1 <- expand.grid(interaction.depth = c(8),
                               n.trees = c(1000),
                               shrinkage = c(0.01), 
                               n.minobsinnode = c(30))
parameters_gbm2 <- expand.grid(interaction.depth = c(5),
                               n.trees = c(2500),
                               shrinkage = c(0.01), 
                               n.minobsinnode = c(50))

set.seed(123456)
dane.gbm_tuned  <- train(modelformula,
                         data = dane_train,
                         distribution = "bernoulli",
                         method = "gbm",
                         tuneGrid = parameters_gbm1,
                         trControl = tc,
                         verbose = FALSE) 
set.seed(123456)
dane.gbm_tuned2  <- train(modelformula,
                          data = dane_train2,
                          distribution = "bernoulli",
                          method = "gbm",
                          tuneGrid = parameters_gbm2,
                          trControl = tc,
                          verbose = FALSE)
dane.gbm_tuned
dane.gbm_tuned2

# n.trees = 1000, interaction.depth = 8, shrinkage = 0.01 and n.minobsinnode = 30
# n.trees = 2500, interaction.depth = 5, shrinkage = 0.01 and n.minobsinnode = 50

# testing set
set.seed(123456)
gbm_tunned = accuracy_ROC(model = dane.gbm_tuned,
                          data  = dane_test,
                          target_variable = "opinia",
                          predicted_class = "pos")

# testing set
set.seed(123456)
gbm_tunned2 = accuracy_ROC(model = dane.gbm_tuned2,
                           data  = dane_test,
                           target_variable = "opinia",
                           predicted_class = "pos")

model_gbm=cbind(gbm,gbm_tunned)
model_gbm=cbind(model_gbm,gbm_tunned2)
model_gbm=as.data.frame(model_gbm)
model_gbm= model_gbm[, order(desc(model_gbm["ROC",])), drop = FALSE]
model_gbm

table = cbind(table, gbm_tunned)
table = round(table, 3)
table = table[, order(desc(table["ROC",])), drop = FALSE]
table


# MODEL VIII
# neural networks

#standaryzacja zmiennych w zbiorze treningowym

dane_train$cena = scale(dane_train$cena)
dane_train$lnagrod = scale(dane_train$lnagrod)
dane_train$lstron = scale(dane_train$lstron)
dane_train$stars = scale(dane_train$stars)
dane_train$sprzedaz = scale(dane_train$sprzedaz)

dane_test$cena = scale(dane_test$cena)
dane_test$lnagrod = scale(dane_test$lnagrod)
dane_test$lstron = scale(dane_test$lstron)
dane_test$stars = scale(dane_test$stars)
dane_test$sprzedaz = scale(dane_test$sprzedaz)


# TRAINING THE NEURAL NETWORK

set.seed(123456)

# We create a matrix with binary variables and the target variable.
dane_train.mtx <- 
  model.matrix(object = modelformula, 
               data   = dane_train)

# We have to manually correct the variables names.
colnames(dane_train.mtx) <- gsub(" ", "_",  colnames(dane_train.mtx))
colnames(dane_train.mtx) <- gsub(",", "_",  colnames(dane_train.mtx))
colnames(dane_train.mtx) <- gsub("/", "",   colnames(dane_train.mtx))
colnames(dane_train.mtx) <- gsub("-", "_",  colnames(dane_train.mtx))
colnames(dane_train.mtx) <- gsub("'", "",   colnames(dane_train.mtx))
colnames(dane_train.mtx) <- gsub("\\+", "", colnames(dane_train.mtx))
colnames(dane_train.mtx) <- gsub("\\^", "", colnames(dane_train.mtx))
colnames(dane_train.mtx)

col_list <- 
  paste(c(colnames(dane_train.mtx[, -1])), collapse = "+")

# neuralnet() requires the data to be numeric! (no factors!)
col_list <- paste(c("opinia_1 ~ ", col_list), collapse = "")
(modelformula2 <- formula(col_list))

# Modelformula - bez factorowych

opinia_1= dane_train$opinia
opinia_1 = as.factor(ifelse(dane_train$opinia == "pos", 1, 2))
opinia_1 = as.numeric(opinia_1)

dane_train_nn = data.frame(opinia_1,dane_train.mtx)

set.seed(123456)
nn <- 
  neuralnet(modelformula2,
            data = dane_train_nn %>%
              sample_n(4000),
            hidden = c(10, 5), # number of neurons in hidden layers
            linear.output = F, # T for regression, F for classification
            learningrate.limit = NULL,
            learningrate.factor = list(minus = 0.5, plus = 1.5),
            algorithm = "rprop+")

plot(nn)


# PREDICTION ON THE TESTING DATASET

# Next, we create matrix with binary variables and the target variable.
dane_test.mtx <- 
  model.matrix(object = modelformula, 
               data = dane_test)


# We also correct variable names.
colnames(dane_test.mtx) <- gsub(" ", "_",  colnames(dane_test.mtx))
colnames(dane_test.mtx) <- gsub(",", "_",  colnames(dane_test.mtx))
colnames(dane_test.mtx) <- gsub("/", "",   colnames(dane_test.mtx))
colnames(dane_test.mtx) <- gsub("-", "_",  colnames(dane_test.mtx))
colnames(dane_test.mtx) <- gsub("'", "",   colnames(dane_test.mtx))
colnames(dane_test.mtx) <- gsub("\\+", "", colnames(dane_test.mtx))

#colnames(dane_test.mtx)

# To generate predictions, we will use again the compute() function.
# We have to ensure, that in the dataset there is no target variable.
nn.pred <- compute(nn, dane_test.mtx[, -1])
nn.prediction <- as.numeric((nn.pred$net.result > 0.5))

unique(nn.prediction)

# Finally, we can calculate confusion matrix for the testing sample.
(confusionMatrix.nn <-
    confusionMatrix(as.factor(nn.prediction),
                    as.factor(ifelse(dane_test$opinia == "pos", 1, 2))) )

# Neural nets inny sposób kodowania zmiennych, "spowolnienie" uczenia w celu uniknięcia przeuczenia

set.seed(123456)
dane_train <-
  dane_train[sample(1:nrow(dane_train),
                    size = 3500,
                    replace = FALSE), ]

require(neuralnet)

dane_train$opinia <-  ifelse(dane_train$opinia == "pos", 1, 0)
dane_train$plec_autora <-  as.integer(dane_train$plec_autora)
dane_train$plec_recenzenta <-  as.integer(dane_train$plec_recenzenta)
dane_train$rodzaj <-  as.integer(dane_train$rodzaj)
dane_train$tematyka <-  as.integer(dane_train$tematyka)
dane_train$formaty <-  as.integer(dane_train$formaty)

set.seed(123456)
# fit neural network
nn=neuralnet(modelformula,data=dane_train, hidden=3,act.fct = "logistic",
             linear.output = FALSE, threshold=0.01, stepmax=200)

plot(nn)

dane_test$opinia <- ifelse(dane_test$opinia == "pos", 1, 0)
dane_test$plec_autora <-  as.integer(dane_test$plec_autora)
dane_test$plec_recenzenta <-  as.integer(dane_test$plec_recenzenta)
dane_test$rodzaj <-  as.integer(dane_test$rodzaj)
dane_test$tematyka <-  as.integer(dane_test$tematyka)
dane_test$formaty <-  as.integer(dane_test$formaty)

Predict=compute(nn,dane_test[,-5])
prob <- Predict$net.result
pred <- ifelse(prob>0.5, 1, 0)
nn.prediction <- as.numeric((Predict$net.result > 0.5))

# Finally, we can calculate confusion matrix for the testing sample.
(confusionMatrix.nn <-
    confusionMatrix(as.factor(nn.prediction),
                    as.factor(dane_test$opinia) ))

# Sieci neuronowe mają największą jakość dopasowania, ale specyficzność równą 1, wrażliwość równą zero, zatem duża dokładność oszacowań wynika z przeuczenia modelu. Pomimo zastosowania parametrów wpływających na spowolnienie uczenia oraz standaryzacji zmiennych, model jest podatny na przeuczenie.

# Porównanie wyników modeli
table = table[, order(desc(table["ROC",])), drop = FALSE]
mm <- as.matrix(table, ncol = 3)
heatmap.2(x = mm, Rowv = F, Colv = T, dendrogram = "none",
          cellnote = mm, notecol = "black", notecex = 1,
          trace = "none", key = F, margins = c(8, 8), key.xlab = colnames(table), col=hsv(1, seq(0,1,length.out = 12) , 1), cexCol= 0.9)

# PODSUMOWANIE
# Tabela powyżej przedstawia wyniki jakości predykcji poszczególnych modeli. 
# W przypadku tego badania, największą wartość ROC ma model random forest oraz GBM. 
# Najmniejszą nastomiast podstawowe drzewo decyzyjne, przed tunningiem parametrów. 
# Najwyższy wskaźnik Accurancy otrzymano dla modelu "tree_prunning", w którym dokonano optymalizacji parametrów minsplit, 
# minbucket oraz maxdepth. Najwyższy wskaźnik Sensitivity otrzymano dla modelu "tree_prunning", natomiast wskaźnik Specificity 
# dla modeli tree_stop_crit, gbm oraz tree_cv.
