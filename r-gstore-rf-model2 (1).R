## Importing packages

# This R environment comes with all of CRAN and many other helpful packages preinstalled.
# You can see which packages are installed by checking out the kaggle/rstats docker image: 
# https://github.com/kaggle/docker-rstats

library(tidyverse) # metapackage with lots of helpful functions
library(randomForest)
library(caret)
library(e1071)
library(mlbench)
list.files(path = "../input")
#Load data
set.seed(0)
trainSample <- read_csv("../input/trainSampleNumeric.csv")

####################################################################################################
# select dataset (traintest); shuffle traintest: Randomly reorder a dataframe by row; will run for 70/30, 60/40, 75/25
traintest<-trainSample
set.seed(555)
index<-sample(1:nrow(traintest), round(0.70 * nrow(traintest),0))
#               select sample; split into training and test sets

# divide traintest into train and test for modeling 
trainM<-traintest[index,]
testM <- traintest[-index,]
dim(trainM)
dim(testM)

#create train& test ID
trainM.ID<-data.frame(trainM$fullVisitorId, trainM$fullId, trainM$transactionRevenue, trainM$transactionRevenueLog)
testM.ID<-data.frame(testM$fullVisitorId, testM$fullId, testM$transactionRevenue, testM$transactionRevenueLog)

#remove IDs from train and any additional variables derived from or target/predict values
trainM <- trainM  %>%
   select(-fullVisitorId, -fullId, -transactionRevenue)
testM <- testM  %>%
   select(-fullVisitorId, -fullId, -transactionRevenue)
dim(trainM)
dim(testM)   
####################################################################################################
#                   RANDOM FOREST (RF) 
#  - method = "cv": The method used to resample the dataset. 
#  - number = n: Number of folders to create
#  - search = "grid": Use the search grid method. For randomized method, use "grid"
####################################################################################################
# Define the control
trControl <- trainControl(method = "cv",
    number = 5,
    search = "grid")

#build base model  
# from previous runs best base is mtry = 16  from range of 2 to 31
set.seed(1234)                                                      
start <- proc.time()[3]
rf_base <- train(transactionRevenueLog~.,
    data = trainM,
    method = "rf",
    metric = "RMSE",   # "Accuracy" for classification, "RMSE" for regression
    importance=TRUE,
    trControl = trControl)
rf_base_mtry <- rf_base$bestTune$mtry 
# Print the results
print('------------------------------ RF base learner results')
#print(rf_base)
#how did the model do using base mtry?
prediction_base <- predict(rf_base, testM[,1:31])
RMSE <- sqrt(sum((prediction_base - testM$transactionRevenueLog)^2)/length(prediction_base))
paste('Test RMSE:',round(RMSE,2))
paste('% of mean:',round(100*RMSE/mean(testM$transactionRevenueLog),2))   # the lower the % of mean the better
#Test RMSE: 3.28614
residualsM <- testM[,32] - prediction_base
head(prediction_base)     #plot(residualsM)
testM.ID$predicted.base <- prediction_base    # Save the predicted values
testM.ID$residuals.base <- residualsM # Save the residual values
testM$predicted.base <- prediction_base # Save the predicted values
testM$residuals.base <- residualsM # Save the residual values
#summary(rf_base)
#rf_base$bestTune$mtry
#print('-----------------------------------------   best rf   ----------------------------------------------------')
print('------------------------------------    results rf_base')
rf_base
plot(rf_base)
varImp(rf_base)
plot(varImp(rf_base))
print('------------------------------------    rf_base final model')
rf_base$finalModel
plot(rf_base$finalModel)
varImp(rf_base$finalModel)
#plot(varImp(rf_base$finalModel))
print('-----------------------------------------   best rf final  end   ----------------------------------------------------')

#plot(rf_base)
#print('min RMSE =')
#min(rf_base$results$RMSE)
#print('max RMSE =')
#max(rf_base$results$RMSE)
end <- proc.time()[3]  
print(paste("This took ", round(end-start,digits = 1), " seconds", sep = ""))
###### END BASE LINE
#1] "Test RMSE: 3.29" 3.284...
#[1] "% of mean: 55.31
#est it is 16
#[1] "This took 8275.5 seconds"

# TUNE RF base 
start <- proc.time()[3]
#tuneGrid <- expand.grid(.mtry = (2: 10))      
tuneGrid <- expand.grid(.mtry = 8)   #best option      
set.seed(1234)
rf_2 <- train(transactionRevenueLog~.,
    data = trainM,
    method = "rf",
    metric = "RMSE",
    tuneGrid = tuneGrid,
    trControl = trControl,
    importance = TRUE,
    nodesize = 15,
    ntree = 150)
# Print the results
print(' RF tuned learner results')
print('-----------------------------------------   best rf   ----------------------------------------------------')
rf_2
print('------------------------------------     rf_2 final model')
rf_2$finalModel
plot(rf_2$finalModel)
print('-----------------------------------------   best rf  end   ----------------------------------------------------')
#how did the model do using base mtry?
#prediction_1 <-predict(rf_2)       #pewdict training data
#summary(prediction_1)
prediction_2 <- predict(rf_2, testM[,1:31])
residualsB = testM[,32] - prediction_2                      #numeric vector
RMSE <- sqrt(sum((prediction_2 - testM$transactionRevenueLog)^2)/length(prediction_2))
paste('Test RMSE:',RMSE)
paste('% of mean:',round(100*RMSE/mean(testM$transactionRevenueLog),2))
print('-----------------------------------------   test rmse end   ----------------------------------------------------')
testM.ID$predicted <- prediction_2 # Save the predicted values
testM.ID$residuals <- residualsB # Save the residual values
testM$predicted <- prediction_2 # Save the predicted values
testM$residuals <- residualsB # Save the residual values

varImp(rf_2)
plot(varImp(rf_2))
end <- proc.time()[3]  
print(paste("This took ", round(end-start,digits = 1), " seconds", sep = ""))
# best mtry = 8

#maxnodes 
start <- proc.time()[3]
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = 8)
for (maxnodes in c(202:204)) {    #has to have more than 1 value # 5..40 (RMSE decreased)   ## best at 203 = 3.3710
    set.seed(1234)
    rf_maxnode <- train(transactionRevenueLog~.,
        data = trainM,
        method = "rf",
        metric = "RMSE",
        tuneGrid = tuneGrid,
        trControl = trControl,
        importance = TRUE,
        nodesize = 15,
        maxnodes = maxnodes,
        ntree = 150)
    current_iteration <- toString(maxnodes)
    store_maxnode[[current_iteration]] <- rf_maxnode
}
results_rf_maxnode <- resamples(store_maxnode)
print('-----------------------------------------   rf maxnode final model  ----------------------------------------------------')
#summary(rf_maxnode)
#results_rf_maxnode
rf_maxnode$finalModel
plot(rf_maxnode$finalModel)
#store_maxnode
print('-----------------------------------------   rf maxnode end  ----------------------------------------------------')
end <- proc.time()[3]  
print(paste("This took ", round(end-start,digits = 1), " seconds", sep = ""))
####################################################################################################
