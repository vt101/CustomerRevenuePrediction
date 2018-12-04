# ENSEMBLE MODEL
 ##############  IGNORE any XGB.CV / XGBCV for Capstone purposes

library(tidyverse) # metapackage with lots of helpful functions
require(xgboost)
library(lubridate)
library(scales)
library(methods)
library(dplyr)
library(caret)
library(Metrics)  # for rmse calculation; function; rmse(actual, predicted)
library(ggplot2)
library(randomForest)
library(tree)
library(caretEnsemble)      ## ensemble modelling
library(rpart)
require(glmnet)

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

#names(testM)
# convert to an xgb matrix
y_trainM <- trainM$transactionRevenueLog
y_testM <- testM$transactionRevenueLog
dtrainM <- xgb.DMatrix(as.matrix(trainM[,1:31]),label = y_trainM)
dtestM <- xgb.DMatrix(as.matrix(testM[,1:31]),label = y_testM)

set.seed(1234)
watchlistM <- list(train=dtrainM, test=dtestM)

######################### RANDOM FOREST

# Define the controlS
trControl <- trainControl(method = "cv",
    number = 5,
    search = "grid")
rf_tuneGrid <- expand.grid(.mtry = 8) 
set.seed(1234)
rf_best <- train(transactionRevenueLog~.,
    data = trainM,
    method = "rf",
    metric = "RMSE",
    tuneGrid = rf_tuneGrid,
    trControl = trControl,
    maxnodes = 203,    
    importance = TRUE,
    nodesize = 15,
    ntree = 150)
# Print the results
print(' RF tuned learner results')
print('-----------------------------------------   best rf   ----------------------------------------------------')
rf_best
print('-----------------------------------------   best rf $fm  ----------------------------------------------------')
rf_best$finalModel
print('-----------------------------------------   graph best rf $fm   ----------------------------------------------------')
plot(rf_best$finalModel)
print('-----------------------------------------   best RF  end   ----------------------------------------------------')
#how did the model do using base mtry?
prediction_rf <- predict(rf_best, testM[,1:31])
residuals_rf = testM[,32] - prediction_rf                      #numeric vector
RMSE_rf <- sqrt(sum((prediction_rf - testM$transactionRevenueLog)^2)/length(prediction_rf))
paste('Test RMSE rf:',RMSE_rf)
paste('% of mean:',round(100*RMSE_rf/mean(testM$transactionRevenueLog),2))
testM.ID$predicted_rf <- prediction_rf # Save the predicted values
testM.ID$residuals_rf <- residuals_rf # Save the residual values
testM$predicted_rf <- prediction_rf # Save the predicted values
testM$residuals_rf <- residuals_rf # Save the residual values

 ggplot(testM, aes(x = pageviews, y = testM$transactionRevenueLog)) +
  geom_segment(aes(xend = pageviews, yend = predicted_rf), alpha = .2) +  # Lines to connect points
  #geom_point(aes(color = residuals_rf)) +
  #scale_color_gradient2(low = "blue", mid = "white", high = "red") +
  #guides(color = FALSE) +
  geom_point() +  # Points of actual values
  geom_point(aes(y = predicted_rf), shape = 1) +  # Points of predicted values
  theme_bw()
  
#RMSE_rf <- sqrt(sum((prediction_rf - testM$transactionRevenueLog)^2)/length(prediction_rf))
#paste('Test RMSE:',RMSE)
#paste('% of mean:',round(100*RMSE/mean(testM$transactionRevenueLog),4))
#residualsB = testM[,32] - prediction_2                      #numeric vector
#testM.ID$predicted.rf <- prediction_2 # Save the predicted values
#testM.ID$residuals.rf <- residualsB # Save the residual values
#testM.$predicted.rf <- prediction_2 # Save the predicted values
#testM$residuals.rf <- residualsB # Save the residual values
#########################################################################################
#####################      XGBoost
##########################################################################################
set.seed(1234)
watchlist1 <- list(train=dtrainM, test=dtestM)
# booster = "gbtree", objective = "reg:linear", eval_metric = "rmse", eta = "0.1", nthread = "2", gamma = "5", max_depth = "6",
# min_child_weight = "100", subsample = "0.8", colsample_bytree = "0.9", alpha = "25", lambda = "25", prediction = "TRUE", silent = "1"
 
         best_param <- list(booster = "gbtree",
                objective = "reg:linear", 
                eval_metric = "rmse",         
                eta=0.1,
                nthread = 2,
                gamma=5, 
                max_depth= 6,
                min_child_weight=100, 
                subsample=0.8, 
                colsample_bytree=0.9,
                alpha = 25,
                lambda = 25,
                prediction = TRUE,
                silent = 1)
set.seed(1234)
best_xgb <- xgb.train(params = best_param,data = dtrainM, trControl = trControl, nrounds = 1500 ,verbose = 0, watchlist = watchlist1, print_every_n = 500, early_stopping_rounds = 100)
print('-----------------------------------------   best xgb   ----------------------------------------------------')
best_xgb
#best_score<-min(best_xgb$evaluation_log[,test_rmse])
#best_xgb$best_score
#best_score

print('------------------------------------   graph resid pred best xgb   ----------------------------------------------------')
prediction.xgb<-predict(best_xgb,dtestM)
residuals.xgb = y_testM - prediction.xgb                      #numeric vector
plot(residuals.xgb,prediction.xgb)
#
importance_matrix.xgb <- xgb.importance(model = best_xgb)
print(importance_matrix.xgb[1:20])
xgb.plot.importance(importance_matrix = importance_matrix.xgb)

testM.ID$predicted.xgb <- prediction.xgb # Save the predicted values
testM.ID$residuals.xgb <- residuals.xgb # Save the residual values
testM$predicted.xgb <- prediction.xgb # Save the predicted values
testM$residuals.xgb <- residuals.xgb # Save the residual values

  ggplot(testM, aes(x = pageviews, y = testM$transactionRevenueLog)) +
  geom_segment(aes(xend = pageviews, yend = predicted.xgb), alpha = .2) +  # Lines to connect points
  geom_point(aes(color = residuals.xgb)) +
  scale_color_gradient2(low = "blue", mid = "white", high = "red") +
  guides(color = FALSE) +
  geom_point() +  # Points of actual values
  geom_point(aes(y = predicted.xgb), shape = 1) +  # Points of predicted values
  theme_bw()
  
print('-----------------------------------------   best xgb end  ----------------------------------------------------')



#####################################################################################################################################
# Example of Bagging algorithms
#control <- trainControl(method="repeatedcv", number=5, repeats=3)
#seed <- 7
#metric <- "RMSE"
# Bagged CART
#set.seed(1234)
#fit.treebag <- train(Class~., data=dataset, method="treebag", metric=metric, trControl=control)
# Random Forest
#set.seed(seed)
#fit.rf <- train(Class~., data=dataset, method="rf", metric=metric, trControl=control)
# summarize results
#bagging_results <- resamples(list(treebag=fit.treebag, rf=fit.rf))
#summary(bagging_results)
#dotplot(bagging_results)


######################################################### ENSEMBLE   ####################################################################
# since there are different numbers of resamples in each model bagging will not occur
# method used will be weighted average : calculate the ratio of RF RMSE to XGB RMSE

#set.seed(1234)
#myControl <- trainControl(
#                method="cv",
#                index = createFolds(trainM[,32], 5),
#                number=5,
#                savePredictions="all",
#                search = "grid"
#             )

# myControl <- trainControl(method="cv", number=5)
# caretList(
#   transactionRevenueLog ~., data = trainM,
#   methodList=c("glm", "rpart"),
#   trControl=myControl
#   )

#greedyEnsemble <- caretEnsemble(
#                                caretList, 
#                                metric="RMSE",
#                                trControl=myControl   #trainControl(number=2)
#                               )
#summary(greedyEnsemble)
#greedyEnsemble


# caretList(
#   Sepal.Length ~ Sepal.Width,
#   head(iris, 50), methodList=c("lm"),
#   tuneList=list(
#     nnet=caretModelSpec(method="nnet", trace=FALSE, tuneLength=1)
#  ),
#   trControl=myControl
#   )
#   ## End(Not run)



#set.seed(750)
#ratio_xgb = round(best_xgb$best_score/rf_best$results$RMSE, 4)
#ratio_xgb
#ensembleM<-  round(ratio_xgb*best_xgb$best_score + (1-ratio_xgb)*rf_best$results$RMSE, 4)
#ensembleM

# evaluate new predicted values
#how did the model do using base mtry?
#prediction_rf <- predict(rf_best, testM[,1:31])
#residuals_rf = testM[,32] - prediction_rf                      #numeric vector
#RMSE_rf <- sqrt(sum((prediction_rf - testM$transactionRevenueLog)^2)/length(prediction_rf))
#paste('Test RMSE rf:',RMSE_rf)
#paste('% of mean:',round(100*RMSE_rf/mean(testM$transactionRevenueLog),2))
#estM.ID$predicted_rf <- prediction_rf # Save the predicted values
#testM.ID$residuals_rf <- residuals_rf # Save the residual values
#testM$predicted_rf <- prediction_rf # Save the predicted values
#testM$residuals_rf <- residuals_rf # Save the residual values

#
#
# summarize results
#control <- trainControl(method="repeatedcv", number=5, repeats=3)
#bagging_results  <- resamples(trControl=control, list(rf=rf_best, treebag=best_xgb))
#summary(bagging_results)
#dotplot(bagging_results)


#myControl <- trainControl(
#                method="cv",
#                index = createFolds(trainM[,32], 5),
#                number=5,
#                savePredictions="all",
#                search = "grid"
#             )

#xgbTreeGrid <- expand.grid(nthread = 2, nrounds = 1500, max_depth = 6, eta = 0.1, gamma = 5, colsample_bytree = 0.9,  subsample = 0.8, min_child_weight = 100)
#glmnetGridElastic <- expand.grid(.alpha = 1) ## notice the . before the parameter  # , .lambda = 0.009

#modelList <<- caretList(
#                 transactionRevenueLog ~., data = trainM, 
#                  trControl=myControl,
                #methodList=c("glm", "rpart"),
#                  metric="RMSE"  ,
#                  tuneList=list(  ## Do not use custom names in list. Will give prediction error with greedy ensemble. Bug in caret.
#                     xgbTree = caretModelSpec(method="xgbTree",  tuneGrid = xgbTreeGrid, nthread = 2),
#                     glmnet=caretModelSpec(method="glmnet", tuneGrid = glmnetGridElastic)) ## Elastic, highly correlated with lasso and ridge regressions
#                trControl=trainControl(number=5, method = "cv")     
#              )
              
#plot(residuals vs prediction)       


#xyplot(resamples(modelList))

#modelCor(resamples(modelList))

#p <- as.data.frame(predict(modelList, newdata=testM))
#print(p)

# weignted ensemble
#set.seed(750)
#greedyEnsemble <- caretEnsemble(
#                                modelList, 
#                                metric="RMSE",
#                                trControl=trainControl(number=2)
#                               )
#summary(greedyEnsemble)
#greedyEnsemble

#p <- as.data.frame(predict(modelList, newdata=testM))
#head(p)

#################  TH END   ##################################