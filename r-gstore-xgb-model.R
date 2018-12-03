#     XGBoost
library(tidyverse) # metapackage with lots of helpful functions
library(xgboost)
library(lubridate)
library(scales)
library(methods)
library(dplyr)
library(caret)
library(Metrics)  # for rmse calculation; function; rmse(actual, predicted)
library(ggplot2)
list.files(path = "../input")

start_all <- proc.time()[3]  
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

# convert to an xgb matrix
y_trainM <- trainM$transactionRevenueLog
y_testM <- testM$transactionRevenueLog
dtrainM <- xgb.DMatrix(as.matrix(trainM[,1:31]),label = y_trainM)
dtestM <- xgb.DMatrix(as.matrix(testM[,1:31]),label = y_testM)

watchlistM <- list(train=dtrainM, test=dtestM)
###############################################################################
###   CREATE BASE MODEL
# xgb.cv will give best iteration of all n possible rounds
param <- list(booster = "gbtree",
                objective = "reg:linear", 
                eval_metric = "rmse",         
                eta=0.3, 
                gamma=0, 
                min_child_weight=1, 
                subsample=1, 
                colsample_bytree=1)
                
print("BASE CV; BASE MODEL")
start <- proc.time()[3]  
set.seed(1234)
xgbcv <- xgb.cv( params = param, data = dtrainM, nrounds = 1000, nfold = 5, showsd = T, stratified = T, print_every_n = 100, early_stopping_rounds = 100, prediction = TRUE, maximize = F)
# Best iteration:
# [18]	train-rmse:2.782476+0.037629	test-rmse:3.354085+0.124774
# details about cross validation (cv)
print('-----------------------------------------   base xgbcv summary  ----------------------------------------------------')
summary(xgbcv)
print('-----------------------------------------   base xgbcv   ----------------------------------------------------')
xgbcv
print('-----------------------------------------   base xgbcv end  ----------------------------------------------------')
#head(xgbcv3$evaluation_log)
#best_iter<-xgbcv3[min(xgbcv3$evaluation_log[,test_rmse_mean])]
#best_score<-min(xgbcv$evaluation_log[,test_rmse_mean])
#paste('best score',min(xgbcv$evaluation_log[,test_rmse_mean]))
#paste('ntreelimit =',xgbcv$best_ntreelimit)
#paste('best iteration',xgbcv$best_iteration)

set.seed(1234)
xgb <- xgb.train(params = param,data = dtrainM, nrounds = 1000,verbose = 1, watchlist = watchlistM, print_every_n = 100, early_stopping_rounds = 100, prediction = TRUE)
#[17]	train-rmse:2.844673	test-rmse:3.248264
#head(xgbcv3$evaluation_log#names(xgb)
#details about the model
print('-----------------------------------------   xgb   ----------------------------------------------------')
xgb
# plot all the trees
#xgb.plot.tree(model = xgb)  # needs HTM or render=FALSEL
print('-----------------------------------------   xgb tree graph (need html) ----------------------------------------------------')
gr <- xgb.plot.tree(model=xgb, trees=0:1, render=FALSE)
gr
#export_graph(gr, 'tree.pdf', width=1500, height=1900)
#export_graph(gr, 'tree.png', width=1500, height=1900)
#best_iter<-xgb[min(xgb$evaluation_log[,test_rmse])]
#best_score<-min(xgb$evaluation_log[,test_rmse])
#xgb$best_iteration
#xgb$best_score
#best_score

print('-----------------------------------------   xgb prediction ----------------------------------------------------')
# predict
baseM<-predict(xgb,dtestM)
residualsM = y_testM - baseM
testM.ID$predicted.base <- baseM # Save the predicted values
testM.ID$residuals.base <- residualsM # Save the residual values
testM$predicted.base <- baseM # Save the predicted values
testM$residuals.base <- residualsM # Save the residual values

print('-----------------------------------------   xgb prediction vs test - polt  ----------------------------------------------------')
plot(baseM,y_testM)

print('-----------------------------------------   xgb  ggplot pred vs test  ----------------------------------------------------')
options(repr.plot.width=8, repr.plot.height=4)
my_data = as.data.frame(cbind(predicted = baseM,
                            observed = y_testM))
# Plot predictions vs test data
ggplot(my_data,aes(predicted, observed)) + geom_point(color = "darkred", alpha = 0.5) + 
    geom_smooth(method=lm)+ ggtitle('Linear Regression ') + ggtitle(" Prediction vs Test Data") +
      xlab("Predecited ") + ylab("Observed ") + 
        theme(plot.title = element_text(color="darkgreen",size=16,hjust = 0.5),
         axis.text.y = element_text(size=12), axis.text.x = element_text(size=12,hjust=.5),
         axis.title.x = element_text(size=14), axis.title.y = element_text(size=14))                            
        
print('-----------------------------------------   xgb importance ----------------------------------------------------')
importance_matrixM <- xgb.importance(model = xgb)
print(importance_matrixM[1:20])
xgb.plot.importance(importance_matrix = importance_matrixM)
print('-----------------------------------------   xgb end  ----------------------------------------------------')
end <- proc.time()[3]  
print(paste("This took ", round(end-start,digits = 1), " seconds", sep = ""))

###  TUNE
#xgb with cv
start <- proc.time()[3]  
set.seed(1234)
max.depths <- c(5,6,7)
etas <- c(0.01, 0.1)
gammas <-c(0,5)
best_params_xgb <- 0
best_params_xgbcv <- 0
best_score_xgb <- 0
best_score_xgbcv <- 0
best_count_xgb <- 0
best_count_xgbcv <- 0
count <- 1
best_count <- 0
for( depth in 1:length(max.depths)){
    for( num in 1:length(etas)){
        for( gam in 1:length(gammas)){
            set.seed(1234)
            param <- list(booster = "gbtree",
                objective = "reg:linear", 
                eval_metric = "rmse",         
                eta=etas[num],
                nthread = 2,
                gamma=gammas[gam], 
                max_depth=max.depths[depth], 
                min_child_weight=100, 
                subsample=0.8, 
                colsample_bytree=0.9,
                alpha = 25,
                lambda = 25)
            xgbcvT <- xgb.cv( params = param, data = dtrainM, nrounds = 5000, nfold = 5, model = T, showsd = T, stratified = T, print_every_n = 500, early_stopping_rounds = 100, prediction = TRUE, maximize = F)
            xgbT <- xgb.train(params = param,data = dtrainM, nrounds = 1000 ,verbose = 0, watchlist = watchlistM, print_every_n = 500, early_stopping_rounds = 100, prediction = TRUE)
                 
            if(count == 1){
                best_params_xgb <- xgbT$params
                best_score_xgb <- xgbT$best_score
                best_params_xgbcv <- xgbcvT$params
               # best_score_xgbcv <- xgbcvT$evaluation_log[xgbcvT$best_iteration]$test_rmse_mean    # NOTE: xgbcvT$best_score does not exist; not a parameter in xgb.cv   #$evaluation_log[1000]$test_rmse_mean
                best_score_xgbcv <- xgbcvT$evaluation_log[xgbcvT$best_iteration]$test_rmse_mean   
          
                best_xgb <-  xgbT    # xgb.train(params = param,data = dtrainM, nrounds =1000 ,verbose = 0, watchlist = watchlistM, print_every_n = 500, early_stopping_rounds = 100, prediction = TRUE)
                best_xgbcv <- xgbcvT   # xgb.cv( params = param, data = dtrainM, nrounds = 5000, nfold = 5, showsd = T, stratified = T, print_every_n = 500, early_stopping_rounds = 100, prediction = TRUE, maximize = F)
                count <- count + 1
            }
            if( xgbT$best_score < best_score_xgb){
                best_count_xgb <- best_count_xgb + 1
                best_params_xgb <- xgbT$params
                best_score_xgb <- xgbT$best_score
                best_xgb <- xgbT   # xgb.train(params = param,data = dtrainM, nrounds =1000 ,verbose = 0, watchlist = watchlistM, print_every_n = 500, early_stopping_rounds = 100, prediction = TRUE)
            }
            if( xgbcvT$evaluation_log[xgbcvT$best_iteration]$test_rmse_mean < best_score_xgbcv){
                best_count_xgbcv <- best_count_xgbcv + 1
                best_params_xgbcv <- xgbcvT$params
                best_score_xgbcv <- xgbcvT$evaluation_log[xgbcvT$best_iteration]$test_rmse_mean  
                best_xgbcv <- xgbcvT   # xgb.cv( params = param, data = dtrainM, nrounds = 5000, nfold = 5, showsd = T, stratified = T, print_every_n = 500, early_stopping_rounds = 100, prediction = TRUE, maximize = F)
            } 
        }    
    }
}

print('-----------------------------------------   best xgbCV   ----------------------------------------------------')
best_xgbcv
print('-----------------------------------------   best xgbcv end  ----------------------------------------------------')
print('-----------------------------------------   best xgb   ----------------------------------------------------')
best_xgb
print('-----------------------------------------   best xgb end  ----------------------------------------------------')
#######################
print('________________________________ xgb prediction _____________________________________________________')
prediction.xgb<-predict(best_xgb,dtestM)
residuals.xgb = y_testM - prediction.xgb                      #numeric vector
#
importance_matrix.xgb <- xgb.importance(model = best_xgb)
print(importance_matrix.xgb[1:20])
xgb.plot.importance(importance_matrix = importance_matrix.xgb)

testM.ID$predicted.xgb <- prediction.xgb # Save the predicted values
testM.ID$residuals.xgb <- residuals.xgb # Save the residual values
testM$predicted.xgb <- prediction.xgb # Save the predicted values
testM$residuals.xgb <- residuals.xgb # Save the residual values

my_data = as.data.frame(cbind(predicted = prediction.xgb,
                            observed = y_testM))
# Plot predictions vs test data
ggplot(my_data,aes(predicted, observed)) + geom_point(color = "darkred", alpha = 0.5) + 
    geom_smooth(method=lm)+ ggtitle('Linear Regression ') + ggtitle(" Prediction vs Test Data") +
      xlab("Predecited ") + ylab("Observed ") + 
        theme(plot.title = element_text(color="darkgreen",size=16,hjust = 0.5),
         axis.text.y = element_text(size=12), axis.text.x = element_text(size=12,hjust=.5),
         axis.title.x = element_text(size=14), axis.title.y = element_text(size=14))    
         
print('------------------------------------- end xgb prediction   ------------------------------------------------------------')
end <- proc.time()[3]  
print(paste("This took ", round(end-start,digits = 1), " seconds", sep = ""))

end_all <- proc.time()[3]  
print(paste("Total time:", round(end_all-start_all,digits = 1), " seconds", sep = ""))
##############
