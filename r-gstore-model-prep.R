## Importing packages

# This R environment comes with all of CRAN and many other helpful packages preinstalled.
# You can see which packages are installed by checking out the kaggle/rstats docker image: 
# https://github.com/kaggle/docker-rstats

library(tidyverse) # metapackage with lots of helpful functions

## Running code

# In a notebook, you can run a single code cell by clicking in the cell and then hitting 
# the blue arrow to the left, or by clicking in the cell and pressing Shift+Enter. In a script, 
# you can run code by highlighting the code you want to run and then clicking the blue arrow
# at the bottom of this window.

## Reading in files

# You can access files from datasets you've added to this kernel in the "../input/" directory.
# You can see the files added to this kernel by running the code below. 

list.files(path = "../input")

#Load data
set.seed(0)
trainAll <- read_csv("../input/r-gstore-eda-all/trainAll.csv")  
trainSample <- read_csv("../input/r-gstore-eda-sample/trainSample.csv")
predictData <- read_csv("../input/r-gstore-predictdataset/predict.csv")
submit <- read_csv("../input/r-gstore-predictdataset/sample_submission_v2.csv")

#names(trainAll)
#names(trainSample)
#names(predictData)

#reorder variables
trainAll<-trainAll[c('fullVisitorId', 'transactionRevenue','channelGrouping', 'visitNumber', 'browser', 'operatingSystem', 'isMobile', 
                       'deviceCategory', 'continent', 'subContinent', 'country', 'region','city','networkDomain','totalhits', 'pageviews',
                       'source' ,'medium', 'isTrueDirect' ,'year' ,'month' ,'hour' ,'wday', 'yday', 'YM', 
                       'browser_dev' ,'browser_os' ,'browser_chan' ,'chan_os', 'country_medium' ,'country_source', 'dev_chan', 
                       'bounces', 'newVisits', 'transactionRevenueLog')]
trainSample<-trainSample[c('fullVisitorId', 'transactionRevenue', 'channelGrouping', 'visitNumber', 'browser', 'operatingSystem', 'isMobile', 
                       'deviceCategory', 'continent', 'subContinent', 'country', 'region','city','networkDomain','totalhits', 'pageviews',
                       'source' ,'medium', 'isTrueDirect' ,'year' ,'month' ,'hour' ,'wday', 'yday', 'YM', 
                       'browser_dev' ,'browser_os' ,'browser_chan' ,'chan_os', 'country_medium' ,'country_source', 'dev_chan', 
                       'bounces', 'newVisits', 'transactionRevenueLog')]
predictData<-predictData[c('fullVisitorId', 'transactionRevenue', 'channelGrouping', 'visitNumber', 'browser', 'operatingSystem', 'isMobile', 
                       'deviceCategory', 'continent', 'subContinent', 'country', 'region','city','networkDomain','totalhits', 'pageviews',
                       'source' ,'medium', 'isTrueDirect' ,'year' ,'month' ,'hour' ,'wday', 'yday', 'YM', 
                       'browser_dev' ,'browser_os' ,'browser_chan' ,'chan_os', 'country_medium' ,'country_source', 'dev_chan', 
                       'bounces', 'newVisits', 'transactionRevenueLog')]                       
                       
                       
#preserve IDs 
trainAll.id <- data.frame(trainAll$fullVisitorId,trainAll$transactionRevenueLog) 
trainSample.id <- data.frame(trainSample$fullVisitorId,trainSample$transactionRevenueLog)
predictData.id <- data.frame(predictData$fullVisitorId,predictData$transactionRevenueLog)

#row counts
trainAll.rows<-nrow(trainAll)
trainSample.rows<-nrow(trainSample)
predictData.rows<-nrow(predictData)

# merge dataframes
trainAll.index <- 1:trainAll.rows
indx1 <- trainAll.rows +1
#indx1
trainSample.index <- indx1: (trainSample.rows +indx1 -1)
indx2 <- trainSample.rows +indx1
#indx2
predictData.index <- indx2: (predictData.rows +indx2 -1)
max(predictData.index)
traintest <- trainAll %>% 
  bind_rows(trainSample) %>%
  bind_rows(predictData)  

dim(traintest)  
cat(trainAll.rows, '+', trainSample.rows, '+', predictData.rows,'=',(trainAll.rows +trainSample.rows +predictData.rows))

glimpse(traintest)
#convert traintest categorical variables to factor
traintest <- traintest %>% 
   mutate_if(is.character,as.factor)

traintest$fullId<-traintest$fullVisitorId
traintest$fullVisitorId<-as.character(traintest$fullVisitorId)
traintest<-traintest[c('fullVisitorId', 'fullId', 'transactionRevenue','channelGrouping', 'visitNumber', 'browser', 'operatingSystem', 'isMobile', 
                       'deviceCategory', 'continent', 'subContinent', 'country', 'region','city','networkDomain','totalhits', 'pageviews',
                       'source' ,'medium', 'isTrueDirect' ,'year' ,'month' ,'hour' ,'wday', 'yday', 'YM', 
                       'browser_dev' ,'browser_os' ,'browser_chan' ,'chan_os', 'country_medium' ,'country_source', 'dev_chan', 
                       'bounces', 'newVisits', 'transactionRevenueLog')]
str(traintest)
#network domain has ~28K levels; this is too much; removing from dataset; this could cause overfitting
traintest <- subset(traintest, select = -c(networkDomain))

#convert traintest categorical variables to factor
traintest <- traintest %>% 
   mutate_if(is.factor, as.numeric)
glimpse(traintest)


write.csv(traintest[trainAll.index,], "trainAllNumeric.csv", row.names = F)
write.csv(traintest[trainSample.index,], "trainSampleNumeric.csv", row.names = F)
write.csv(traintest[predictData.index,], "predictDataNumeric.csv", row.names = F)
write.csv(submit, "sample_submission_v2.csv", row.names = F)

#row counts verification
nrow(trainAll)
nrow(trainSample)
nrow(predictData)
