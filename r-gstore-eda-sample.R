## Importing packages

library(tidyverse) # metapackage with lots of helpful functions
library(magrittr)
library(ggplot2)
library(jsonlite)
library(readr)
library(dplyr)
library(tidyr)
library(data.table)
library(lubridate)  # datetime functions
library(scales)
library(repr)
library(zoo)    # used for yearmon function 
library(mice)
library(highcharter)
library(GGally)   # used for correlation matrix

list.files(path = "../input")
#Load data
set.seed(0)
traintest <- read_csv("../input/train_v1.csv")  

#dimension and view of training data
str(traintest)

# flatten JSON blobs
t_device <- paste("[", paste(traintest$device, collapse = ","), "]") %>% fromJSON(flatten = T)
t_geoNetwork <- paste("[", paste(traintest$geoNetwork, collapse = ","), "]") %>% fromJSON(flatten = T)
t_totals <- paste("[", paste(traintest$totals, collapse = ","), "]") %>% fromJSON(flatten = T)
t_trafficSource <- paste("[", paste(traintest$trafficSource, collapse = ","), "]") %>% fromJSON(flatten = T)

print('DEVICE')   # 16 variables
names(t_geoNetwork)
print('GEO NETWORK')   # 11 variables
names(t_geoNetwork)
print('TOTALS')     # 10 variables;
names(t_totals)  
print('TRAFFIC SOURCE')    #13 variables
names(t_trafficSource)

print('TOTALS')     # 10 variables; time on site will be changed to integer for modeling purposes but are really minutes
# NOTE there will be 2 columns with names hits; changing hits to totalhits.
colnames(t_totals)[2]<-'totalhits'
str(t_totals)
total.cols <- c('visits', 'totalhits', 'pageviews', 'bounces', 'newVisits', 'transactionRevenue')

# convert metric charcter columns to numeric
t_totals[, total.cols] <- sapply(t_totals[, total.cols], as.numeric)

#convert NA to 0;  The missing transactionRevenues must mean that there was no purchase made with the visit.
t_totals$visits[is.na(t_totals$visits)] = 0
t_totals$totalhits[is.na(t_totals$totalhits)] = 0
t_totals$pageviews[is.na(t_totals$pageviews)] = 0
t_totals$bounces[is.na(t_totals$bounces)] = 0
t_totals$newVisits[is.na(t_totals$newVisits)] = 0
t_totals$transactionRevenue[is.na(t_totals$transactionRevenue)] = 0

#add newly formatted dataframes to original test and training dataframes
traintest <- traintest %>%
  cbind(t_device, t_geoNetwork, t_totals, t_trafficSource) %>%
  select(-device, -geoNetwork, -totals, -trafficSource)     #drop the initial json blobs

#Remove temporary data frames
rm(t_device); rm(t_geoNetwork); rm(t_totals); rm(t_trafficSource)

#a glance at the modified dataframe
glimpse(traintest)  

# from above glance you can see NA can be infered in different ways
#defining NA occurences
na_def <- c('unknown.unknown', '(not set)', 'not available in demo dataset', 
             '(not provided)', '(none)', '<NA>')

#Convert values in the data to N/A for the train set
for(col in names(traintest)) {
  set(traintest, i=which(traintest[[col]] %in% na_def), j=col, value=NA)
}

print('CUSTOMIZE DATASET: creating additional dataset sample from whole dataset: shrinking size down to ~40K')
#check if NA values of observations with no purchase fall in line with the bigger picture (missing values for the whole dataset)
nz_traintest<- subset(traintest,traintest$transactionRevenue > 0)  
z_traintest<- subset(traintest,traintest$transactionRevenue <= 0)  
paste('whole training dataset: % of non=zero revenue data=', round(100*nrow(nz_traintest)/nrow(traintest), 2))
set.seed(555)
#when dataset too large; scaling back to < 40K rows
index<-sample(1:nrow(z_traintest), round(0.025 * nrow(z_traintest),0))
traintest.tmp<-z_traintest[index,]
traintest_sample <- rbind(nz_traintest, traintest.tmp) 
paste('sample training dataset: % of non=zero revenue data=', round(100*nrow(nz_traintest)/nrow(traintest_sample), 2))
traintest_all<-traintest
# 2 datasets: traintest_all (whole dataset), traintest_sample (partial train dataset)

#######################################################################################
# deal with prediction data here
# add prediction data to trainset for processing

#######################################################################################
# traintest can be what ever dataset is required
traintest<-traintest_sample
#traintest<-traintest_all
#######################################################################################

#plot missing values traintest
options(repr.plot.height=4)
NAcol <- colSums(is.na(traintest))
NAcount <- sort(colSums(sapply(traintest, is.na)), decreasing = TRUE)
NADF <- data.frame(variable=names(NAcount), missing=NAcount)
NADF$PctMissing <- round(((NADF$missing/nrow(traintest))*100),2)
NADF %>%
    ggplot(aes(x=reorder(variable, PctMissing), y=PctMissing)) +
    geom_bar(stat='identity', fill='steelblue') + coord_flip(y=c(0,110)) +
    labs(x="Features with NA", y="  Percent missing") +
    geom_text(size=2, aes(label=paste0(NADF$PctMissing, "%"), hjust= -0.01))
    
#plot missing values z_traintest
options(repr.plot.height=4)
NAcol <- colSums(is.na(z_traintest))
NAcount <- sort(colSums(sapply(z_traintest, is.na)), decreasing = TRUE)
NADF <- data.frame(variable=names(NAcount), missing=NAcount)
NADF$PctMissing <- round(((NADF$missing/nrow(z_traintest))*100),2)
NADF %>%
    ggplot(aes(x=reorder(variable, PctMissing), y=PctMissing)) +
    geom_bar(stat='identity', fill='steelblue') + coord_flip(y=c(0,110)) +
    labs(x="Features with NA", y=" Non Revenue Producing Observations Percent missing") +
    geom_text(size=2, aes(label=paste0(NADF$PctMissing, "%"), hjust= -0.01))
    
    
#check if NA values of observations with purchases fall in line with the bigger picture (missing values for the whole dataset)
options(repr.plot.height=4)
NAcol <- colSums(is.na(nz_traintest))
NAcount <- sort(colSums(sapply(nz_traintest, is.na)), decreasing = TRUE)
NADF <- data.frame(variable=names(NAcount), missing=NAcount)
NADF$PctMissing <- round(((NADF$missing/nrow(nz_traintest))*100),2)
NADF %>%
    ggplot(aes(x=reorder(variable, PctMissing), y=PctMissing)) +
    geom_bar(stat='identity', fill='steelblue') + coord_flip(y=c(0,110)) +
    labs(x="Features with NA", y=" Revenue Producing Observations Percent missing") +
    geom_text(size=2, aes(label=paste0(NADF$PctMissing, "%"), hjust= -0.01))


# campaignCode- only exists in the train data (according to documentation and observation)
traintest <- subset(traintest, select = -c(campaignCode,sessionId))

#also removing other irrelevant features (according to documentation and observation)
traintest <- subset(traintest, select = -c(adwordsClickInfo.criteriaParameters, networkLocation, referralPath,latitude, longitude, cityId, screenResolution, screenColors, language, flashVersion, mobileDeviceMarketingName, mobileDeviceInfo, mobileInputSelector, mobileDeviceModel,
mobileDeviceBranding, operatingSystemVersion, browserSize, browserVersion,metro, adwordsClickInfo.page, adwordsClickInfo.slot, adwordsClickInfo.gclId, adwordsClickInfo.adNetworkType, adwordsClickInfo.isVideoAd, keyword, campaign,adContent))
print("Missing (75% or more) train data features identified and removed: adwordsClickInfo.criteriaParameters, networkLocation, latitude, longitude, cityId, screenResolution, screenColors, language, flashVersion,")
print("mobileDeviceMarketingName, mobileDeviceInfo, mobileInputSelector, mobileDeviceModel, mobileDeviceBranding, operatingSystemVersion, browserSize, browserVersion, campaignCode, metro,")
print("adwordsClickInfo.page, adwordsClickInfo.slot, adwordsClickInfo.gclId, adwordsClickInfo.adNetworkType, adwordsClickInfo.isVideoAd, keyword, campaign,adContent")


#Feature variance
#column variance content: number of distinct entries within each column
# the higher the number the more distinct values, i.e more variance
feature_sub_grps <- sapply(traintest, n_distinct)
feature_sub_grps

traintest %>% select(-c(transactionRevenue)) %>% 
  map_dfr(n_distinct) %>% 
  gather() %>% 
  ggplot(aes(reorder(key, -value), value)) +
  geom_bar(stat = "identity", fill="steelblue") + 
  scale_y_log10(breaks = c(5, 50, 250, 500, 1000, 10000, 50000)) +
  geom_text(aes(label = value), vjust = 1.6, color = "white", size=2.0) +
  theme_minimal() +
  labs(x = "features", y = "Number of unique values") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
  print('Feature variance displayed.')
#names(traintest)

#remove socialEngagementType and visits - 0 variance
traintest <- subset(traintest, select = -c(visits,socialEngagementType))
print('socialEngagementType & visits removed from train dataset')

str(traintest)

# convert visitStartTime & date to datetime & date
traintest$date <- as_date(as.character(traintest$date))
#date range of analysis
range(traintest$date)
traintest$visitStartTime <- as_datetime(traintest$visitStartTime)

#Feature Engineering: breakdown visitstarttime and date to year month date
traintest$year<-year(as_datetime(traintest$visitStartTime))
traintest$month<-month(as_datetime(traintest$visitStartTime))
traintest$hour<-hour(as_datetime(traintest$visitStartTime))
traintest$wday<-wday(as.Date(traintest$visitStartTime,'%d-%m-%Y'))    # use wday(as.Date(train$date,'%d-%m-%Y')) to get numerical instead of character value
traintest$yday<-yday(as.Date(traintest$visitStartTime,'%d-%m-%Y'))   #year date

# create feature year_month: works; use for fine tuning model later remove year, month, day features to prevent high correlation
traintest$year_month<-as.yearmon(traintest$visitStartTime)
traintest$YM<- as.factor(traintest$year_month)
traintest <- subset(traintest, select = -c(visitStartTime,year_month))

#convert logical values to numeric / integer
traintest$isMobile <- ifelse(traintest$isMobile, 1L, 0L)
traintest$isMobile[is.na(traintest$isMobile)] = 0L
traintest$isTrueDirect <- ifelse(traintest$isTrueDirect, 1L, 0L)
traintest$isTrueDirect[is.na(traintest$isTrueDirect)] = 0L

# updated view of missing values
options(repr.plot.height=4)
NAcol <- colSums(is.na(traintest))
NAcount <- sort(colSums(sapply(traintest, is.na)), decreasing = TRUE)
NADF <- data.frame(variable=names(NAcount), missing=NAcount)
NADF$PctMissing <- round(((NADF$missing/nrow(traintest))*100),2)
NADF %>%
    ggplot(aes(x=reorder(variable, PctMissing), y=PctMissing)) +
    geom_bar(stat='identity', fill='steelblue') + coord_flip(y=c(0,110)) +
    labs(x="Features with NA", y=" Percent missing") +
    geom_text(size=2, aes(label=paste0(NADF$PctMissing, "%"), hjust= -0.01))
    
#see features are still missing values
colSums(is.na(traintest)) 
print('There are no numeric missing values.')  
print('Categorical missing values for browser (8), source (69), continent (1468), subContinent (1468), country (1468) and operating system (4695) range from 0.01% - 0.52%.')
print('Features with 5% or less missing data will be imputed. All others will be assigned the values missing')    

#####################################################################################################################
#do EDA visulaizations here
#transactions
#visitor metrics
#non visitor metrics

print('*****************        Review of data - traget feature    ***************')
summary(traintest$transactionRevenue)
print('*****************        distribution is right skewed; mean is greater than median     *************')

#For EDA purposes only -- transaction value more readable
#Divide transactionRevenue by 1,000,000 (1e+06)
traintest$revenue<-traintest$transactionRevenue/ 1e+06  #1000000
summary(traintest$revenue)
print('*****************        distribution is right skewed; mean is greater than median     *************')

# since transactionRevenue is so large the recommendation is to transform feature to log
traintest$transactionRevenue1 <- log(traintest$transactionRevenue)
traintest$transactionRevenue2 <- log1p(traintest$transactionRevenue)

# add new column to dataset current code to identify non zero transactions
traintest$purchaseYN<-0
traintest$purchaseYN[(traintest$transactionRevenue > 0)] = 1

#split transactions into zero and non-zero transactions
nz_traintest<- subset(traintest,traintest$transactionRevenue > 0)  
z_traintest<- subset(traintest,traintest$transactionRevenue <= 0)  

# About visitors: count the total revenue, total visits and total pageviews per customer/visitor
visitor<-data.frame(traintest %>%
                      group_by(fullVisitorId) %>%
                      summarize(tot_pageviews=sum(pageviews), tot_revenue=sum(revenue),tot_revenueLog=sum(log(transactionRevenue)), tot_numOfVisits= max(visitNumber), tot_newVisits= sum(newVisits), tot_hits= sum(totalhits), tot_bounces= sum(bounces), spender=max(purchaseYN)))
# any NAs
anyNA(visitor)

#####################################################################################################################

#visualizaton
hist(traintest$revenue)

# visulaize log(transactionRevenue)
traintest %>% 
  ggplot(aes(x=log(transactionRevenue), y=..density..)) + 
  geom_histogram(fill='steelblue', na.rm=TRUE, bins=40) + 
  geom_density(aes(x=log(transactionRevenue)), fill='orange', color='orange', alpha=0.3, na.rm=TRUE) + 
  labs(
    title = 'Distribution of transaction revenue',
    x = 'Natural Log of transaction revenue'
  )

print('Distribution tending towards normal when using the natural log of the target variable')

# calculating skewness using (e1071 package)
library(e1071) 
print('transaction revenue skewness actual vs log vs log1p')
skewness(traintest$transactionRevenue)
skewness(traintest$transactionRevenue1)
skewness(traintest$transactionRevenue2)

# there are few very large transactions compared to the numerous much smaller transactions
print('transaction revenue skewness for purchases only actual vs log vs log1p')
skewness(nz_traintest$transactionRevenue)
skewness(nz_traintest$transactionRevenue1)
skewness(nz_traintest$transactionRevenue2)

#distribution is positively skewed tending towards normal

# analysis of nondivide data into purchases and no-purchases for further analysis
print('Base Observations')
paste('Total number of observations/visits =', nrow(traintest))
paste('Total unique visitors =', total_visitors<-nrow(visitor))
paste('Visitors who made at least one purchase =', total_spender<-sum(visitor$spender))
paste('Total number of revenue generating visits = ',sum(traintest$purchaseYN))
paste('Only',round(100 * sum(traintest$purchaseYN)/nrow(traintest), 2),'% of visits generated revenue')
paste('Total new visits =',total_newVisits<-sum(visitor$tot_newVisits))
paste('Total revenue generated (/10e6) =', total_revenue<-sum(traintest$revenue))
paste('Total page views =', total_pageviews<-sum(visitor$tot_pageviews))

paste('Total Hits =', total_hits<-sum(visitor$tot_hits))
paste('Total bounces =', total_bounces<-sum(visitor$tot_bounces))

#summarize non zero transactions  
summary(nz_traintest)
#summarize zero transactions
summary(z_traintest)

#visulaization of retruning visitors; % visitor retention overtime
repeat_visitors<-traintest%>%
group_by(visitNumber)%>%
summarize(Current_visit=n())%>%filter(visitNumber<=20)%>%
mutate( Next_visit= lead(Current_visit, 1), next_visit_= lead(visitNumber, 1))%>%
mutate(Visits= paste(visitNumber, "to", next_visit_))%>%
mutate( retention= round( Next_visit/Current_visit,2))%>%
select(Visits,Current_visit,Next_visit, retention)

repeat_visitors

#Revenue per visit
y<-traintest$revenue
traintest %>% 
  bind_cols(as_tibble(y)) %>% 
  group_by(visitNumber) %>% 
  summarise(revenue = sum(transactionRevenue/1e+06)) %>%
  ggplot(aes(x = visitNumber, y = revenue)) +
  geom_point(color="steelblue", size=0.5) +
  theme_minimal() +
  scale_x_continuous(breaks=c(1,2, 3, 5, 10, 15, 25, 50, 100), limits=c(0, 105))+
  scale_y_continuous(labels = comma)
  
  #plot graph to show purchases only: revenue vs time, revenue vs country, continent, visits
#revenue vs number of visits
#country 

y<-traintest$revenue
p1 <- traintest %>% 
  bind_cols(as_tibble(y)) %>% 
  mutate(date = ymd(date)) %>% 
  group_by(date) %>% 
  summarize(visits = n()) %>% 
  ungroup() %>% 
  ggplot(aes(x = date, y = visits)) + 
  geom_line() +
  geom_smooth() + 
  labs(x = "") +
  theme_minimal()
plot(p1)

# time series: transaction revenue
p2 <- traintest %>% 
  bind_cols(as_tibble(y)) %>% 
  mutate(date = ymd(date)) %>% 
  group_by(date) %>% 
  summarize(revenue = mean(value)) %>% 
  ungroup()  %>% 
  ggplot(aes(x = date, y = revenue)) + 
  geom_line() +
  stat_smooth() +
  labs(x = "") +
  theme_minimal()
plot(p2)

#do EDA visulaizations here
#transactions
#visitor metrics
#non visitor metrics


#CLEAN UP
#remove additional features developed for EDA
traintest<-traintest %>%
   select(-purchaseYN, -transactionRevenue1, -transactionRevenue2, -revenue, -date)    

#Remove temporary data frames
rm(nz_traintest); rm(z_traintest); rm(visitor)
traintest$transactionRevenueLog<-log1p(traintest$transactionRevenue)

# At this point only categorical variables need imputation
# There are no numeric missing values.Features with 5% or less missing data will be imputed. 
# All others will be assigned the values missing
print('Features with 5% or less missing data will be imputed. All other categorical NAs will be replaced with the word Missing')
traintest$networkDomain[is.na(traintest$networkDomain)] <- "Missing" 
traintest$medium[is.na(traintest$medium)] <- "Missing" 
traintest$city[is.na(traintest$city)] <- "Missing"
traintest$region[is.na(traintest$region)] <- "Missing" 

traintest$browser[is.na(traintest$browser)] <- "Missing" 
traintest$source[is.na(traintest$source)] <- "Missing" 
traintest$country[is.na(traintest$country)] <- "Missing" 
traintest$subContinent[is.na(traintest$subContinent)] <- "Missing" 
traintest$operatingSystem[is.na(traintest$operatingSystem)] <- "Missing" 
traintest$continent[is.na(traintest$continent)] <- "Missing" 
colSums(is.na(traintest)) 

#FEATURE ENGINEERING
multi.dim <- function(x) n_distinct(x) > 1  # from KXX Group XBG Gstore v2
#create new columns
traintest <-traintest  %>%
  mutate(browser_dev = str_c(browser, "_", deviceCategory),
         browser_os = str_c(browser, "_", operatingSystem),
         browser_chan = str_c(browser,  "_", channelGrouping),
         chan_os = str_c(operatingSystem, "_", channelGrouping),
         country_medium = str_c(country, "_", medium),
         country_source = str_c(country, "_", source),
         dev_chan = str_c(deviceCategory, "_", channelGrouping))
         
#prepare for correlation & modeling
# backup traintest
traintest.backup<-traintest
#list id features
traintest.id<-data.frame(traintest$visitId, traintest$fullVisitorId)

#convert all character columns to factor except for the traintest.id features
traintest.cor <-traintest  %>%
   select(-visitId, -fullVisitorId) %>%
   mutate_if(is.character, factor)

#keeping vector of categorical features 
traintest_categorical_feature <- names(Filter(is.factor,traintest.cor))
traintest_categorical_feature       

#view traintest.cor
str(traintest.cor)
#network domain has ~28K levels; this is too much; removing from dataset; this could cause overfitting
traintest.cor <- subset(traintest.cor, select = -c(networkDomain))

# Label encoding: convert traintest.cor to numeric values
traintest.cor.numeric <- traintest.cor %>% 
  mutate_if(is.factor, as.integer) 
glimpse(traintest.cor.numeric)

#correlation matrix
print('traintest.coef')
traintest.coef<-round(cor(traintest.cor.numeric), 2)
traintest.coef
class(traintest.coef)

#select features for correlation
review<- traintest.coef[,4:18]
review

ggcorr(traintest.cor.numeric, label = TRUE,label_round = 1, label_size = 2)     #,geom = "circle",nbreaks = 5   

#####NOTES
# No modifications to dataset based on correlation results : the ensemble methods will address this issue
# channel grouping is highly correlated to source(.87) and medium(.77) .. source maybe will be removed
# YM is highly correlated with year(.86), month(.87) and yday(.87) ..year will be removed
# isMobile is highly correlated with device category(.94) 
##glimpse(traintest)  # YM is a factor; converting to character
traintest$YM<-as.character(traintest$YM)
## save traintest sample     
write.csv(traintest, "trainSample.csv", row.names = F)
#glimpse(traintest)

