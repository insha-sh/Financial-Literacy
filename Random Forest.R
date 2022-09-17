#Random forest - Student Data - Savings Score
library(randomForest)
library(ggplot2)
library(tidyverse)
library(cowplot)
library(readxl)

#Set Working Directory
setwd("D:/Project - Financial Literacy")

#Importing the Dataset
data = read_xlsx("RFDATA.xlsx", sheet = "RFdata-Student")
head(data)

#Encode the categorical variables
str(data)
attach(data)
data$DISTRICT = as.factor(DISTRICT)
data$GENDER = as.factor(GENDER)
data$EDUCATION = as.factor(EDUCATION)
data$WORK_STATUS = as.factor(WORK_STATUS)
data$STU_INC = as.factor(STU_INC)
data$GUIDANCE = as.factor(GUIDANCE)
data$GEN_SCORE = as.factor(GEN_SCORE)
data$SAV_SCORE = as.factor(SAV_SCORE)
data$INSUR_SCORE = as.factor(INSUR_SCORE)
data$INV_SCORE = as.factor(INV_SCORE)
data$PER_FIN_SCORE = as.factor(PER_FIN_SCORE)

str(data)

#Build the random forest model
set.seed(50)
rfmodel1 <- randomForest(SAV_SCORE~., data=data)
rfmodel1

rfmodel2 <- randomForest(SAV_SCORE~., ntree = 1000,data=data)
rfmodel2 

#Compare both the models
par(mfrow = c(2,1))
plot(rfmodel1) ; plot(rfmodel2)
par(mfrow = c(1,1))

#Check the number of splits for which oob errors is minimum
oob.value <- vector(length = 10)
for(i in 1:10){
  temp.model<-randomForest(SAV_SCORE~., data = data, mytri = i, ntree = 1000)
  oob.value[i]<- temp.model$err.rate[nrow(temp.model$err.rate),1]
}

which(oob.value == min(oob.value))
# [4]

#Create the final model
rfmodel_op <- randomForest(SAV_SCORE ~ ., 
                      data=data,
                      ntree=1000, 
                      proximity=TRUE, 
                      mtry=2)

rfmodel_op


#Create the multidimensional plot
distance.matrix <- as.dist(1-rfmodel_op$proximity)

mds.stuff <- cmdscale(distance.matrix, eig=TRUE, x.ret=TRUE)
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)

mds.values <- mds.stuff$points
mds.data <- data.frame(Sample=rownames(mds.values),
                       X=mds.values[,1],
                       Y=mds.values[,2],
                       Status=data$SAV_SCORE)

ggplot(data=mds.data, aes(x=X, y=Y, label=Sample)) + 
  geom_text(aes(color=Status)) +
  theme_bw() +
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep="")) +
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep="")) +
  ggtitle("MDS plot using (1 - Random Forest Proximities)")

