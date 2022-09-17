#Random forest Working professionals - Predict Investment Scores

#Load the libraries 
library(randomForest)
library(ggplot2)
library(tidyverse)
library(cowplot)
library(readxl)

#Set Working Directory
setwd("D:/Project - Financial Literacy")

#Import the dataset
data = read_xlsx("RFDATA.xlsx", sheet = "RFdata-Working")
head(data)

#Encode the variables
str(data)
attach(data)
data$DISTRICT = as.factor(DISTRICT)
data$GENDER = as.factor(GENDER)
data$MARITAL_STATUS = as.factor(MARITAL_STATUS)
data$EDUCATION = as.factor(EDUCATION)
data$WORK_STATUS = as.factor(WORK_STATUS)
data$WORKING_SECTOR = as.factor(WORKING_SECTOR)
data$WORK_PROF_SALARY = as.factor(WORK_PROF_SALARY)
data$GUIDANCE_EXPERTS = as.factor(GUIDANCE_EXPERTS)
data$GEN_SCORE = as.factor(GEN_SCORE)
data$SAV_SCORE = as.factor(SAV_SCORE)
data$INSUR_SCORE = as.factor(INSUR_SCORE)
data$INV_SCORE = as.factor(INV_SCORE)

str(data)

#Build the random forest models with 500 and 1000 trees to compare
set.seed(40)
rfmodel1 <- randomForest(INV_SCORE~., proximity = TRUE ,data=data)
rfmodel1

rfmodel2 <- randomForest(INV_SCORE~., ntree = 1000,data=data)
rfmodel2 

#Compare the models
par(mfrow = c(2,1))
plot(rfmodel1) ; plot(rfmodel2)
par(mfrow = c(1,1))
#Stabilizes at 500
#choose rfmodel1 with no of splits = 3

rfmodel1

#MDS plot
distance.matrix <- as.dist(1-rfmodel1$proximity)

mds.stuff <- cmdscale(distance.matrix, eig=TRUE, x.ret=TRUE)
mds.var.per <- round(mds.stuff$eig/sum(mds.stuff$eig)*100, 1)

mds.values <- mds.stuff$points
mds.data <- data.frame(Sample=rownames(mds.values),
                       X=mds.values[,1],
                       Y=mds.values[,2],
                       Status=data$INV_SCORE)

ggplot(data=mds.data, aes(x=X, y=Y, label=Sample)) + 
  geom_text(aes(color=Status)) +
  theme_bw() +
  xlab(paste("MDS1 - ", mds.var.per[1], "%", sep="")) +
  ylab(paste("MDS2 - ", mds.var.per[2], "%", sep="")) +
  ggtitle("MDS plot using (1 - Random Forest Proximities)")

