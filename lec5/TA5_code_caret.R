rm(list=ls())
library(caret)
library(mlbench)
library(Hmisc)
library(randomForest)

# DGP: 5 true + 45 pure noise variables
n <- 100
p <- 40
sigma <- 1
set.seed(1)
sim <- mlbench.friedman1(n, sd = sigma)
colnames(sim$x) <- c(paste("real", 1:5, sep = ""),
                     paste("bogus", 1:5, sep = ""))
bogus <- matrix(rnorm(n * p), nrow = n)
colnames(bogus) <- paste("bogus", 5+(1:ncol(bogus)), sep = "")
x <- cbind(sim$x, bogus)
y <- sim$y

# predictors centered and scaled
normalization <- preProcess(x)
x <- predict(normalization, x)
x <- as.data.frame(x)
subsets <- c(1:5, 10, 15, 20, 25)

# Fit linear model
set.seed(10)

ctrl <- rfeControl(functions = lmFuncs,
                   method = "repeatedcv",
                   repeats = 5, # do 5 cv's
                   number = 10, # this is default 10-fold cv
                   verbose = TRUE) # FALSE if don't want too much output

lmProfile <- rfe(x, y,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile # this is a list

# Variable names that are picked in the final model
predictors(lmProfile)

# Coefficients in the final model
lmProfile$fit 

# Resample 
head(lmProfile$resample)

# Visualize results
plot(lmProfile)
#trellis.par.set(caretTheme())
plot(lmProfile, type = c("g", "o"))

## Random Forest
# Helper functions, simple version of rfFuncs
rfRFE <-  list(summary = defaultSummary,
               fit = function(x, y, first, last, ...){
                 library(randomForest)
                 randomForest(x, y, importance = first, ...)
                 },
               pred = function(object, x)  predict(object, x),
               rank = function(object, x, y) {
                 vimp <- varImp(object)
                 vimp <- vimp[order(vimp$Overall,decreasing = TRUE),,drop = FALSE]
                 vimp$var <- rownames(vimp)                  
                 vimp
                 },
               selectSize = pickSizeBest,
               selectVar = pickVars)
# `summary' function(obs, pred) to compute performance metrics
rfRFE$summary # defaultSummary or twoClassSummary (for classification with two classes)

# `fit' function trains the model, first=all variables, last=final set
rfRFE$fit

# `pred' function 
rfRFE$pred

# `rank' function returns varImportance ranking
rfRFE$rank

# `selectSize` function: pickSizeBest or pickSizeTolerance
rfRFE$selectSize

# `selectVar' returns variable names ranked
rfRFE$selectVar

# Example usage
ctrl$functions <- rfRFE
ctrl$returnResamp <- "all"
set.seed(10)
rfProfile <- rfe(x, y, sizes = subsets, rfeControl = ctrl)
rfProfile

# Visualizing resampling profile
trellis.par.set(caretTheme())
plot1 <- plot(rfProfile, type = c("g", "o"))
plot2 <- plot(rfProfile, type = c("g", "o"), metric = "Rsquared")
print(plot1, split=c(1,1,1,2), more=TRUE)
print(plot2, split=c(1,2,1,2))

# density plot
plot1 <- xyplot(rfProfile, 
                type = c("g", "p", "smooth"), 
                ylab = "RMSE CV Estimates")
plot2 <- densityplot(rfProfile, 
                     subset = Variables < 5, 
                     adjust = 1.25, 
                     as.table = TRUE, 
                     xlab = "RMSE CV Estimates", 
                     pch = "|")
print(plot1, split=c(1,1,1,2), more=TRUE)
print(plot2, split=c(1,2,1,2))