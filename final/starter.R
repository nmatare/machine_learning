# machine learning starter script
options(width = 230)

# load required packages
library(Matrix); library(data.table)
library(xgboost); library(ranger); library(h2o); library(parallel); library(gamlr); library(parallel)

# source data and functions
download.file('https://www.dropbox.com/s/si4ha9gk4ptptgh/data-matchup.rds?dl=1', destfile = "data-matchup-norm.rds", method = "auto") # download matchup 
# download.file('https://www.dropbox.com/s/a3pqoik37ey0lzf/data-scale.rds?dl=1', destfile = "data.scale.rds", method = "auto") # download standardized data
download.file('https://www.dropbox.com/s/gw5x1z3pg8cf7hb/data-stats.rds?dl=1', destfile = "data.norm.rds", method = "auto") # download norm data
source('https://www.dropbox.com/s/cx5jalemtzvyhch/functions.R?dl=1') # get custom utility scripts 

# this data is not scaled or normalized!
data <- readRDS('data.norm.rds') 
# this is the matchup information in range [-1, 0, 1]; simple cBind will work to match them together ie newdat <- cbind(data, as.matrix(matchup))
matchup <- readRDS('data-matchup-norm.rds') 
# this data is scaled between [0-1] for all continous vars in the range [0 - Inf], and scaled between [-1, 1] for all continous vars in the range [-Inf, Inf]
# data <- readRDS('data.scale.rds') # in case you want scaled data

# let's explore data
colnames(data)
dim(data) # its huge, cut down
data <- data[1:10000, ] # uncomment to use all
# all features are coded so you can grep them easily like this:
# all the predictor variables you can try to predict
data[ ,grep(".y_var", colnames(data)), with = FALSE]
# NLP/text data!
data[ ,grep(".text", colnames(data)), with = FALSE]
# lets look at the points scored by the opposing team
data[ ,grep("points.stat_opposing_team", colnames(data)), with = FALSE] 

# only inspect games where the player actually played?
data[game_played == TRUE] 

# take a look at player stat variables
# look at correlations among some vars
cor(data[ ,points.stat_player.game.1], data[,minutes.stat_player.game.1]) 
#points are correlated to minutes!
plot(data[ ,points.stat_player.game.1], data[,minutes.stat_player.game.1]) 

# maybe I only want to run a per player model ??, 
data[full_name == "Brandon Knight"]

# lets look at the positions available
colnames(data)[grep("position", colnames(data))]
# maybe I only want to run a model on centers
data[game_position.bioC == 1] # 1/0 dummy var

# lets look at the gamedays
head(data[,game_date])

# lets make this into games played by day
data[ ,game_date := as.numeric(as.Date(game_date, origin = '1970-01-01', tz = "UTC"))]
# lets order by gameday
data <- data[order(game_date)]

# total number of gamedays we have available 
g <- dim(unique(data[ ,grep('^game_date$', colnames(data)), with = FALSE]))[1]
g

# lets run some models
# I remove all the predictor vars except for the one I want to predict
data[ ,grep(".y_var", colnames(data)), with = FALSE] # all the predictor vars
# lets predict points ^ and $ is just REGEX language specifiing the start and stop of the words
y <- "^points.y_var$" 
Y <- data[ ,grep(y, colnames(data)), with = FALSE]

# remove all the old predictor vars
newdat <- data[ ,!grep(".y_var", colnames(data)), with = FALSE]

# remove game_played because we won't know this at time t
newdat <- newdat[ ,!grep("^game_played$", colnames(newdat)), with = FALSE]
newdat <- newdat[ ,!grep("^full_name$", colnames(newdat)), with = FALSE] # remove noise

newdat <- cbind(Y, newdat)
newdat # cool now I can use everything else to predict 

# cross validate models
# Example 1
# type 'crossValidateTS' into the console and look at the notes to get a better sense of what each option does
# models currently supported are "LM", "RF", "XGB", "FNN.h2o","ARMA"
# LM = gamlr, RF = ranger, XGB = xgboost, FNN.h2o = deeplearning.h2o, ARMA = auto.arima
# type ? to input their arguments; aka ?gamlr

# I will train on the past 40 games and then predict the next game until I run out of data
eval <- crossValidateTS(
			data = as.matrix(newdat), # specify the where the data is coming from; YOU MUST convert to matrix first
			predCol = "points.y_var", # this is the name of the predictor column
			timeCol = "game_date", # each period is one game and is determined by this column game_date
			period = 570, # how many periods to train on, (aka 570 games) because one period is one day (game_date); will predict on next periods (game_days) game
			verbose = TRUE, 
			anchored = FALSE, # whether to roll the training window or keep it fixed at observation 1
			
			# specify model parameters ?gamlr for this example
			cluster = NULL,		
			model = 'LM', # use a linear model (LASSO), and specify the parameters
			free = NULL, 
			lamba.min.ratio = 1e-9, 
			lambda.start = 1.00, 
			gamma = 0
			# note the function will not use timeCol as a predictor column [ it gets removed]
)

#lets look at our out-of-sample predictions
names(eval[[1]])
eval[[1]]$testing.period
eval[[1]]$yhat_point # predicted value
eval[[1]]$true_point # true value
eval[[1]]$RMSE # RMSE
coef(eval[[1]]$model) # let's grab the betas from this model
plot(eval[[1]]$model) # let's look at LASSO path
# finally, let's loop through all out-of-sample periods and get the RMSEs
RMSEs <- lapply(eval, function(x) x$RMSE)
mean(do.call(rbind, RMSEs)) # looks like an average RMSE of 4.2!

# Example 2
eval <- crossValidateTS(
			data = as.matrix(newdat), # specify the where the data is coming from; YOU MUST convert to matrix first
			predCol = "points.y_var", # this is the name of the predictor column
			timeCol = "game_date", # each period is one game and is determined by this column game_date
			period = g - 2, # or you could say I want 2 out of sample predictions, where g is the total number of games in data; here i'll get two out of same tests
			verbose = TRUE, 
			anchored = TRUE, # whether to roll the training window or keep it fixed at observation 1
			
			# specify model parameters ?ranger for this example
			model = 'RF', # use trees this time
			ntree = 100, nodesize = 5
			# note the function will not use timeCol as a predictor column [ it gets removed]
)

RMSEs <- lapply(eval, function(x) x$RMSE); RMSEs
# lets look at the second periods model
eval[[2]]$model 

# if you are uncomfortable with what's going on behind this function; this is essentially what is happening, but in a for loop:
data <- data[order(game_date)] #reorder by date in case its not already ordered
data <- data[ ,-which(colnames(data) %in% c('game_date')), with = FALSE]# remove the game date column 

# I remove all the predictor vars except for the one I want to predict
data[ ,grep(".y_var", colnames(data)), with = FALSE] # all the predictor vars
# lets predict points ^ and $ is just REGEX language specifiing the start and stop of the words
y <- "^points.y_var$" 
Y <- data[ ,grep(y, colnames(data)), with = FALSE]

# remove all the old predictor vars
newdat <- data[ ,!grep(".y_var", colnames(data)), with = FALSE]

# remove game_played because we won't know this at time t
newdat <- newdat[ ,!grep("^game_played$", colnames(newdat)), with = FALSE]
newdat <- newdat[ ,!grep("^full_name$", colnames(newdat)), with = FALSE] # remove noise

newdat <- cbind(Y, newdat)
newdat # cool now I can use everything else to predict 

splitdat <- 0.01 # use 1% for testing
train.idx <- 1:(NROW(newdat) * (1 - splitdat)) # in crossValidate this moves by one period forward in time
validate.idx <- (tail(train.idx, 1) + 1):NROW(newdat) # in crossValidate this moves by one period forward in time

train <- newdat[train.idx, ]
validate <- newdat[validate.idx, ]
test <- NULL # real life prediction

linear <- lm(points.y_var ~., data = train) # sample model
yhats <- predict(linear, validate)
RMSE <- sqrt(mean((validate[ ,points.y_var] - yhats) ^ 2))
RMSE # OOS prediction
