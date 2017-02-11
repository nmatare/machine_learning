dimensionReduction <- function(data, title = "average", type = c("PCA", "Kmeans", "Kmode"), plotit = FALSE){	

				# performs dimension reduction
				# data is the df
				# title: specify the suffix of the colnames you wish to reduce 
				# type: type of dimension reduction to perform 

				if(type == "PCA"){

					data <- data[ ,apply(data, 2, var, na.rm = TRUE) != 0] # remove columns where variance is 0
					xPCA <- prcomp(data, scale = TRUE) 
					zPCA <- predict(xPCA)[ ,1:20] #use first 20PCAs #these are now the latent factors
					colnames(zPCA) <- paste("pca_", rep(1:NCOL(zPCA)), "." ,title, sep = "")
					return(zPCA)
				} 

				if(type == "Kmeans"){

					kfit <- mclapply(1:20, mc.cores = detectCores() - 1, function(k) kmeans(scale(data), k)) 
					kbic <- sapply(kfit, kIC,"B"); kaicc <- sapply(kfit, kIC,"A")
					k <-  which.min(kbic) # find optimal k
					print(paste("Optimal number of clusters found at: ", k, sep = ""))

					if(plotit){
						plot(kaicc, xlab = "K", ylab = "IC", ylim = range(c(kaicc,kbic)), bty = "n", type = "l", lwd = 2) # get them on same page
						abline(v = which.min(kaicc))
						lines(kbic, col = 4, lwd = 2)
						abline(v = which.min(kbic), col = 4)
					}

					xclust <- as.matrix(sparse.model.matrix( ~ naref(factor(kfit[[k]]$cluster)))[ ,-1]) # outstanding error, sometimes fails??
					colnames(xclust) <- paste("cluster_", rep(1:NCOL(xclust)), "." ,title, sep = "")
					return(xclust)
				}

				if(type == "Kmode"){

					kfit <- mclapply(1:5, mc.cores = detectCores() - 1, function(k) kmodes(dummies, k)) #cluster players teams based upon Hamming similiarty (similiar to jaccard distance)
					kbic <- sapply(kfit, kIC2,"B")
					k <-  which.min(kbic) 
					xclust <- kfit[[kbest]]$cluster
					print(paste("Optimal number of clusters found at: ", k, sep = ""))

					xclust <- as.matrix(sparse.model.matrix( ~ naref(factor(kfit[[k]]$cluster)))[ ,-1])
					colnames(xclust) <- paste("cluster_", rep(1:NCOL(xclust)), "." ,title, sep = "")
					return(xclust)
				}
}

getMemUsage <- function() paste(round(sum(sapply(ls(envir = globalenv()),function(x){object.size(get(x))})) / 1e8, 2), "GB currently used")

getAUC <- function(data.prob){

				#Gets the AUC of each classification category, where data.prob has both respose and control 
				names <- unique(colnames(data.prob))
				results <- list()
				for(name in names){
					.temp <- as.matrix(data.prob[ ,grep(name, colnames(data.prob))]) #first arg is binary y-var, second arg is prob #adde
					if(all(.temp[ ,1] == 0) | NROW(.temp) == 1){ #if no observations of catgory observed return NULL
						results[[name]] <- NULL
						next
					}
					.roc <- pROC:::roc(.temp[,1], predictor = .temp[,2])
					results[[name]] <- as.numeric(.roc$auc)

					data.frame(.roc$sensitivities, .roc$specificities, .roc$thresholds)

					 
				}
				out <- do.call(cbind, results)
				return(out)
}

getRMSE <- function(y, yhat){ sqrt(mean((y - yhat) ^ 2))}

getRMCE <- function(y, yhat){ sqrt(mean((y - yhat) ^ 4))} #penalizes values that are further away

getCoef <- function(model){

				#Helper function to extract coefs from DMR
				data <- suppressWarnings(data.frame(coef.name = dimnames(coef(model))[[1]], 
									coef.value = as.matrix(coef(model))))[,-1]
				betas <- apply(data, 2, function(x) x[x != 0])
				nonzero <- t(do.call(rbind.fill, (lapply(betas, function(x) as.data.frame(t(x))))))
				nonzero[is.na(nonzero)] <- 0
				colnames(nonzero) <- suppressWarnings(dimnames(coef(model))[[2]])
				return(nonzero)
}

crossValidateTS <- function(data, model, timeCol, predCol, period, cluster = NULL, anchored = TRUE, verbose = FALSE, singular = FALSE, free = NULL, rounds, max.dept, ntree, nodesize, ...)
{
				# Performs leave-one-out cross validation on time series data
				# Depends on 'parallel' library and model specific library 

				# data: sparse matrix of cleaned data
				# model: type of algorithim to use
				# colName: colname in data object that for which period will be taken
				# 		  traditional xts this would be minutes, days, years.
				# predCol: colname of prediction variable (y)
				# timeCol: colname of date variable (will be removed during prediction)
				# period: number of periods too use for training window
				# cluster: cluster object for parallized code 
				# anchored: whether to roll by period, or use all past information
				# singular: where OOS frame is 1 row or multiple (used for by player analysis)
				# free: colnames not penalized when using gamlr
				
				# *Big thanks to QuantStrat authors for template code

				### Environment Setup
			    require(parallel)
				if(is.data.frame(data) | is.data.table(data)) stop("Error: please input a sparse matrix")
				
				dateCol <- grep(timeCol, colnames(data))
				predCol <- which(colnames(data) %in% predCol)

				if(length(dateCol) > 1)	stop("Error: Multiple date columns found in data object")

				if(length(predCol) == 0 | length(dateCol) == 0) stop("Error: Could not find correct date and/or predictor columns")

				data <- data[order(data[, dateCol], decreasing = FALSE), ] #reorder by date in case its not already ordered
				if(verbose)	print(getMemUsage())

				ep <- c(1, which(diff(cumsum(!duplicated(data[ ,dateCol]))) == 1), NROW(data)) #endpoints of each 'period' use xts::ep for xts objects
				
				if(singular) ep <- seq(1, NROW(data), by = 1) # endpoints are each instance

				results <- list()
			    if(anchored) training.start <- ep[1] # uses all residual information

				model <- match.arg(model, c("linear", "RF", "XGB", "BART", "DNN" ,"ARMA"))
				cores <- detectCores() - 1

			    print(paste(" === Starting leave-one-out cross validation via", model, "model === "), sep = "")

			    ### Model Run
			    k <- 1; while(TRUE){

			    	result <- list()

				    if(!anchored) training.start <- ep[k] #starts at first period 
					
					training.end   <- ep[k + period] #ends at first game + period
					testing.end <- ep[k + period + 1]

					if(is.na(training.end) | is.na(testing.end) & k == 1) stop("Error: not enough data given the specified window lengths")

					if(is.na(training.end) | is.na(testing.end)) break# stop if training or testing is beyond last data
				 						
					train.data <- data[training.start:training.end, ,drop = FALSE]
					test.data <- data[(training.end + 1):testing.end, ,drop = FALSE]
		
					x <- train.data[ ,-c(dateCol, predCol), drop = FALSE] # where x remove game_date and data points		
					y <- train.data[ ,predCol, drop = FALSE]

					xx <- test.data[ ,-c(dateCol, predCol), drop = FALSE]
					yy <- test.data[ ,predCol, drop = FALSE]
		
					if(verbose)	print(paste("Training on rows ", training.start, " to ", training.end,  " | ", "Testing on row(s) ", training.end + 1, " to ", testing.end, sep = ""))

					# Model predictions
					switch(model, 
			      
						linear = # LASSO regresssion; Selects AICc because CV cannot be used (time series)
						{

							if(!is.null(free)) free <- which(colnames(x) %in% free) # which colmns are NOT penalized

							if(NCOL(y) == 1){ #single precitor variable so regression 	
								
								require(gamlr)
								LASSO <- gamlr(x, y, family = 'gaussian', verb = FALSE, free = free, ... = ...)

								betas <- as.matrix(coef(LASSO)[which(coef(LASSO) != 0),])
								yhat <- as.matrix(predict(LASSO, newdata = xx)); rownames(yhat) <- rownames(xx) #OOS prediction
								RMSE <- as.matrix(getRMSE(yy, yhat))

							} else if(NCOL(y) > 1){ #multiple predictor variables so logit or mulitnomial logit

								require(distrom)
								DMR <- dmr(cluster, x, y, verb = FALSE, free = free, ... = ...)	
								
								betas <- getCoef(DMR)
								prob <- predict(suppressWarnings(coef(DMR)), xx, type = "response")
								yhat.class <- predict(suppressWarnings(coef(DMR)), xx, type = "class")
								if(!singular) ROC <- getAUC(cbind(yy, prob)) #need at least two obs for ROC; OOS AUC across classes
							}
						},

						XGB =
						{	
							require(xgboost)
							if(NCOL(y) == 1){

								XGBST <- xgboost(data = x, label = y, max_depth = max.dept, nthread = cores, nrounds = rounds, verbose = 0, objective = "reg:linear", ... = ...)
								yhat <- predict(XGBST, xx)
								RMSE <- as.matrix(getRMSE(yy, yhat))

							} else if(NCOL(y) > 1){ 

								y <- as.matrix(y) # convert sparse matrix into dense matrix
								y.transformed <- as.matrix(as.numeric(factor(y %*% 1:NCOL(y), labels = colnames(y)))) # if classification problem, reverse model.matrix for package

								XGBST <- xgboost(data = x, label = y.transformed - 1, num_class = NCOL(y), nthread = cores, # labels must start at 0, thus -1
												 nrounds = rounds, max_depth = max.dept, verbose = 0, objective = "multi:softprob", ... = ...)

								prob <- matrix(predict(XGBST, xx, type = "prob"), ncol = NCOL(y), byrow = TRUE, dimnames = list(c(rownames(xx)), c(colnames(y)))) # predict and change vector to matrix
								yhat.class <- colnames(prob)[apply(prob, 1, which.max)] # simple max probability function	
								if(!singular) ROC <- getAUC(cbind(yy, prob)) # need at least two obs for ROC; OOS AUC across classes

							}

							betas <- xgb.importance(feature_names = colnames(x), model = XGBST)
						},

						BART = ### STILL IN DEV
						{	
							require(BayesTree)
							BART <- bart(	x.train = as.data.frame(as.matrix(x)), y.train = as.vector(y), x.test = as.data.frame(as.matrix(xx)),
											ntree = 100, ndpost = 200, nskip = 100, verbose = TRUE
										)

							bfhat = BART$yhat.test.mean

							as.matrix(getRMSE(yy, bfhat))
							
							plot(BART) # plot bart fit
						},

						DNN = ### STILL IN DEV
						{	#DNN
						},

						SVM = ### STILL IN DEV
						{	#SVM	
						},

						RF = #randomForest
						{
							require(ranger)
							if(NCOL(y) == 1){ #single precitor variable so regression

								y.transformed <- as.matrix(y); colnames(y.transformed) <- "y.transformed" # no factor levels, but need to convert into dense matrix
								RF <- ranger(y.transformed ~., data = cbind.data.frame(y.transformed, as.matrix(x)), probability = FALSE, classification = FALSE,
										 	 num.trees = ntree, write.forest = TRUE, num.threads = cores, importance = 'impurity', verbose = FALSE, ... = ...)

								yhat <- predict(RF, data = as.matrix(xx), type = 'response')$predictions
								RMSE <- as.matrix(getRMSE(yy, yhat))

							} else if(NCOL(y) > 1){ # multiple predictor variables so classification problem

								y <- as.matrix(y) # convert sparse matrix into dense matrix
								y.transformed <- factor(y %*% 1:NCOL(y), labels = colnames(y)) # if classification problem, reverse model.matrix for package

								RF <- ranger(y.transformed ~., data = cbind.data.frame(y.transformed, as.matrix(x)), probability = TRUE, classification = TRUE, verbose = FALSE,
											 num.trees = ntree, min.node.size = nodesize, write.forest = TRUE, num.threads = cores, importance = 'impurity', ... = ...)

								prob <- predict(RF, data = as.matrix(xx), type = 'response')$predictions
								yhat.class <- colnames(prob)[apply(prob, 1, which.max)] #simple max probability function							
								if(!singular) ROC <- getAUC(cbind(yy, prob)) #OOS AUC across classes

							}

							betas <- ranger::importance(RF) # not really betas, vars of most importance
						},

						ARMA =
						{
							require(forecast)
							if(!singular) stop("Cannot run ARMA model on non time-series depedent data")

							ARMA <- auto.arima(as.matrix(y), max.p = 5, max.q = 5, ic = "aicc")
							yhat <- as.matrix(as.numeric(forecast(ARMA, h = 1)$mean)); rownames(yhat) <- rownames(xx)
							RMSE <- as.matrix(getRMSE(yy, yhat))
							betas <- coef(ARMA) #because doesn't have vars of importance 
						}
						
					)

					#storage for training/testing period
					result$training.period <- paste("Rows ", training.start, " to ", training.end, sep = "")
					result$testing.period  <- paste("Row ", training.end + 1, " to ", testing.end, sep = "")

					result$betas <- as.matrix(betas) #vars of most importance

					if(NCOL(y) == 1){

						result$RMSE <- as.matrix(RMSE)  
						result$true_point <- as.matrix(yy) #true point
						result$yhat_point <- as.matrix(yhat) #predicted point	

					} else if(NCOL(y) > 1){ #hence multinomical classificaiton problem

						result$ROC  <- as.matrix(ROC) 
						result$true_class <- as.matrix(yy) # true class
						result$yhat_class <- as.matrix(yhat.class) # predicted class	
						result$yhat_prob  <- as.matrix(prob) # predicted probabilites

					}

					results[[k]] <- result 
					k <- k + 1; gc()
				}

				return(results)
}