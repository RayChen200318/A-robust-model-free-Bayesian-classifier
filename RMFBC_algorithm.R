rm(list = ls())

RMFBC <- function(X_train, Y_train, X_test) {
  C <- levels(Y_train)
  r <- ncol(X_train)
  N_train <- nrow(X_train)
  q <- length(C)
  K_gamma <- r * pi ^ (r / 2) / gamma(r / 2 + 1)
  
  Phi <- vector("list", q)
  for (i in 1:q) {
    Phi[[i]] <- list()
  }
  
  for (i in 1:N_train) {
    class_index <- which(C == Y_train[i])
    Phi[[class_index]] <- rbind(Phi[[class_index]], X_train[i, ])
  }
  
  phi <- numeric(q)
  N <- numeric(q)
  #for (i in 1:q) { 
  #  N[i] <- nrow(Phi[[i]])
  #  phi[i] <- N[i] / N_train
  #}
  # for (i in 1:q) { # flat prior
  # phi[i] <- 1/q 
  # }
  for (i in 1:q) { # Laplace smooth prior
   N[i] <- nrow(Phi[[i]])
   phi[i] <- (N[i] + 1) / (N_train + q)
  }
  
  N_test <- nrow(X_test)
  P <- matrix(0, N_test, q)
  B <- matrix(0, N_test, q)
  
  for (i in 1:N_test) {
    for (k in 1:q) {
      if (nrow(Phi[[k]]) > 0) {
        tau <- as.numeric(knnx.dist(Phi[[k]], X_test[i, , drop = FALSE], k = 1))
        P[i, k] <- 2^-(r * log2(tau+1) + log2((K_gamma * (N_train - 1)) / r) + gamma / log(2))
      } else {
        P[i, k] <- 0
      }
    }
    
    Z <- sum(P[i, ])
    B[i, ] <- P[i, ] * phi / Z
  }
  
  predictions <- numeric(N_test)
  for (i in 1:N_test) {
    theta <- which.max(B[i, ])
    predictions[i] <- C[theta]
  }
  
  return(predictions)
}