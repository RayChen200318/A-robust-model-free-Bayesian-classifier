library(caret)
library(adabag)
library(datasets)
library(class)
library(bnlearn)
library(e1071)
library(FNN)
library(MASS)
library(fastICA)
library(progress)

rm(list = ls())

total_steps <- 10
pb <- progress_bar$new(
  format = "  处理中 [:bar] :percent :elapsed 剩余时间: :eta",
  total = total_steps, clear = FALSE, width = 60
)


letter <- read.csv("C:/Users/Ray Chen/Desktop/letter-recognition.data",header = FALSE)
lettertype <- letter[,1]
letter <- letter[,-1]
letter <- cbind(letter, lettertype)
letter$lettertype <- as.factor(letter$lettertype)
for (i in 1:(ncol(letter)-1)) {
  letter[,i] <- as.numeric(letter[,i])
}

# Instance clone process

str(letter)

# PCA
pca_result <- prcomp(letter[, 1:(ncol(letter)-1)], center = TRUE, scale. = TRUE)
summary(pca_result)
pca_features <- as.data.frame(pca_result$x[,1:9])


nn <- 1e6
gamma <- sum(1 / (1:nn)) - log(nn)
print(gamma)

MFBC <- function(X_train, Y_train, X_test) {
  C <- levels(Y_train)
  r <- ncol(X_train)
  N_train <- nrow(X_train)
  q <- length(C)
  K_gamma <- r * pi ^ (r / 2) / gamma(r / 2 + 1)
  # 初始化Phi
  Phi <- vector("list", q)
  for (i in 1:q) {
    Phi[[i]] <- list()
  }
  
  # 将训练样本按照类别放入Phi
  for (i in 1:N_train) {
    class_index <- which(C == Y_train[i])
    Phi[[class_index]] <- rbind(Phi[[class_index]], X_train[i, ])
  }
  
  # 计算每个类别的phi值
  phi <- numeric(q)
  N <- numeric(q)
  for (i in 1:q) { #频率先验
    N[i] <- nrow(Phi[[i]])
    phi[i] <- N[i] / N_train
  }
  # for (i in 1:q) { # flat prior
  # phi[i] <- 1/q 
  # }
  #for (i in 1:q) { # Laplace smooth prior
  #  N[i] <- nrow(Phi[[i]])
  #  phi[i] <- (N[i] + 1) / (N_train + q)
  #}
  
  # 初始化P和B矩阵
  N_test <- nrow(X_test)
  P <- matrix(0, N_test, q)
  B <- matrix(0, N_test, q)
  
  # 计算P和B
  for (i in 1:N_test) {
    for (k in 1:q) {
      if (nrow(Phi[[k]]) > 0) {
        tau <- as.numeric(knnx.dist(Phi[[k]], X_test[i, , drop = FALSE], k = 1))
        #tau <- as.numeric(mean(knnx.dist(Phi[[k]], X_test[i, , drop = FALSE], k = 3)))
        P[i, k] <- 2^-(r * log2(tau + 1) + log2((K_gamma * (N[k] - 1)) / r) + gamma / log(2))
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

max_it <- 10
set.seed(123)
folds <- createFolds(letter$lettertype, k = max_it, list = TRUE, returnTrain = TRUE)

accuracy_mfbc <- numeric(max_it)
accuracy_nbc <- numeric(max_it)
accuracy_pca_nbc <- numeric(max_it)
accuracy_nn <- numeric(max_it)
accuracy_knn <- numeric(max_it)
accuracy_svm<- numeric(max_it)
accuracy_pca_mfbc <- numeric(max_it)


m <- 1
for (m in 1:max_it) {
  pb$tick()
  #folds <- createFolds(letter$lettertype, k = max_it, list = TRUE, returnTrain = TRUE)
  index <- folds[[m]]
  X_train <- letter[index, 1:(ncol(letter)-1)]
  X_test <- letter[-index, 1:(ncol(letter)-1)]
  Y_train <- letter[index, (ncol(letter))]
  Y_test <- letter[-index, (ncol(letter))]
  X_train_pca <- pca_features[index, 1:ncol(pca_features)]
  X_test_pca <- pca_features[-index, 1:ncol(pca_features)]
  
  # kNN
  k <- 3
  knn_pred <- knn(train = X_train, test = X_test, cl = Y_train, k = k)
  
  confusion_matrix_kNN <- table(knn_pred, Y_test)
  accuracy_knn[m] <- sum(diag(confusion_matrix_kNN)) / sum(confusion_matrix_kNN)
  
  # NN
  k <- 1
  nn_pred <- knn(train = X_train, test = X_test, cl = Y_train, k = k)
  
  confusion_matrix_NN <- table(nn_pred, Y_test)
  accuracy_nn[m] <- sum(diag(confusion_matrix_NN)) / sum(confusion_matrix_NN)
  
  # NBC
  nbc_model <- naiveBayes(Y_train ~ ., data = cbind(X_train, Y_train))
  
  nbc_pred <- predict(nbc_model, cbind(X_test, Y_test))
  
  confusion_matrix_nbc <- table(nbc_pred, Y_test)
  
  accuracy_nbc[m] <- sum(diag(confusion_matrix_nbc)) / sum(confusion_matrix_nbc)
  
  # PCA + NBC
  
  pca_nbc_model <- naiveBayes(Y_train ~ ., data = cbind(X_train_pca, Y_train))
  
  pca_nbc_pred <- predict(pca_nbc_model, cbind(X_test_pca, Y_test))
  
  confusion_matrix_pcanbc <- table(pca_nbc_pred, Y_test)
  
  accuracy_pca_nbc[m] <- sum(diag(confusion_matrix_pcanbc)) / sum(confusion_matrix_pcanbc)
  
  # SVM
  svm_model <- svm(Y_train ~ ., data = cbind(X_train, Y_train), kernel = "linear", cost = 1, scale = TRUE)
  
  svm_pred <- predict(svm_model, cbind(X_test, Y_test))
  confusion_matrix_svm <- table(Predicted = svm_pred, Actual = Y_test)
  
  accuracy_svm[m] <- sum(diag(confusion_matrix_svm)) / sum(confusion_matrix_svm)
  
  # LDA
  #lda_model <- lda(Y_train ~ ., data = cbind(X_train, Y_train))
  
  #lda_pred <- predict(lda_model, newdata = cbind(X_test, Y_test))
  #confusion_matrix_lda <- table(Predicted = lda_pred, Actual = Y_test)
  
  #accuracy_lda[m] <- sum(diag(confusion_matrix_lda)) / sum(confusion_matrix_lda)
  
  # MFBC
  mfbc_pred <- MFBC(X_train, Y_train, X_test)
  
  confusion_matrix_mfbc <- table(mfbc_pred, Y_test)
  accuracy_mfbc[m] <- sum(diag(confusion_matrix_mfbc)) / sum(confusion_matrix_mfbc)
  
  # MFBC + PCA
  mfbc_pca_pred <- MFBC(X_train_pca, Y_train, X_test_pca)
  
  confusion_matrix_mfbc_pca <- table(mfbc_pca_pred, Y_test)
  accuracy_pca_mfbc[m] <- sum(diag(confusion_matrix_mfbc_pca)) / sum(confusion_matrix_mfbc_pca)
  
}

accuracy_matrix <- cbind(accuracy_knn,accuracy_nn,accuracy_svm,accuracy_nbc,accuracy_pca_nbc,accuracy_mfbc,accuracy_pca_mfbc)
accuracy_matrix

output <- matrix(0,7,ncol(accuracy_matrix))
colnames(output) <- colnames(accuracy_matrix)
for (i in 1:(ncol(accuracy_matrix))) {
  output[,i] <- sort(accuracy_matrix[, i],decreasing = TRUE)[1:7]
}
print(output)

mu <- matrix(0,1,ncol(accuracy_matrix))
colnames(mu) <- colnames(accuracy_matrix)
sd <- matrix(0,1,ncol(accuracy_matrix))
colnames(sd) <- colnames(accuracy_matrix)
range <- matrix(0,1,ncol(accuracy_matrix))
colnames(range) <- colnames(accuracy_matrix)

for (i in 1:(ncol(accuracy_matrix))) {
  mu[i] <- mean(output[,i])
  sd[i] <- sd(output[,i])
  range[i] <- max(output[,i]) - min(output[,i])
}
result <- rbind(mu,sd,range)
print(result)



