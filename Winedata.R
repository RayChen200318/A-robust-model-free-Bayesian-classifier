library(caret)
library(adabag)
library(datasets)
library(class)
library(bnlearn)
library(e1071)
library(FNN)
library(MASS)

rm(list = ls())
wine <- read.csv("C:/Users/Ray Chen/Desktop/wine.data")
xx <- wine[,1]
wine <- wine[,-1]
wine <- cbind(wine, xx)
wine$xx <- as.factor(wine$xx)

set.seed(42)  
index <- sample(1:nrow(wine), size = 0.7 * nrow(wine))
X_train <- wine[index, 1:(ncol(wine)-1)]
X_test <- wine[-index, 1:(ncol(wine)-1)]
Y_train <- wine[index, (ncol(wine))]
Y_test <- wine[-index, (ncol(wine))]

# kNN
k <- 5
knn_pred <- knn(train = X_train, test = X_test, cl = Y_train, k = k)

confusion_matrix_kNN <- table(knn_pred, Y_test)
accuracy_knn <- sum(diag(confusion_matrix_kNN)) / sum(confusion_matrix_kNN)

print(confusion_matrix_kNN)
print(paste("Accuracy: ", round(accuracy_knn, 4)))

# NN
k <- 1
nn_pred <- knn(train = X_train, test = X_test, cl = Y_train, k = k)

confusion_matrix_NN <- table(nn_pred, Y_test)
accuracy_nn <- sum(diag(confusion_matrix_NN)) / sum(confusion_matrix_NN)

print(confusion_matrix_NN)
print(paste("Accuracy: ", round(accuracy_nn, 4)))

# lda
lda_model <- lda(Y_train ~ ., data = cbind(X_train, Y_train))

print(lda_model)

lda_pred <- predict(lda_model, cbind(X_test))

print(head(lda_pred$class))

confusion_matrix_lda <- table(lda_pred$class, Y_test)
print(confusion_matrix_lda)

accuracy_lda <- sum(diag(confusion_matrix_lda)) / sum(confusion_matrix_lda)
print(paste("Accuracy: ", round(accuracy_lda, 4)))

# NBC
nbc_model <- naiveBayes(Y_train ~ ., data = cbind(X_train, Y_train))

nbc_pred <- predict(nbc_model, cbind(X_test, Y_test))

confusion_matrix_nbc <- table(nbc_pred, Y_test)
print(confusion_matrix_nbc)

accuracy_nbc <- sum(diag(confusion_matrix_nbc)) / sum(confusion_matrix_nbc)
print(paste("Accuracy: ", round(accuracy_nbc, 4)))

# TAN
X_train_fac <- as.data.frame(matrix(0,nrow = nrow(X_train), ncol = ncol(X_train)))
for(i in 1:ncol(X_train)) {
  X_train_fac[,i] <- as.factor(X_train[,i])
}
X_test_fac <- as.data.frame(matrix(0,nrow = nrow(X_test), ncol = ncol(X_test)))
for(i in 1:ncol(X_test)) {
  X_test_fac[,i] <- as.factor(X_test[,i])
}
tan_structure <- tree.bayes(cbind(X_train_fac, Y_train), "Y_train")
print(tan_structure)
plot(tan_structure)

tan_test <- cbind(X_test_fac,Y_test)
colnames(tan_test)[ncol(tan_test)] <- "Y_train"
tan_pred <- predict(tan_structure, tan_test, prob=FALSE)
confusion_matrix_tan <- table(tan_pred, Y_test)
print(confusion_matrix_tan)
accuracy_tan <- sum(diag(confusion_matrix_tan)) / sum(confusion_matrix_tan)
print(paste("Accuracy: ", round(accuracy_tan, 4)))

# SVM
svm_model <- svm(Y_train ~ ., data = cbind(X_train, Y_train), kernel = "linear", cost = 1, scale = TRUE)

svm_pred <- predict(svm_model, cbind(X_test, Y_test))
confusion_matrix_svm <- table(Predicted = svm_pred, Actual = Y_test)
print(confusion_matrix_svm)

accuracy_svm <- sum(diag(confusion_matrix_svm)) / sum(confusion_matrix_svm)
print(paste("Accuracy:", round(accuracy_svm, 4)))

# MFBC
nn <- 1e6
gamma <- sum(1 / (1:nn)) - log(nn)
print(gamma)

C <- levels(Y_train)
r <- ncol(X_train)
N_train <- nrow(X_train)
q <- length(C)
K_gamma <- r * pi ^ (r / 2) / gamma(r / 2 + 1)

find_NN <- function() {
  
}



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
  #for (i in 1:q) { #频率先验
    #N[i] <- nrow(Phi[[i]])
    #phi[i] <- N[i] / N_train
 # }
 # for (i in 1:q) { # flat prior
   # phi[i] <- 1/q 
 # }
  for (i in 1:q) { # Laplace smooth prior
    N[i] <- nrow(Phi[[i]])
    phi[i] <- (N[i] + 1) / (N_train + q)
  }
  
  # 初始化P和B矩阵
  N_test <- nrow(X_test)
  P <- matrix(0, N_test, q)
  B <- matrix(0, N_test, q)
  
  # 计算P和B
  for (i in 1:N_test) {
    for (k in 1:q) {
      if (nrow(Phi[[k]]) > 0) {
        tau <- knnx.dist(Phi[[k]], X_test[i, , drop = FALSE], k = 1)
        P[i, k] <- 2^-(r * log2(tau + 1) + log2((K_gamma * (N_train - 1)) / r) + log2(gamma / log(2)))
      } else {
        P[i, k] <- 0
      }
    }
    
    Z <- sum(P[i, ])
    B[i, ] <- P[i, ] * phi /Z
  }
  
  predictions <- numeric(N_test)
  for (i in 1:N_test) {
    theta <- which.max(B[i, ])
    predictions[i] <- C[theta]
  }
  
  return(predictions)
}

mfbc_pred <- MFBC(X_train, Y_train, X_test)

confusion_matrix_mfbc <- table(mfbc_pred, Y_test)
print(confusion_matrix_mfbc)
accuracy_mfbc <- sum(diag(confusion_matrix_mfbc)) / sum(confusion_matrix_mfbc)
print(accuracy_mfbc)


# 10-fold validation
folds <- createFolds(wine$xx, k = 10, list = TRUE, returnTrain = TRUE)

# 打印每个折叠的训练集和测试集索引
for (i in 1:10) {
  cat("Fold", i, "\n")
  cat("Training indices:", folds[[i]], "\n")
  cat("Test indices:", setdiff(1:nrow(wine), folds[[i]]), "\n\n")
}

max_it <- 10
accuracy_lda <- numeric(max_it)
accuracy_mfbc <- numeric(max_it)
accuracy_nbc <- numeric(max_it)
accuracy_nn <- numeric(max_it)
accuracy_knn <- numeric(max_it)
accuracy_svm<- numeric(max_it)
accuracy_tan <- numeric(max_it)
m <- 1
for (m in 1:max_it) {
  set.seed(m ^ 6 +3 * m + 12)  
  index <- folds[[m]]
  X_train <- wine[index, 1:(ncol(wine)-1)]
  X_test <- wine[-index, 1:(ncol(wine)-1)]
  Y_train <- wine[index, (ncol(wine))]
  Y_test <- wine[-index, (ncol(wine))]
  
  # kNN
  k <- 5
  knn_pred <- knn(train = X_train, test = X_test, cl = Y_train, k = k)
  
  confusion_matrix_kNN <- table(knn_pred, Y_test)
  accuracy_knn[m] <- sum(diag(confusion_matrix_kNN)) / sum(confusion_matrix_kNN)
  
  # NN
  k <- 1
  nn_pred <- knn(train = X_train, test = X_test, cl = Y_train, k = k)
  
  confusion_matrix_NN <- table(nn_pred, Y_test)
  accuracy_nn[m] <- sum(diag(confusion_matrix_NN)) / sum(confusion_matrix_NN)
  
  # lda
  lda_model <- lda(Y_train ~ ., data = cbind(X_train, Y_train))
  
  lda_pred <- predict(lda_model, cbind(X_test, Y_test))
  
  confusion_matrix_lda <- table(lda_pred$class, Y_test)
  
  accuracy_lda[m] <- sum(diag(confusion_matrix_lda)) / sum(confusion_matrix_lda)
  
  # NBC
  nbc_model <- naiveBayes(Y_train ~ ., data = cbind(X_train, Y_train))
  
  nbc_pred <- predict(nbc_model, cbind(X_test, Y_test))
  
  confusion_matrix_nbc <- table(nbc_pred, Y_test)
  
  accuracy_nbc[m] <- sum(diag(confusion_matrix_nbc)) / sum(confusion_matrix_nbc)
  
  # TAN
  X_train_fac <- as.data.frame(matrix(0,nrow = nrow(X_train), ncol = ncol(X_train)))
  for(i in 1:ncol(X_train)) {
    X_train_fac[,i] <- as.factor(X_train[,i])
  }
  X_test_fac <- as.data.frame(matrix(0,nrow = nrow(X_test), ncol = ncol(X_test)))
  for(i in 1:ncol(X_test)) {
    X_test_fac[,i] <- as.factor(X_test[,i])
  }
  tan_structure <- tree.bayes(cbind(X_train_fac, Y_train), "Y_train")
  
  tan_test <- cbind(X_test_fac,Y_test)
  colnames(tan_test)[ncol(tan_test)] <- "Y_train"
  tan_pred <- predict(tan_structure, tan_test, prob=FALSE)
  confusion_matrix_tan <- table(tan_pred, Y_test)
  accuracy_tan[m] <- sum(diag(confusion_matrix_tan)) / sum(confusion_matrix_tan)
  
  # SVM
  svm_model <- svm(Y_train ~ ., data = cbind(X_train, Y_train), kernel = "linear", cost = 1, scale = TRUE)
  
  svm_pred <- predict(svm_model, cbind(X_test, Y_test))
  confusion_matrix_svm <- table(Predicted = svm_pred, Actual = Y_test)
  
  accuracy_svm[m] <- sum(diag(confusion_matrix_svm)) / sum(confusion_matrix_svm)
  
  # MFBC
  mfbc_pred <- MFBC(X_train, Y_train, X_test)
  
  confusion_matrix_mfbc <- table(mfbc_pred, Y_test)
  accuracy_mfbc[m] <- sum(diag(confusion_matrix_mfbc)) / sum(confusion_matrix_mfbc)
}

accuracy_matrix <- cbind(accuracy_knn,accuracy_lda,accuracy_nn,accuracy_svm,accuracy_tan,accuracy_nbc,accuracy_mfbc)
colMeans(accuracy_matrix)
