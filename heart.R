rm(list = ls())
library(caret)
library(readxl)

heart <- read.csv("C:/Users/Ray Chen/Desktop/heart.csv")
heart$output <- as.factor(heart$output)
# Training Set and Testing set
set.seed(42) 
index <- sample(1:nrow(heart), size = 0.7 * nrow(heart)) 
train_data <- heart[index, ] 
test_data <- heart[-index, ] 

# =====================================kNN=====================================
library(class)
# Training Set and Testing set
set.seed(42) 
index <- sample(1:nrow(heart), size = 0.7 * nrow(heart)) 
train_data <- heart[index, ] 
test_data <- heart[-index, ] 

train_labels <- train_data$output
test_labels <- test_data$output

# Training the kNN model
k <- 9
pred_knn <- knn(train = train_data, test = test_data, cl = train_labels, k = k)

# Evaluating the result
confusion_matrix_kNN <- table(pred_knn, test_labels)
accuracy_knn <- sum(diag(confusion_matrix_kNN)) / sum(confusion_matrix_kNN)

# Output
print(confusion_matrix_kNN)
print(paste("Accuracy: ", round(accuracy_knn, 4)))

# =====================================NN=====================================
library(class)

# Training the NN model
k <- 1
pred_nn <- knn(train = train_data, test = test_data, cl = train_labels, k = k)

# Evaluating the result
confusion_matrix_NN <- table(pred_knn, test_labels)
accuracy_nn <- sum(diag(confusion_matrix_NN)) / sum(confusion_matrix_NN)

# Output
print(confusion_matrix_NN)
print(paste("Accuracy: ", round(accuracy_nn, 4)))

# =============================LDA===============================
library(MASS)

lda_model <- lda(output ~ ., data = train_data)

print(lda_model)

pred_lda <- predict(lda_model, test_data)

print(head(pred_lda$class))

confusion_matrix_lda <- table(pred_lda$class, test_data$output)
print(confusion_matrix_lda)

accuracy_lda <- sum(diag(confusion_matrix_lda)) / sum(confusion_matrix_lda)
print(paste("Accuracy: ", round(accuracy_lda, 4)))

# ==============================Naive Bayesian Classifier====================================
library(e1071)
model <- naiveBayes(output ~ ., data = train_data)

# 预测
pred_nbc <- predict(model, test_data)

# 查看结果
confusion_matrix_nbc <- table(pred_nbc, test_data$output)
print(confusion_matrix_nbc)

# 计算准确率
accuracy_nbc <- sum(diag(confusion_matrix_nbc)) / sum(confusion_matrix_nbc)
print(paste("Accuracy: ", round(accuracy_nbc, 4)))
# ============================TAN classifier====================================(暂时没写出)
library(bnlearn)

for (i in 1:ncol(heart)) {
  heart[,i] <- as.factor(heart[,i])
}

index <- sample(1:nrow(heart), size = 0.7 * nrow(heart)) 
train_data <- heart[index, ] 
test_data <- heart[-index, ] 

tan_structure <- tree.bayes(train_data, "output")
print(tan_structure)
plot(tan_structure)

pred_tan <- predict(tan_structure, test_data, prob=FALSE)
confusion_matrix_tan <- table(pred_tan, test_data$output)
print(confusion_matrix_tan)
accuracy_tan <- sum(diag(confusion_matrix_tan)) / sum(confusion_matrix_tan)
print(paste("Accuracy: ", round(accuracy_tan, 4)))

# =====================================SVM=======================================
library(e1071)

svm_model <- svm(output ~ ., data = train_data, kernel = "linear", cost = 1, scale = TRUE)

pred_svm <- predict(svm_model, test_data)
confusion_matrix_svm <- table(Predicted = pred_svm, Actual = test_data$output)
print(confusion_matrix_svm)

accuracy_svm <- sum(diag(confusion_matrix_svm)) / sum(confusion_matrix_svm)
print(paste("Accuracy: ", round(accuracy_svm, 4)))

# =======================Model-Free Bayesian Classifier==========================

# 输入参数：
# X_train: 训练样本矩阵
# Y_train: 训练样本的标签向量
# X_test: 测试样本矩阵
# c: 类别标签的集合
# K_gamma: 常数参数
library(FNN)  # 用于最近邻居搜索

data("iris")
str(iris)

# Training Set and Testing set
set.seed(42)  
index <- sample(1:nrow(heart), size = 0.7 * nrow(heart))
X_train <- heart[index, 1:ncol(heart)-1]
X_test <- heart[-index, 1:ncol(heart)-1]
Y_train <- heart[index, ncol(heart)]
Y_test <- heart[-index, ncol(heart)]
C <- levels(Y_train)

r <- ncol(X_train)
N_train <- nrow(X_train)
q <- length(C)
K_gamma <- r * pi ^ (r / 2) / gamma(r / 2 + 1)


# 计算Euler-Mascheroni常数的近似值
n <- 1000000
gamma_approx <- sum(1 / (1:n)) - log(n)
gamma_approx

MFBC <- function(X_train, Y_train, X_test, C, K_gamma) {
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
  for (i in 1:q) {
    N[i] <- nrow(Phi[[i]])
    phi[i] <- N[i] / N_train
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
        P[i, k] <- 2^-(r * log2(tau + 1) + log2((K_gamma * (N_train - 1)) / r) + log2(gamma_approx / log(2)))
      } else {
        P[i, k] <- 0
      }
    }
     
    Z <- sum(P[i, ])
    if (Z != 0) {
      for (k in 1:q) {
        B[i, k] <- (P[i, k] * phi[k]) / Z
      }
    } else {
      B[i, ] <- 0
    }
  }
  
  # 预测测试样本的类别
  predictions <- numeric(N_test)
  for (i in 1:N_test) {
    theta <- which.max(B[i, ])
    predictions[i] <- C[theta]
  }
  
  return(predictions)
}

pred_MFBC <- MFBC(X_train, Y_train, X_test, C, K_gamma)

confusion_matrix_MFBC <- table(pred_MFBC, Y_test)
print(confusion_matrix_MFBC)
accuracy_MFBC <- sum(diag(confusion_matrix_MFBC)) / sum(confusion_matrix_MFBC)
print(paste("Accuracy: ", round(accuracy_MFBC, 4)))

# ===============================AdaBoost=======================================
library(adabag)
library(datasets)

data(iris)

set.seed(123)  
index <- sample(1:nrow(iris), 0.7 * nrow(iris))
train_data <- iris[index, ]
test_data <- iris[-index, ]

adaboost_model <- boosting(Species ~ ., data = train_data, boos = TRUE, mfinal = 50)
predictions <- predict(adaboost_model, newdata = test_data)
print(predictions$confusion)
print(paste("Accuracy: ", sum(diag(predictions$confusion)) / sum(predictions$confusion)))

