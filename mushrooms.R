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
library(rpart)
library(glmnet)
library(dbscan)
library(ggdendro)
library(ggplot2)

rm(list = ls())

total_steps <- 10
pb <- progress_bar$new(
  format = "  处理中 [:bar] :percent :elapsed 剩余时间: :eta",
  total = total_steps, clear = FALSE, width = 60
)

mushroom <- read.csv2("C:/Users/Ray Chen/Desktop/data/mushrooms.csv", sep = ",")

mushroom <- subset(mushroom, select = -veil.type)

mushroom[mushroom == "?"] <- NA


mushroom_coded <- mushroom
for (col in names(mushroom_coded)) {
  mushroom_coded[[col]] <- as.numeric(as.factor(mushroom_coded[[col]]))
}

levels_list <- list()
for (col in names(mushroom_coded)) {
  mushroom_coded[[col]] <- as.numeric(as.factor(mushroom_coded[[col]]))
  levels_list[[col]] <- levels(as.factor(mushroom[[col]]))
}

class <- mushroom_coded[,1]
mushroom_coded <- mushroom_coded[, -1]
mushroom_coded <- cbind(mushroom_coded, class)

impute_knn <- function(data, k = 5) {
  for (col in 1:ncol(data)) {
    missing_index <- which(is.na(data[, col]))
    if (length(missing_index) > 0) {
      complete_data <- data[!is.na(data[, col]), ]
      incomplete_data <- data[is.na(data[, col]), ]
      
      knn_data <- knn.reg(train = complete_data[, -col], test = incomplete_data[, -col], y = complete_data[, col], k = k)$pred
      data[missing_index, col] <- knn_data
    }
  }
  return(data)
}

# 使用kNN填补缺失值
mushroom_imputed_coded <- impute_knn(mushroom_coded)

mushroom_imputed_coded[[ncol(mushroom)]] <- as.factor(mushroom_imputed_coded[[ncol(mushroom)]])

# 将数值数据转换回因子
mushroom_imputed <- mushroom_imputed_coded
for (col in names(mushroom_imputed)) {
  levels <- levels_list[[col]]
  mushroom_imputed[[col]] <- factor(levels[mushroom_imputed_coded[[col]]], levels = levels)
}

# 查看填补后的数据
print(mushroom_imputed)

setwd("C:/Users/Ray Chen/Desktop/data") # 请替换为你希望保存CSV文件的路径

# 将数据框写入CSV文件
write.csv(mushroom_imputed, 
          file = "imputed_mushrooms.csv",    # 替换为你希望保存的文件名
          row.names = TRUE)           # 是否写入行名，默认是TRUE

library(plotly)

pca_result <- prcomp(mushroom_dummy[, 1:(ncol(mushroom_dummy)-1)], center = FALSE, scale. = FALSE)
summary(pca_result)
# 提取主成分
pca_data <- data.frame(pca_result$x)
pca_data$class <- mushroom_dummy$class

var_explained <- pca_result$sdev^2 / sum(pca_result$sdev^2)
cum_var_explained <- cumsum(var_explained)

# 创建数据框用于绘图
scree_data <- data.frame(
  Principal_Component = 1:length(var_explained),
  Variance_Explained = var_explained,
  Cumulative_Proportion = cum_var_explained
)


# 绘制Scree Plot和Cumulative Proportion Plot
scree_plot <- ggplot(scree_data, aes(x = Principal_Component)) +
  geom_line(aes(y = Variance_Explained), color = "blue", size = 0.5) +
  geom_point(aes(y = Variance_Explained), color = "blue", size = 2) +
  geom_line(aes(y = Cumulative_Proportion), color = "red", linetype = "dashed", size = 0.5) +
  geom_point(aes(y = Cumulative_Proportion), color = "red", size = 2) +
  geom_text(data = subset(scree_data, Principal_Component == 12), 
            aes(y = Variance_Explained, label = round(Variance_Explained, 3)), 
            vjust = -1, size = 3, color = "blue") +  # 仅添加第12个点的方差比例标签
  geom_text(data = subset(scree_data, Principal_Component == 12), 
            aes(y = Cumulative_Proportion, label = round(Cumulative_Proportion, 3)), 
            vjust = 1.5, size = 3, color = "red") +  # 仅添加第12个点的累计比例标签
  labs(title = "Scree Plot with Cumulative Proportion",
       x = "Principal Component",
       y = "Proportion of Variance Explained") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 15),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  ) +
  scale_y_continuous(sec.axis = sec_axis(~., name = "Cumulative Proportion"))

print(scree_plot)

pca_3d_data <- pca_data[, 1:3]
names(pca_3d_data) <- c("PC1", "PC2", "PC3")
pca_3d_data$class <- pca_data$class

# 绘制3D Individual Plot
p_3d <- plot_ly(data = pca_3d_data, 
                x = ~PC1, 
                y = ~PC2, 
                z = ~PC3, 
                color = ~class, 
                colors = c('blue', 'red'), 
                type = 'scatter3d', 
                mode = 'markers',
                marker = list(size = 2)) %>%  # 设置点的大小
  layout(title = '3D Individual Plot by Class',
         scene = list(xaxis = list(title = 'PC1'),
                      yaxis = list(title = 'PC2'),
                      zaxis = list(title = 'PC3')))



# 显示3D Plot
p_3d

contributions <- abs(pca_result$rotation[, 1]) * 100

# 创建数据框用于绘图
contrib_data <- data.frame(
  Variable = names(contributions),
  Contribution = contributions
)

# 选择前10个贡献最大的变量
top_contrib_data <- contrib_data %>% 
  arrange(desc(Contribution)) %>% 
  head(20)

# 绘制Contribution of Variables to Dim-1 Plot，仅显示前20个变量
contrib_plot <- ggplot(top_contrib_data, aes(x = reorder(Variable, Contribution), y = Contribution)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  geom_text(aes(label = round(Contribution, 2)), hjust = -0.1, size = 3) +  # 添加贡献度标签
  labs(title = "Top 20 Contributions of Variables to Dim-1",
       x = "Variables",
       y = "Contribution (%)") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 15),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  ) +
  coord_flip()  # 将柱状图翻转，使变量名显示在Y轴上

# 显示Contribution Plot
print(contrib_plot)

contributions <- abs(pca_result$rotation[, 2]) * 100

# 创建数据框用于绘图
contrib_data <- data.frame(
  Variable = names(contributions),
  Contribution = contributions
)

# 选择前20个贡献最大的变量
top_contrib_data <- contrib_data %>% 
  arrange(desc(Contribution)) %>% 
  head(20)

# 绘制Contribution of Variables to Dim-2 Plot，仅显示前20个变量
contrib_plot <- ggplot(top_contrib_data, aes(x = reorder(Variable, Contribution), y = Contribution)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  geom_text(aes(label = round(Contribution, 2)), hjust = -0.1, size = 3) +  # 添加贡献度标签
  labs(title = "Top 20 Contributions of Variables to Dim-2",
       x = "Variables",
       y = "Contribution (%)") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 15),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  ) +
  coord_flip()  # 将柱状图翻转，使变量名显示在Y轴上

# 显示Contribution Plot
print(contrib_plot)



library(dplyr)
library(tidyr)
library(plyr)

calculate_prior <- function(data) {
  prior <- table(data) / length(data)
  return(prior)
}

# 定义计算条件概率的函数
calculate_likelihood <- function(data, feature, target) {
  likelihood <- prop.table(table(data, feature), margin = 1)
  return(likelihood)
}

bayesian_impute <- function(data) {
  complete_data <- data
  for (col in names(data)) {
    missing_index <- which(is.na(data[[col]]))
    if (length(missing_index) > 0) {
      # 计算先验概率
      prior <- calculate_prior(data[[col]])
      
      # 计算条件概率
      for (i in missing_index) {
        likelihood <- sapply(names(data), function(f) {
          if (f != col) {
            return(calculate_likelihood(data[[col]], data[[f]], data[[col]]))
          } else {
            return(NULL)
          }
        })
        
        # 计算后验概率
        posterior <- prior
        for (f in names(likelihood)) {
          if (!is.null(likelihood[[f]])) {
            posterior <- posterior * likelihood[[f]][, data[i, f]]
          }
        }
        
        # 选择后验概率最大的类别
        imputed_value <- names(which.max(posterior))
        complete_data[i, col] <- imputed_value
      }
    }
  }
  return(complete_data)
}

mushroom_bayesian_imputed <- bayesian_impute(mushroom)

# 分离最后一列（目标变量）
mushroom_features <- mushroom_imputed[, -ncol(mushroom_imputed)]
mushroom_class <- mushroom_imputed[, ncol(mushroom_imputed)]

# 使用dummyVars函数将分类数据转换为哑变量
dummy_model <- dummyVars(~ ., data = mushroom_features, fullRank = TRUE)
mushroom_features_dummies <- predict(dummy_model, newdata = mushroom_features)

# 合并哑变量和目标变量
mushroom_dummy <- data.frame(mushroom_features_dummies, class = mushroom_class)

# 查看转换后的数据
print(mushroom_dummy)


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
set.seed(1331)
folds <- createFolds(mushroom_dummy$class, k = max_it, list = TRUE, returnTrain = TRUE)

accuracy_mfbc <- numeric(max_it)
accuracy_nbc <- numeric(max_it)
accuracy_nn <- numeric(max_it)
accuracy_lasso <- numeric(max_it)
accuracy_knn <- numeric(max_it)
accuracy_svm <- numeric(max_it)
accuracy_probit <- numeric(max_it)

m <- 1
for (m in 1:max_it) {
  pb$tick()
  index <- folds[[m]]
  X_train <- mushroom_dummy[index, 1:(ncol(mushroom_dummy)-1)]
  X_test <- mushroom_dummy[-index, 1:(ncol(mushroom_dummy)-1)]
  Y_train <- mushroom_dummy[index, (ncol(mushroom_dummy))]
  Y_test <- mushroom_dummy[-index, (ncol(mushroom_dummy))]
  
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
  
  # NBC
  nbc_model <- naiveBayes(Y_train ~ ., data = cbind(X_train, Y_train))
  
  nbc_pred <- predict(nbc_model, cbind(X_test, Y_test))
  
  confusion_matrix_nbc <- table(nbc_pred, Y_test)
  
  accuracy_nbc[m] <- sum(diag(confusion_matrix_nbc)) / sum(confusion_matrix_nbc)
  
  
  # SVM
  svm_model <- svm(Y_train ~ ., data = cbind(X_train, Y_train), kernel = "linear", cost = 1, scale = FALSE)
  
  svm_pred <- predict(svm_model, cbind(X_test, Y_test))
  confusion_matrix_svm <- table(Predicted = svm_pred, Actual = Y_test)
  
  accuracy_svm[m] <- sum(diag(confusion_matrix_svm)) / sum(confusion_matrix_svm)
  
   #LDA
  #lda_model <- lda(Y_train ~ ., data = cbind(X_train, Y_train))
  
  #lda_pred <- predict(lda_model, newdata = cbind(X_test, Y_test))
  #confusion_matrix_lda <- table(Predicted = lda_pred$class, Actual = Y_test)
  
  #accuracy_lda[m] <- sum(diag(confusion_matrix_lda)) / sum(confusion_matrix_lda)
  
  # MFBC
  mfbc_pred <- MFBC(X_train, Y_train, X_test)
  
  confusion_matrix_mfbc <- table(mfbc_pred, Y_test)
  accuracy_mfbc[m] <- sum(diag(confusion_matrix_mfbc)) / sum(confusion_matrix_mfbc)
  
  # Probit regression
  probit_model <- glm(Y_train ~ ., data = cbind(X_train, Y_train), family = binomial(link = "probit"))
  probit_pred <- predict(probit_model, newdata = X_test, type = "response")
  probit_class <- ifelse(probit_pred > 0.5, 1, 0)
  probit_class <- factor(probit_class, levels = c(0, 1))
  confusion_matrix_probit <- table(Predicted = probit_class, Actual = Y_test)
  accuracy_probit[m] <- sum(diag(confusion_matrix_probit)) / sum(confusion_matrix_probit)
  
  # LASSO regression
  Y_train_lasso <- as.numeric(Y_train) - 1
  Y_test_lasso <- as.numeric(Y_test) - 1
  X_train_lasso <- as.matrix(X_train)
  X_test_lasso <- as.matrix(X_test)
  lasso_model <- cv.glmnet(X_train_lasso, Y_train_lasso, family = "binomial", alpha = 1)
  lasso_pred <- predict(lasso_model, s = lasso_model$lambda.min, newx = X_test_lasso, type = "response")
  lasso_class <- ifelse(lasso_pred > 0.5, 1, 0)
  lasso_class <- factor(lasso_class, levels = c(0, 1))
  confusion_matrix_lasso <- table(Predicted = lasso_class, Actual = Y_test)
  accuracy_lasso[m] <- sum(diag(confusion_matrix_lasso)) / sum(confusion_matrix_lasso)

}

# k-Means
X <-  mushroom_dummy[,1:(ncol(mushroom_dummy)-1)]
Y <-  as.factor(mushroom_dummy[,ncol(mushroom_dummy)])
kmeans_result <- kmeans(X, centers = 2, nstart = 25)
cluster_assignments <- kmeans_result$cluster
confusion_matrix_kmeans <- table(Predicted = cluster_assignments, Actual = Y)
print(confusion_matrix_kmeans)
accuracy_kmeans <- sum(diag(confusion_matrix_kmeans)) / sum(confusion_matrix_kmeans)
accuracy_kmeans
# hierarchical clustering
dist_matrix <- dist(X)
hclust_result <- hclust(dist_matrix, method = "ward.D2")
cluster_assignments  <- cutree(hclust_result, k = 2)
confusion_matrix_hclust <- table(Predicted = cluster_assignments, Actual = Y)
print(confusion_matrix_hclust)
accuracy_hclust <- sum(diag(confusion_matrix_hclust)) / sum(confusion_matrix_hclust)
accuracy_hclust
dendro_data <- dendro_data(hclust_result)

pca_features <- as.data.frame(pca_result$x[,1:12])

pca_features$class <- mushroom_dummy$class

m <- 1
for (m in 1:max_it) {
  pb$tick()
  index <- folds[[m]]
  X_train <- pca_features[index, 1:(ncol(pca_features)-1)]
  X_test <- pca_features[-index, 1:(ncol(pca_features)-1)]
  Y_train <- pca_features[index, (ncol(pca_features))]
  Y_test <- pca_features[-index, (ncol(pca_features))]
  
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
  
  # NBC
  nbc_model <- naiveBayes(Y_train ~ ., data = cbind(X_train, Y_train))
  
  nbc_pred <- predict(nbc_model, cbind(X_test, Y_test))
  
  confusion_matrix_nbc <- table(nbc_pred, Y_test)
  
  accuracy_nbc[m] <- sum(diag(confusion_matrix_nbc)) / sum(confusion_matrix_nbc)
  
  
  # SVM
  svm_model <- svm(Y_train ~ ., data = cbind(X_train, Y_train), kernel = "linear", cost = 1, scale = FALSE)
  
  svm_pred <- predict(svm_model, cbind(X_test, Y_test))
  confusion_matrix_svm <- table(Predicted = svm_pred, Actual = Y_test)
  
  accuracy_svm[m] <- sum(diag(confusion_matrix_svm)) / sum(confusion_matrix_svm)
  
  #LDA
  #lda_model <- lda(Y_train ~ ., data = cbind(X_train, Y_train))
  
  #lda_pred <- predict(lda_model, newdata = cbind(X_test, Y_test))
  #confusion_matrix_lda <- table(Predicted = lda_pred$class, Actual = Y_test)
  
  #accuracy_lda[m] <- sum(diag(confusion_matrix_lda)) / sum(confusion_matrix_lda)
  
  # MFBC
  mfbc_pred <- MFBC(X_train, Y_train, X_test)
  
  confusion_matrix_mfbc <- table(mfbc_pred, Y_test)
  accuracy_mfbc[m] <- sum(diag(confusion_matrix_mfbc)) / sum(confusion_matrix_mfbc)
  
  # Probit regression
  probit_model <- glm(Y_train ~ ., data = cbind(X_train, Y_train), family = binomial(link = "probit"))
  probit_pred <- predict(probit_model, newdata = X_test, type = "response")
  probit_class <- ifelse(probit_pred > 0.5, 1, 0)
  probit_class <- factor(probit_class, levels = c(0, 1))
  confusion_matrix_probit <- table(Predicted = probit_class, Actual = Y_test)
  accuracy_probit[m] <- sum(diag(confusion_matrix_probit)) / sum(confusion_matrix_probit)
  
  # LASSO regression
  Y_train_lasso <- as.numeric(Y_train) - 1
  Y_test_lasso <- as.numeric(Y_test) - 1
  X_train_lasso <- as.matrix(X_train)
  X_test_lasso <- as.matrix(X_test)
  lasso_model <- cv.glmnet(X_train_lasso, Y_train_lasso, family = "binomial", alpha = 1)
  lasso_pred <- predict(lasso_model, s = lasso_model$lambda.min, newx = X_test_lasso, type = "response")
  lasso_class <- ifelse(lasso_pred > 0.5, 1, 0)
  lasso_class <- factor(lasso_class, levels = c(0, 1))
  confusion_matrix_lasso <- table(Predicted = lasso_class, Actual = Y_test)
  accuracy_lasso[m] <- sum(diag(confusion_matrix_lasso)) / sum(confusion_matrix_lasso)
  
}

accuracy_matrix <- cbind(accuracy_knn,accuracy_nn,accuracy_svm,accuracy_nbc,accuracy_mfbc,accuracy_lasso,accuracy_probit)
accuracy_matrix

output <- matrix(0,10,ncol(accuracy_matrix))
colnames(output) <- colnames(accuracy_matrix)
for (i in 1:(ncol(accuracy_matrix))) {
  output[,i] <- sort(accuracy_matrix[, i],decreasing = TRUE)[1:10]
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



# k-Means
X <-  pca_features[,1:(ncol(pca_features)-1)]
Y <-  as.factor(pca_features[,ncol(pca_features)])
kmeans_result <- kmeans(X, centers = 2, nstart = 25)
cluster_assignments <- kmeans_result$cluster
confusion_matrix_kmeans <- table(Predicted = cluster_assignments, Actual = Y)
print(confusion_matrix_kmeans)
accuracy_kmeans <- sum(diag(confusion_matrix_kmeans)) / sum(confusion_matrix_kmeans)
accuracy_kmeans
# hierarchical clustering
dist_matrix <- dist(X)
hclust_result <- hclust(dist_matrix, method = "ward.D2")
cluster_assignments  <- cutree(hclust_result, k = 2)
confusion_matrix_hclust <- table(Predicted = cluster_assignments, Actual = Y)
print(confusion_matrix_hclust)
accuracy_hclust <- sum(diag(confusion_matrix_hclust)) / sum(confusion_matrix_hclust)
accuracy_hclust
dendro_data <- dendro_data(hclust_result)

ggdendrogram(dendro_data, lables = FALSE,rotate = FALSE, size = 1) +
  labs(title = "Hierarchical Clustering Dendrogram", x = "Sample index", y = "Height") +
  theme_minimal(base_size = 15) +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

