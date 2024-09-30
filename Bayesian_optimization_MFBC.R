library(FNN)
library(rBayesianOptimization)

MFBC <- function(X_train, Y_train, X_test, prior) {
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
    B[i, ] <- P[i, ] * prior / Z
  }
  
  predictions <- numeric(N_test)
  for (i in 1:N_test) {
    theta <- which.max(B[i, ])
    predictions[i] <- C[theta]
  }
  
  return(predictions)
}

# 定义目标函数
objective_function <- function(initial_prior) {
  # 将数据集划分为训练集和验证集
  set.seed(123)
  train_indices <- sample(1:nrow(X_train), size = 0.8 * nrow(X_train))
  X_train_subset <- X_train[train_indices, ]
  Y_train_subset <- Y_train[train_indices]
  X_valid <- X_train[-train_indices, ]
  Y_valid <- Y_train[-train_indices]
  
  # 使用当前phi值进行训练和预测
  predictions <- MFBC(X_train_subset, Y_train_subset, X_valid, initial_prior)
  
  # 计算准确率
  accuracy <- sum(predictions == Y_valid) / length(Y_valid)
  
  # 返回目标值（负的准确率，因为BayesianOptimization是最小化目标函数）
  return(list(Score = accuracy, Pred = 0))
}

# 获取类别数量
q <- length(levels(Y_train))

# 设置贝叶斯优化的范围
bounds <- list(
  for (i in 1:q) {
    bounds[paste0("phi_", i)] <- c(0.001, 1)
  }
)
for (i in 1:q) {
  bounds[[paste0("phi_", i)]] <- c(0.001, 5)
}

initial_prior <- numeric(q)
N <- numeric(q)
for (i in 1:q) { #频率先验
  N[i] <- nrow(Phi[[i]])
  initial_prior[i] <- N[i] / N_train
}
# 进行贝叶斯优化
bayes_opt_result <- BayesianOptimization(
  FUN = objective_function,
  bounds = bounds,
  init_points = 5,
  n_iter = 25,
  acq = "ei"
)

# 输出最优的phi值
best_phi_values <- bayes_opt_result$Best_Par
print(best_phi_values)

