library(latex2exp)
library(ggplot2)
library(dplyr)
library(ggpubr)
library(scales)

rowSds = function(D) {
  apply(D, 1, sd)
}
colSds = function(D) {
  apply(D, 2, sd)
}


# ----------------------------------
# simulation
# ----------------------------------

# ----------------------------------
# Section 5.1.1: different h
h_value = seq(0, 0.8, 0.1)

plot_list = list()
for (setting_no in 1:4) {
  table_mean = sapply(1:length(h_value), function(i){
    s = sapply(0:99, function(j){
      D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_h/setting_', setting_no, '_',
                          j,'.csv'), header = FALSE)
      D[, i]
    })
    rowMeans(s)
  })
  
  table_sd = sapply(1:length(h_value), function(i){
    s = sapply(0:99, function(j){
      D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_h/setting_', setting_no, '_',
                          j,'.csv'), header = FALSE)
      D[, i]
    })
    rowSds(s)
  })
  
  rownames(table_mean) = c("Single-task", "Pooled", "ERM", "MoM", "ARMUL", "AdaptRep", "pERM", "Spectral", "GLasso")
  rownames(table_sd) = c("Single-task", "Pooled", "ERM", "MoM", "ARMUL", "AdaptRep", "pERM", "Spectral", "GLasso")
  
  table_all = data.frame(mean = as.vector(table_mean), 
                         sd = as.vector(table_sd), 
                         h = rep(h_value, each = nrow(table_mean)), 
                         method = rep(factor(rownames(table_mean)), length(h_value)))
  
  if (setting_no == 1) {
    plot_list[[setting_no]] = table_all %>% ggplot(mapping = aes(x = h, y = mean, group = method, color = method, shape = method)) + 
      geom_line(size = 0.5) + geom_point(size = 2) + scale_shape_manual(values = 0:8) + ylab(TeX(r"(max error in $S$)")) +
      ggtitle(TeX(r"($n = 100, p = 30, r = 5, T = 50, \epsilon = 0$)")) + theme(plot.title = element_text(hjust = 0.5))
  } else if (setting_no == 2){
    plot_list[[setting_no]] = table_all %>% ggplot(mapping = aes(x = h, y = mean, group = method, color = method, shape = method)) + 
      geom_line(size = 0.5) + geom_point(size = 2) + scale_shape_manual(values = 0:8)  + ylab(TeX(r"(max error in $S$)")) +
      ggtitle(TeX(r"($n = 100, p = 50, r = 5, T = 50, \epsilon = 0$)")) + theme(plot.title = element_text(hjust = 0.5))
  } else if (setting_no == 3){
    plot_list[[setting_no]] = table_all %>% ggplot(mapping = aes(x = h, y = mean, group = method, color = method, shape = method)) + 
      geom_line(size = 0.5) + geom_point(size = 2) + scale_shape_manual(values = 0:8) + ylab(TeX(r"(max error in $S$)")) +
      ggtitle(TeX(r"($n = 100, p = 80, r = 5, T = 50, \epsilon = 0$)")) + theme(plot.title = element_text(hjust = 0.5))
  } else if (setting_no == 4){
    plot_list[[setting_no]] = table_all %>% ggplot(mapping = aes(x = h, y = mean, group = method, color = method, shape = method)) + 
      geom_line(size = 0.5) + geom_point(size = 2) + scale_shape_manual(values = 0:8) + ylab(TeX(r"(max error in $S$)")) +
      ggtitle(TeX(r"($n = 150, p = 80, r = 10, T = 50, \epsilon = 0$)")) + theme(plot.title = element_text(hjust = 0.5))
  }
  
}

# 10 x 8 PDF -> Figure 5
ggarrange(plotlist = plot_list, nrow = 2, ncol = 2, common.legend = TRUE, legend = "bottom")


# ----------------------------------
# Section 5.1.2: different epsilon
epsilon_value = seq(0, 0.1, 0.02)

plot_list = list()
for (setting_no in 1:2) {
  # S
  table_mean_S = sapply(1:length(epsilon_value), function(i){
    s = sapply(0:99, function(j){
      D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_epsilon/setting_', setting_no, '_',
                          j,'.csv'), header = FALSE)
      D[1:9, i]
    })
    rowMeans(s)
  })
  
  table_sd_S = sapply(1:length(epsilon_value), function(i){
    s = sapply(0:99, function(j){
      D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_epsilon/setting_', setting_no, '_',
                          j,'.csv'), header = FALSE)
      D[1:9, i]
    })
    rowSds(s)
  })
  
  
  rownames(table_mean_S) = c("Single-task", "Pooled", "ERM", "MoM", "ARMUL", "AdaptRep", "pERM", "Spectral", "GLasso")
  rownames(table_sd_S) = c("Single-task", "Pooled", "ERM", "MoM", "ARMUL", "AdaptRep", "pERM", "Spectral", "GLasso")
  
  table_all_S = data.frame(mean = as.vector(table_mean_S), 
                         sd = as.vector(table_sd_S), 
                         epsilon = rep(epsilon_value, each = nrow(table_mean_S)), 
                         method = rep(factor(rownames(table_mean_S)), length(epsilon_value)))
  
  # Sc
  table_mean_Sc = sapply(2:length(epsilon_value), function(i){
    s = sapply(0:99, function(j){
      D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_epsilon/setting_', setting_no, '_',
                          j,'.csv'), header = FALSE)
      D[10:18, i]
    })
    rowMeans(s)
  })
  
  table_sd_Sc = sapply(2:length(epsilon_value), function(i){
    s = sapply(0:99, function(j){
      D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_epsilon/setting_', setting_no, '_',
                          j,'.csv'), header = FALSE)
      D[10:18, i]
    })
    rowSds(s)
  })
  
  rownames(table_mean_Sc) = c("Single-task", "Pooled", "ERM", "MoM", "ARMUL", "AdaptRep", "pERM", "Spectral", "GLasso")
  rownames(table_sd_Sc) = c("Single-task", "Pooled", "ERM", "MoM", "ARMUL", "AdaptRep", "pERM", "Spectral", "GLasso")
  
  table_all_Sc = data.frame(mean = as.vector(table_mean_Sc), 
                           sd = as.vector(table_sd_Sc), 
                           epsilon = rep(epsilon_value[-1], each = nrow(table_mean_Sc)), 
                           method = rep(factor(rownames(table_mean_Sc)), length(epsilon_value[-1])))
  
  if (setting_no == 1) {
    plot_list[[1]] = table_all_S %>% ggplot(mapping = aes(x = epsilon, y = mean, group = method, color = method, shape = method)) + 
      geom_line(size = 0.5) + geom_point(size = 2) + scale_shape_manual(values = 0:8) + ylab(TeX(r"(max error in $S$)")) +
      theme(plot.title = element_text(hjust = 0.5)) + xlab(TeX(r"(\epsilon)")) + 
      scale_x_continuous(labels = percent, breaks = unique(epsilon_value))
    
    plot_list[[2]] = table_all_Sc %>% ggplot(mapping = aes(x = epsilon, y = mean, group = method, color = method, shape = method)) + 
      geom_line(size = 0.5) + geom_point(size = 2) + scale_shape_manual(values = 0:8) + ylab(TeX(r"(max error in $S^c$)")) +
      theme(plot.title = element_text(hjust = 0.5)) + xlab(TeX(r"(\epsilon)")) + 
      scale_x_continuous(labels = percent, breaks = unique(epsilon_value[-1]))
  } else if (setting_no == 2){
    plot_list[[3]] = table_all_S %>% ggplot(mapping = aes(x = epsilon, y = mean, group = method, color = method, shape = method)) + 
      geom_line(size = 0.5) + geom_point(size = 2) + scale_shape_manual(values = 0:8) + ylab(TeX(r"(max error in $S$)")) +
      theme(plot.title = element_text(hjust = 0.5)) + xlab(TeX(r"(\epsilon)")) + 
      scale_x_continuous(labels = percent, breaks = unique(epsilon_value))
    
    plot_list[[4]] = table_all_Sc %>% ggplot(mapping = aes(x = epsilon, y = mean, group = method, color = method, shape = method)) + 
      geom_line(size = 0.5) + geom_point(size = 2) + scale_shape_manual(values = 0:8) + ylab(TeX(r"(max error in $S^c$)")) +
      theme(plot.title = element_text(hjust = 0.5)) + xlab(TeX(r"(\epsilon)")) +
      scale_x_continuous(labels = percent, breaks = unique(epsilon_value[-1]))
  } 
  
}


ann1 <- ggplot() + ggtitle(TeX(r"($n = 100, p = 50, r = 5, T = 100, h = 0$)")) + theme(plot.title = element_text(hjust = 1.8, vjust = -0.6)) + 
  coord_cartesian(clip = 'off') 

ann2 <- ggplot() + 
  geom_text(aes(x=0, y=0, label = ""), 
            parse = TRUE, size = 4, hjust = -1) +
  theme_void()


ann3 <- ggplot() + ggtitle(TeX(r"($n = 150, p = 80, r = 10, T = 100, h = 0$)")) + theme(plot.title = element_text(hjust = 1.8, vjust = -0.6)) + 
  coord_cartesian(clip = 'off') 

ann4 <- ggplot() + 
  geom_text(aes(x=0, y=0, label = ""), 
            parse = TRUE, size = 4, hjust = -1) +
  theme_void()

# 10 x 8 PDF -> Figure 6
ggarrange(ann1, ann2, plot_list[[1]], plot_list[[2]], ann3, ann4, plot_list[[3]], plot_list[[4]], nrow = 4, ncol = 2, legend = "bottom", common.legend = TRUE,
          heights = c(0.03, 0.3, 0.03, 0.3)) 


# ----------------------------------
# Section 5.1.3: different T
T_value = seq(10, 200, 15)

plot_list = list()
for (setting_no in 1:2) {
  table_mean = sapply(1:length(T_value), function(i){
    s = sapply(0:99, function(j){
      D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_T/setting_', setting_no, '_',
                          j,'.csv'), header = FALSE)
      D[1:9, i]
    })
    rowMeans(s)
  })
  
  table_sd = sapply(1:length(T_value), function(i){
    s = sapply(0:99, function(j){
      D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_T/setting_', setting_no, '_',
                          j,'.csv'), header = FALSE)
      D[1:9, i]
    })
    rowSds(s)
  })
  
  
  rownames(table_mean) = c("Single-task", "Pooled", "ERM", "MoM", "ARMUL", "AdaptRep", "pERM", "Spectral", "GLasso")
  rownames(table_sd) = c("Single-task", "Pooled", "ERM", "MoM", "ARMUL", "AdaptRep", "pERM", "Spectral", "GLasso")
  
  table_all = data.frame(mean = as.vector(table_mean), 
                         sd = as.vector(table_sd), 
                         T = rep(T_value, each = nrow(table_mean)), 
                         method = rep(factor(rownames(table_mean)), length(T_value)))
  
  if (setting_no == 1) {
    plot_list[[setting_no]] = table_all %>% ggplot(mapping = aes(x = T, y = mean, group = method, color = method, shape = method)) + 
      geom_line(size = 0.5) + geom_point(size = 2) + scale_shape_manual(values = 0:8) + ylab(TeX(r"(max error in $S$)")) +
      ggtitle(TeX(r"($n = 100, p = 50, r = 5, h = 0, \epsilon = 0$)")) + theme(plot.title = element_text(hjust = 0.5))
  } else if (setting_no == 2){
    plot_list[[setting_no]] = table_all %>% ggplot(mapping = aes(x = T, y = mean, group = method, color = method, shape = method)) + 
      geom_line(size = 0.5) + geom_point(size = 2) + scale_shape_manual(values = 0:8)  + ylab(TeX(r"(max error in $S$)")) +
      ggtitle(TeX(r"($n = 100, p = 50, r = 5, h = 0, \epsilon = 4\%$)")) + theme(plot.title = element_text(hjust = 0.5))
  } 
  
}

# 10 x 4.5 PDF -> Figure 7
ggarrange(plotlist = plot_list, nrow = 1, ncol = 2, common.legend = TRUE, legend = "bottom")


# ----------------------------------
# Section 5.1.3: computational time
T_value = seq(10, 200, 15)

table_mean = sapply(1:length(T_value), function(i){
  s = sapply(0:99, function(j){
    D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_T_time/',
                        j,'.csv'), header = FALSE)
    t(D)[1:9, i]
  })
  rowMeans(s)
})

table_sd = sapply(1:length(T_value), function(i){
  s = sapply(0:99, function(j){
    D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_T_time/',
                        j,'.csv'), header = FALSE)
    t(D)[1:9, i]
  })
  rowSds(s)
})


rownames(table_mean) = c("ARMUL", "AdaptRep", "Single-task", "Pooled", "MoM", "pERM", "ERM", "Spectral", "GLasso")
rownames(table_sd) = c("ARMUL", "AdaptRep", "Single-task", "Pooled", "MoM", "pERM", "ERM", "Spectral", "GLasso")

table_all = data.frame(mean = as.vector(table_mean), 
                       sd = as.vector(table_sd), 
                       T = rep(T_value, each = nrow(table_mean)), 
                       method = rep(factor(rownames(table_mean)), length(T_value)))

# 10*5 PDF -> Figure 8
table_all %>% ggplot(mapping = aes(x = T, y = log(mean), group = method, color = method, shape = method)) + 
    geom_line(size = 0.5) + geom_point(size = 2) + scale_shape_manual(values = 0:8) + ylab("log computational time (in seconds)") +
    ggtitle(TeX(r"($n = 100, p = 50, r = 5, h = 0, \epsilon = 0$)")) + theme(plot.title = element_text(hjust = 0.5),
                                                                             legend.position = "bottom")


# ----------------------------------
# Section 5.1.4: different signal strength
theta_norm = seq(0.5, 5, 0.5)

table_mean_S = sapply(1:length(theta_norm), function(i){
  s = sapply(0:99, function(j){
    D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_theta/',
                        j,'.csv'), header = FALSE)
    D[c(1, 3, 7, 8), i]
  })
  rowMeans(s)
})

table_sd_S = sapply(1:length(theta_norm), function(i){
  s = sapply(0:99, function(j){
    D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_theta/',
                        j,'.csv'), header = FALSE)
    D[c(1, 3, 7, 8), i]
  })
  rowSds(s)
})

table_all = data.frame(error = as.vector(table_mean_S), 
                       method = rep(c("Single-task", "ERM", "pERM", "Spectral"), 10),
                       theta_norm = rep(theta_norm, each = 4))


# 10 x 5 PDF -> Figure 9
table_all %>% ggplot(mapping = aes(x = theta_norm, y = error, group = method, color = method, shape = method)) + 
  geom_line(size = 0.5) + geom_point(size = 2) + scale_shape_manual(values = c(2,5,7, 8)) + ylab(TeX(r"(error of each task)")) +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "bottom") + xlab(TeX(r"($\|\theta^{(t)*}\|_2$)")) + 
  scale_color_manual(values = hue_pal()(9)[c(3,6,8, 9)]) + scale_x_continuous(breaks = theta_norm)

# correlation
apply(table_mean_S, 1, function(x){cor(x, theta_norm)})


# ----------------------------------
# Section 5.1.5: adaptivity to unknown r
h_value = seq(0, 0.8, 0.1)

plot_list = list()
for (setting_no in 1:4) {
  table_mean = sapply(1:length(h_value), function(i){
    s = sapply(0:99, function(j){
      D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_r_adaptive/setting_', setting_no, '_',
                          j,'.csv'), header = FALSE)
      D[, i]
    })
    rowMeans(s)
  })
  
  table_sd = sapply(1:length(h_value), function(i){
    s = sapply(0:99, function(j){
      D = read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/simulation/sim_r_adaptive/setting_', setting_no, '_',
                          j,'.csv'), header = FALSE)
      D[, i]
    })
    rowSds(s)
  })
  
  
  rownames(table_mean) = c("pERM-oracle", "pERM-adaptive", "Spectral-oracle", "Spectral-adaptive", "Single-task", "r")
  rownames(table_sd) = c("pERM-oracle", "pERM-adaptive", "Spectral-oracle", "Spectral-adaptive", "Single-task", "r")
  if (setting_no %in% 1:2) {
    rescale_ratio1 = 1.5/5
    table_mean["r",] = table_mean["r",]*rescale_ratio1
  } else {
    rescale_ratio2 = 1.5/10
    table_mean["r",] = table_mean["r",]*rescale_ratio2
  }
  
  
  table_all = data.frame(mean = as.vector(table_mean), 
                         sd = as.vector(table_sd), 
                         h = rep(h_value, each = nrow(table_mean)), 
                         method = rep(factor(rownames(table_mean), 
                                             levels = c("pERM-adaptive", "pERM-oracle", "Spectral-adaptive", "Spectral-oracle", "Single-task", "r")), 
                                      length(h_value)))
  
  
  if (setting_no %in% 1:2) {
    plt = ggplot() + geom_line(data = table_all, aes(x = h, y = mean, group = method, color = method), size = 0.5) + 
      geom_point(data = table_all, aes(x = h, y = mean, shape = method, color = method), size = 2) +
      scale_shape_manual(values = c(10, 5, 11, 8, 7, 16),
                         labels = c("pERM-adaptive", "pERM-oracle", "Spectral-adaptive", "Spectral-oracle", "Single-task", TeX(r"($\hat{r}$)"))) + 
      ylab(TeX(r"(max error in $S$)")) +
      theme(plot.title = element_text(hjust = 0.5)) +
      scale_color_manual(values = c(hue_pal()(10)[8], hue_pal()(9)[6], hue_pal()(11)[5], hue_pal()(9)[9], hue_pal()(9)[8], "red"),
                         labels = c("pERM-adaptive", "pERM-oracle", "Spectral-adaptive", "Spectral-oracle", "Single-task", TeX(r"($\hat{r}$)"))) +
      scale_y_continuous(
        sec.axis = sec_axis(~ ./rescale_ratio1, name = TeX(r"($\hat{r}$)"), breaks = 2:5)  # Adjust the transformation
      ) + theme(
        axis.title.y.right = element_text(color = "red"),
        legend.position = "bottom",
        legend.key.spacing.x = unit(c(0.5,0.5,0.5,0.5,2,0.5), "cm")
      ) + guides(shape = guide_legend(nrow = 1))
  } else {
    plt2 = ggplot() + geom_line(data = table_all, aes(x = h, y = mean, group = method, color = method), size = 0.5) + 
      geom_point(data = table_all, aes(x = h, y = mean, shape = method, color = method), size = 2) +
      scale_shape_manual(values = c(10, 5, 11, 8, 7, 16),
                         labels = c("pERM-adaptive", "pERM-oracle", "Spectral-adaptive", "Spectral-oracle", "Single-task", TeX(r"($\hat{r}$)"))) + 
      ylab(TeX(r"(max error in $S$)")) +
      theme(plot.title = element_text(hjust = 0.5)) +
      scale_color_manual(values = c(hue_pal()(10)[8], hue_pal()(9)[6], hue_pal()(11)[5], hue_pal()(9)[9], hue_pal()(9)[8], "red"),
                         labels = c("pERM-adaptive", "pERM-oracle", "Spectral-adaptive", "Spectral-oracle", "Single-task", TeX(r"($\hat{r}$)"))) +
      scale_y_continuous(
        sec.axis = sec_axis(~ ./rescale_ratio2, name = TeX(r"($\hat{r}$)"))  # Adjust the transformation
      ) + theme(
        axis.title.y.right = element_text(color = "red"),
        legend.position = "bottom",
        legend.key.spacing.x = unit(c(0.5,0.5,0.5,0.5,2,0.5), "cm")
      ) + guides(shape = guide_legend(nrow = 1))
  }
  
  
  if (setting_no == 1) {
    plot_list[[setting_no]] = plt + ggtitle(TeX(r"($n = 100, p = 50, r = 5, T = 50, \epsilon = 0$)"))
  } else if (setting_no == 2){
    plot_list[[setting_no]] = plt + ggtitle(TeX(r"($n = 100, p = 50, r = 5, T = 50, \epsilon = 4\%$)"))
  } else if (setting_no == 3){
    plot_list[[setting_no]] = plt2 + ggtitle(TeX(r"($n = 150, p = 80, r = 10, T = 50, \epsilon = 0$)"))
  } else if (setting_no == 4){
    plot_list[[setting_no]] = plt2 + ggtitle(TeX(r"($n = 150, p = 80, r = 10, T = 50, \epsilon = 4\%$)"))
  }
  
}

# 10 x 7 PDF -> Figure 10
ggarrange(plotlist = plot_list, nrow = 2, ncol = 2, common.legend = TRUE, legend = "bottom")


# ----------------------------------
# real data
# ----------------------------------

# ----------------------------------
# Section 5.2: real-data
table_har_mean = sapply(c(5,10,15), function(r){
  table_result = matrix(nrow = 2, ncol = 7, dimnames = list(c('error', 'sd'),
                                                            c('Single-task', 'Pooled', 'ERM', 'ARMUL', 'pERM', 'spectral', 'GLasso')))
  
  # error
  table_result['error', ] = rowMeans(sapply(0:99, function(j){
    D = try(read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/real-data/har_r/r_',r,'_',
                            j,'.csv'), header = FALSE))
    if (class(D) != 'try-error') {
      apply(D, 1, mean)
    } else {
      rep(NA, 7)
    }
  }), na.rm = TRUE)
  
  table_result['sd', ] = apply(sapply(0:99, function(j){
    D = try(read.csv(paste0('/Users/yetian/Desktop/Dropbox/Columbia/Research/Project/Representation-MTL/experiments/real-data/har_r/r_',r,'_',
                            j,'.csv'), header = FALSE))
    if (class(D) != 'try-error') {
      apply(D, 1, mean)
    } else {
      rep(NA, 7)
    }
  }), 1, function(x){sd(x, na.rm = TRUE)})
  
  table_result
}, simplify = FALSE)


# print the latex table -> Table 2
r_list = c(5,10,15)
for (i in 1:4) {
  if (i == 1) {
    cat("$r$/Method ")
    cat(paste0("& ", colnames(table_har_mean[[1]])))
    cat("\\\\ \\hline")
    cat("\n")
  } else {
    for (j in 1:(ncol(table_har_mean[[1]])+1)) {
      if (j == 1) {
        cat(paste0("r = ", r_list[i-1]), " ")
      } else {
        cat(paste0("& ", round(table_har_mean[[i-1]]["error", j-1]*100, 2), " (", round(table_har_mean[[i-1]]["sd", j-1]*100, 2), ") "))
      }
      if (j == ncol(table_har_mean[[1]])+1) {
        cat("\\\\ \\hline")
        cat("\n")
      }
    }
  }
}





