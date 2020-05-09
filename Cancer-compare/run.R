
run_cancer = function(reg, over, senn, name)
{
  
  d_reg = read.csv(reg)
  d_over = read.csv(over)
  d_senn = read.csv(senn)
  
  d = rbind(d_reg, d_over)
  colnames(d) = c("Dataset", "Acc", "Variance", "MAPLE-PF", "MAPLE-NF", "MAPLE-S", "LIME-PF", "LIME-NF", "LIME-S")
  d = subset(d, select = -c(Dataset, Variance))
  
  d = data.frame(apply(d, 2, function(x) {x <- gsub("\\[", "", x)}))
  d = data.frame(apply(d, 2, function(x) {x <- gsub("\\]", "", x)})) 
  colnames(d) = c("Acc", "MAPLE-PF", "MAPLE-NF", "MAPLE-S", "LIME-PF", "LIME-NF", "LIME-S")
  
  for(ii in c(1:7))
  {
    d[ , ii] = as.numeric(as.character(d[ , ii]))
  }
  
  
  colnames(d_senn) = c("Dataset", "Acc", "MAPLE-PF", "MAPLE-NF", "MAPLE-S", "LIME-PF", "LIME-NF", "LIME-S")
  d_senn = subset(d_senn, select = -c(Dataset))
  
  d = rbind(d, d_senn)
  
  colnames(d) = c("Acc", "MAPLE-PF", "MAPLE-NF", "MAPLE-S", "LIME-PF", "LIME-NF", "LIME-S")

  
  d = signif(d, 2)
  write.csv(d, name)
}

run_cancer("../Cancer-normal/results_mean.csv", "../Cancer-over/results_mean.csv", "../Cancer-senn/results_mean.csv", "cancer_mean.csv")
run_cancer("../Cancer-normal/results_sd.csv", "../Cancer-over/results_sd.csv", "../Cancer-senn/results_sd.csv", "cancer_sd.csv")

d_mean = read.csv("cancer_mean.csv", row.names = 1)
d_sd = read.csv("cancer_sd.csv", row.names = 1)

d = d_mean

for(ii in c(1:3))
{
  for(jj in c(1:7))
  {
    d[ii,jj] = paste(toString(d_mean[ii,jj])," (",toString(d_sd[ii,jj]),")", sep = "")
  }
}

write.csv(d, "cancer.csv")
