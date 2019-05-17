

run_med = function(none, reg, approx, name)
{
  level = 3
  
  med_none = read.csv(none)
  colnames(med_none) = c("Dataset", "Accuracy", "Variance", "MAPLE-PF", "MAPLE-NF", "MAPLE-Stability", "LIME-PF", "LIME-NF", "LIME-Stability")
  
  med_reg = read.csv(reg)
  colnames(med_reg) = c("Dataset", "Accuracy", "Variance", "MAPLE-PF", "MAPLE-NF", "MAPLE-Stability", "LIME-PF", "LIME-NF", "LIME-Stability") 
  
  med_1d = read.csv(approx)
  colnames(med_1d) = c("Dataset", "Accuracy", "Variance", "MAPLE-PF", "MAPLE-NF", "MAPLE-Stability", "LIME-PF", "LIME-NF", "LIME-Stability")
  
  med  = data.frame(t(med_none), t(med_reg), t(med_1d))
  colnames(med) = c("Unregularized", "Regularized", "Regularized1D")

  med = med[-c(1), ]
  
  data = as.matrix(med)
  
  for(ii in c(1:3))
  {
    data[1, ii] = signif(as.numeric(data[1, ii]), level)
  }
  
  for(ii in c(2:8))
  {
    for(jj in c(1:3))
    {
      x = data[ii,jj]
      x = gsub("\\[","",x)
      x = gsub("\\]","",x)
      x = strsplit(x, " ")[[1]]
      x[1] = signif(as.numeric(x[1]),level)
      x[2] = signif(as.numeric(x[2]),level)
      data[ii,jj] = paste(x[1], ", ", x[2], sep = "")
    }
  }
  
  med = data.frame(data)
  
  write.csv(med, name)
}

run_med("../Med-None/results_mean.csv", "../Med-LF/results_mean.csv", "../Med-LF1D/results_mean.csv", "med_mean.csv")
run_med("../Med-None/results_sd.csv", "../Med-LF/results_sd.csv", "../Med-LF1D/results_sd.csv", "med_sd.csv")

med_mean = as.matrix(read.csv("med_mean.csv", row.names = 1))
med_sd = as.matrix(read.csv("med_sd.csv", row.names = 1))

med = as.matrix(med_mean)

for(ii in c(1:3))
{
  med[1,ii] = paste(toString(med_mean[1,ii])," (",toString(med_sd[1,ii]),")", sep = "")
}

for(ii in c(2:8))
{
  for(jj in c(1:3))
  {
    m = strsplit(med_mean[ii,jj], ", ")[[1]]
    sd = strsplit(med_sd[ii,jj], ", ")[[1]]
    
    med[ii,jj] = paste(m[1], " (", sd[1], "), ", m[2], " (", sd[2], ")", sep = "")
  }
}

med = data.frame(med)
write.csv(med, "med.csv")
