

run_msd = function(none, reg, approx, name)
{
  level = 3
  
  msd_none = read.csv(none)
  colnames(msd_none) = c("Dataset", "Accuracy", "Variance", "LIME-PF", "LIME-NF", "LIME-Stability")
  
  msd_reg = read.csv(reg)
  colnames(msd_reg) = c("Dataset", "Accuracy", "Variance", "LIME-PF", "LIME-NF", "LIME-Stability") 
  
  msd_1d = read.csv(approx)
  colnames(msd_1d) = c("Dataset", "Accuracy", "Variance", "LIME-PF", "LIME-NF", "LIME-Stability")
  
  msd  = data.frame(t(msd_none), t(msd_reg), t(msd_1d))
  colnames(msd) = c("Unregularized", "Regularized", "Regularized1D")
  

  msd = msd[-c(1), ]
  
  data = as.matrix(msd)
  
  for(ii in c(1:3))
  {
    data[1, ii] = signif(as.numeric(data[1, ii]), level)
  }
  
  for(ii in c(2:5))
  {
    for(jj in c(1:3))
    {
      x = data[ii,jj]
      x = gsub("\\[","",x)
      x = gsub("\\]","",x)
      data[ii,jj] = signif(as.numeric(x),level)
    }
  }
  
  msd = data.frame(data)
  
  write.csv(msd, name)
}

run_msd("../MSD-None/results_mean.csv", "../MSD-LF/results_mean.csv", "../MSD-LF1D/results_mean.csv", "msd_mean.csv")
run_msd("../MSD-None/results_sd.csv", "../MSD-LF/results_sd.csv", "../MSD-LF1D/results_sd.csv", "msd_sd.csv")

msd_mean = as.matrix(read.csv("msd_mean.csv", row.names = 1))
msd_sd = as.matrix(read.csv("msd_sd.csv", row.names = 1))

msd = as.matrix(msd_mean)

for(ii in c(1:3))
{
  msd[1,ii] = paste(toString(msd_mean[1,ii])," (",toString(msd_sd[1,ii]),")", sep = "")
}

for(ii in c(2:5))
{
  for(jj in c(1:3))
  {
    msd[ii,jj] = paste(toString(msd_mean[ii,jj])," (",toString(msd_sd[ii,jj]),")", sep = "")
  }
}

msd = data.frame(msd)
write.csv(msd, "msd.csv")
