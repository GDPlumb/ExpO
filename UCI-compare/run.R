
run_uci = function(none, reg, approx, name)
{
  
  uci_none = read.csv(none)
  uci_reg = read.csv(reg)
  uci_1d = read.csv(approx)

  colnames(uci_none) = c("Dataset", "MSE", "Variance", "MAPLE-PF", "MAPLE-NF", "MAPLE-Stability", "LIME-PF", "LIME-NF", "LIME-Stability")
  colnames(uci_reg) = c("Dataset", "MSE Reg", "Variance Reg", "MAPLE-PF Reg", "MAPLE-NF Reg", "MAPLE-Stability Reg", "LIME-PF Reg", "LIME-NF Reg", "LIME-Stability Reg")
  colnames(uci_1d) = c("Dataset", "MSE 1D", "Variance 1D", "MAPLE-PF 1D", "MAPLE-NF 1D", "MAPLE-Stability 1D", "LIME-PF 1D", "LIME-NF 1D", "LIME-Stability 1D")
  
  uci_none = t(uci_none)
  uci_reg = t(uci_reg)
  uci_1d = t(uci_1d)

  uci = rbind(uci_none, uci_reg, uci_1d)
  colnames(uci) = uci[1, ]

  uci = uci[-c(1, 10, 19),] #Drop the "dataset" rows
  uci = uci[-c(2, 10, 18), ] #Drop the "variance" rows

  order = c(1,8,15,2,9,16,3,10,17,4,11,18,5,12,19,6,13,20,7,14,21)
  uci = uci[order, ]

  uci = data.frame(apply(uci, 2, function(x) {x <- gsub("\\[", "", x)}))
  uci = data.frame(apply(uci, 2, function(x) {x <- gsub("\\]", "", x)}))

  for(ii in c(1:6))
  {
    uci[ , ii] = as.numeric(as.character(uci[ , ii]))
  }

  uci = signif(uci, 2)
  write.csv(uci, name)
}

run_uci("../UCI-None/results_mean.csv", "../UCI-LF/results_mean.csv", "../UCI-LF1D/results_mean.csv", "uci_mean.csv")
run_uci("../UCI-None/results_sd.csv", "../UCI-LF/results_sd.csv", "../UCI-LF1D/results_sd.csv", "uci_sd.csv")

uci_mean = read.csv("uci_mean.csv", row.names = 1)
uci_sd = read.csv("uci_sd.csv", row.names = 1)

uci = uci_mean

for(ii in c(1:21))
{
  for(jj in c(1:6))
  {
    uci[ii,jj] = paste(toString(uci_mean[ii,jj])," (",toString(uci_sd[ii,jj]),")", sep = "")
  }
}

write.csv(uci, "uci.csv")
