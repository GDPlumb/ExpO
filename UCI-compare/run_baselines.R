
run_uci = function(none, l2, l1, name)
{
  
  uci_none = read.csv(none)
  uci_l2 = read.csv(l2)
  uci_l1 = read.csv(l1)

  colnames(uci_none) = c("Dataset", "MSE", "Variance", "MAPLE-PF", "MAPLE-NF", "MAPLE-Stability", "LIME-PF", "LIME-NF", "LIME-Stability")
  colnames(uci_l2) = c("Dataset", "MSE L2", "Variance L2", "MAPLE-PF L2", "MAPLE-NF L2", "MAPLE-Stability L2", "LIME-PF L2", "LIME-NF L2", "LIME-Stability L2")
  colnames(uci_l1) = c("Dataset", "MSE L1", "Variance L1", "MAPLE-PF L1", "MAPLE-NF L1", "MAPLE-Stability L1", "LIME-PF L1", "LIME-NF L1", "LIME-Stability L1")
  
  uci_none = t(uci_none)
  uci_l2 = t(uci_l2)
  uci_l1 = t(uci_l1)

  uci = rbind(uci_none, uci_l2, uci_l1)
  colnames(uci) = uci[1, ]

  uci = uci[-c(1, 10, 19), ] #Drop the "dataset" rows
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

run_uci("../UCI-None/results_mean.csv", "../UCI-L2/results_mean.csv", "../UCI-L1/results_mean.csv", "uci_mean_baselines.csv")
