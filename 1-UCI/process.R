
data = read.csv("results.csv")
data[-1] = round(data[-1], 5)
colnames(data) = c("Dataset", "Acc", "Standard", "Causal", "Stability")
data = data[order(data$Dataset), ]
write.csv(data, "table.csv", row.names = F)
