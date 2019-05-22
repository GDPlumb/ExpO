data = as.matrix(read.csv("msd.csv"))
n = 20

t.test2 <- function(m1,m2,s1,s2,n1,n2,m0=0,equal.variance=FALSE)
{
  if( equal.variance==FALSE ) 
  {
    se <- sqrt( (s1^2/n1) + (s2^2/n2) )
    # welch-satterthwaite df
    df <- ( (s1^2/n1 + s2^2/n2)^2 )/( (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1) )
  } else
  {
    # pooled standard deviation, scaled by the sample sizes
    se <- sqrt( (1/n1 + 1/n2) * ((n1-1)*s1^2 + (n2-1)*s2^2)/(n1+n2-2) ) 
    df <- n1+n2-2
  }      
  t <- (m1-m2-m0)/se 
  dat <- c(m1-m2, se, t, 2*pt(-abs(t),df))    
  names(dat) <- c("Difference of means", "Std Error", "t", "p-value")
  return(dat) 
}

parse = function(x)
{
  x = as.character(x)
  x = strsplit(x, " ")[[1]]
  
  u = as.numeric(x[1])
  
  sd = gsub("\\(", "", x[2])
  sd = gsub("\\)", "", sd)
  sd = as.numeric(sd)
  
  return(c(u,sd))
}

for(ii in c(1:5))
{
  
  out = parse(data[ii, 2])
  u_none = out[1]
  sd_none = out[2]

  out = parse(data[ii, 3])
  u_reg = out[1]
  sd_reg = out[2]
  
  out = parse(data[ii, 4])
  u_1d = out[1]
  sd_1d = out[2]

  if(t.test2(u_none, u_reg, sd_none, sd_reg, n, n)[["p-value"]] < 0.05)
  {
    print(paste(data[ii,1], "none vs full"))
  }
  
  if(t.test2(u_1d, u_none, sd_1d, sd_none, n, n)[["p-value"]] < 0.05)
  {
    print(paste(data[ii,1], "none vs 1d"))
  }
  
  if(t.test2(u_1d, u_reg, sd_1d, sd_reg, n, n)[["p-value"]] < 0.05)
  {
    print(paste(data[ii,1], "1d vs full"))
  }

}