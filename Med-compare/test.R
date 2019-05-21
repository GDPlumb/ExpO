data = as.matrix(read.csv("med.csv"))
n = 10

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

parse2 = function(x)
{
  x = as.character(x)
  
  x = strsplit(x, ", ")[[1]]
  x1 = x[1]
  x2 = x[2]
  
  d1 = parse(x1)
  d2 = parse(x2)
  
  return(c(d1, d2))
}

for(ii in c(1:8))
{
  if(ii == 1)
  {
    out = parse(data[ii,2])
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
  else
  {
    out = parse2(data[ii, 2])
    u1_none = out[1]
    sd1_none = out[2]
    u2_none = out[3]
    sd2_none = out[4]
    
    out = parse2(data[ii, 3])
    u1_reg = out[1]
    sd1_reg = out[2]
    u2_reg = out[3]
    sd2_reg = out[4]
    
    out = parse2(data[ii, 4])
    u1_1d = out[1]
    sd1_1d = out[2]
    u2_1d = out[3]
    sd2_1d = out[4]
    
    if(t.test2(u1_none, u1_reg, sd1_none, sd1_reg, n, n)[["p-value"]] < 0.05)
    {
      print(paste(data[ii,1], "positive none vs full"))
    }
    
    if(t.test2(u1_1d, u1_none, sd1_1d, sd1_none, n, n)[["p-value"]] < 0.05)
    {
      print(paste(data[ii,1], "positive none vs 1d"))
    }
    
    if(t.test2(u1_1d, u1_reg, sd1_1d, sd1_reg, n, n)[["p-value"]] < 0.05)
    {
      print(paste(data[ii,1], "positive 1d vs full"))
    }
    
    if(t.test2(u2_none, u2_reg, sd2_none, sd2_reg, n, n)[["p-value"]] < 0.05)
    {
      print(paste(data[ii,1], "negative none vs full"))
    }
    
    if(t.test2(u2_1d, u2_none, sd2_1d, sd2_none, n, n)[["p-value"]] < 0.05)
    {
      print(paste(data[ii,1], "negative none vs 1d"))
    }
    
    if(t.test2(u2_1d, u2_reg, sd2_1d, sd2_reg, n, n)[["p-value"]] < 0.05)
    {
      print(paste(data[ii,1], "negative 1d vs full"))
    }
  }


}