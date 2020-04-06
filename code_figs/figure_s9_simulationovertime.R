source("00_dependencies.R")


n = 1000
xsd = 0.9
nsd = 0.6  #sd of noise in cross section
mc = 0.08   #mean change in features
sdc = 0.25   #sd of change in features

rc1 = rc2 = rd1 = rc1w = rc2w = rd1w = c()  #vectors to hold results
for (i in 1:100) {
    
    x1 = rnorm(n,0,xsd)  #features in year 1
    ee = rnorm(n,0,nsd/2)  #persistent component unrelated to features
    y1 = x1 + rnorm(n,0,nsd)  
    
    # generate cluster-specific change in features
    clustnum = data.frame(clust=round(rnorm(1000,25,10)))  #generate cluster id for each obs, with different numbers of hholds in each cluster
    clust = clustnum %>% dplyr::group_by(clust) %>% dplyr::summarise(n=n())
    chg = data.frame(clust,chg=rnorm(dim(clust)[1],mc,sdc))  #generate a cluster specific change
    clust = left_join(clustnum,chg)  # so this is cluster specific change in features related to wealth
    x2 = x1 + clust$chg  #houseshold change in features
    y2 = x2 + rnorm(n,0,nsd)
    yd = y2 - y1  #change in y
    
    xd = x2 - x1
    data=data.frame(hhold=1:n,clust,x1,x2,y1,y2,yd,xd)
    
    cdata = data %>% group_by(clust) %>% summarise_all(mean)  #collapse to cluster level
    rc1 = c(rc1,summary(lm(y1 ~ x1,data=cdata))$r.squared)  #cross sectional regression in first year on observed
    rc2 = c(rc2,summary(lm(y2 ~ x2,data=cdata))$r.squared)  # cross sectional regression in second year on observed
    rd1 = c(rd1,summary(lm(yd ~ xd,data=cdata))$r.squared)  # over time regression on observed
    print(i)
}

pdf('../processed_fig/FigureS9.pdf',width=12,height=4)
par(mfrow=c(1,3)) 
hist(rc1,main="cross section year 1",xlim=c(0,1),las=1,xlab="r2")
abline(v=mean(rc1),col="red",lty=2,lwd=2)
hist(rc2,main="cross section year 2",xlim=c(0,1),las=1,xlab="r2")
abline(v=mean(rc2),col="red",lty=2,lwd=2)
hist(rd1,main="deltas",xlim=c(0,1),las=1,xlab="r2")
abline(v=mean(rd1),col="red",lty=2,lwd=2)
dev.off()
