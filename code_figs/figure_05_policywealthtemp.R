source("00_dependencies.R")


#read in DHS data that is georeferenced to geo2 (district)
dta <- read.csv('../data/output/cluster_pred_dhs_indices_gadm2.csv') 

#downloaded and kept bio1, per http://www.worldclim.org/bioclim
r <- raster('../data/wc2.0_bio_10m_01.tif')
pts <- data.frame(dta$lon,dta$lat)
dta$tmean <- raster::extract(r,pts)
r <- raster('../data/wc2.0_bio_10m_05.tif') #raster of max temperature???
dta$tmax <- raster::extract(r,pts)
r <- raster('../data/wc2.0_bio_10m_12.tif') #raster of precipitation????
dta$prec <- raster::extract(r,pts)


# function for running polynomials of chosen order, here on monthly tmax w fixed effects on svyid 
#' x: name of the variable you want to model
#' xx: the temperatures at which you'd like to estimate the modeled variable
#' ord: order of polynomial, default 3
runreg <- function(x,xx=seq(20,40,0.1),ord=3,center=25,data=dta) {
    fmla <- as.formula(paste0(x,"~ poly(tmax,ord,raw=T) | svyid"))
    mod <- felm(fmla, data=data)
    yy = as.vector(t(as.matrix(coef(mod)[1:ord]))%*%t(matrix(nrow=length(xx),ncol=ord,data=poly(xx,ord,raw=T))))
    yy = yy-yy[xx==center]
    mx = xx[which(yy==max(yy))]  #temperature at which the thing maximizes
    return(list(yy,mx))
}

or = 4
cnt = 30
xx=seq(20,40,0.1)
pr = list()

set.seed(52390)

for (i in 1:100) {  #bootstrap
    smp <- sample(1:dim(dta)[1],replace=T)
    y1 <- runreg("survey",ord=or,center=cnt,data=dta[smp,],xx=xx)
    y2 <- runreg("index",ord=or,center=cnt,data=dta[smp,],xx=xx)
    y3 <- runreg("Resnet.18.RGB.Transfer",ord=or,center=cnt,data=dta[smp,],xx=xx)
    y4 <- runreg("Ridge.NL.mean.scalar",ord=or,center=cnt,data=dta[smp,],xx=xx)  #KNN nightlights
    y5 <- runreg("KNN.NL.mean.scalar",ord=or,center=cnt,data=dta[smp,],xx=xx)  #KNN nightlights
    pr[[i]] <- list(y1,y2,y3,y4,y5)
}
# point estimate
y1 <- runreg("survey",ord=or,center=cnt,data=dta,xx=xx)
y2 <- runreg("index",ord=or,center=cnt,data=dta,xx=xx)
y3 <- runreg("Resnet.18.RGB.Transfer",ord=or,center=cnt,data=dta,xx=xx)
y4 <- runreg("Ridge.NL.mean.scalar",ord=or,center=cnt,data=dta,xx=xx)
y5 <- runreg("KNN.NL.mean.scalar",ord=or,center=cnt,data=dta,xx=xx)
pt <- list(y1,y2,y3,y4,y5)

clz <- c("black","red","blue","orange","green")
cll <- apply(sapply(clz, col2rgb)/255, 2, function(x) rgb(x[1], x[2], x[3], alpha=0.05)) 

att=seq(-0.85,-0.7,length.out = 4)
pdf(file="../raw_fig/Figure5.pdf",width=10,height=5)
par(mfrow=c(1,2))
plot(xx,y1[[1]],type="n",las=1,xla="maximum temperature (C)",ylab="wealth index")
for (i in 1:100) {
    for (j in 1:5) {
        lines(xx,pr[[i]][[j]][[1]],col=cll[j])
    }
}
for (j in 1:5) {
    lines(xx,pt[[j]][[1]],col=clz[j])
}


# try poverty targeting experiment

# targeting across the whole distribution (vs targeting within-country)
dta <- read.csv('../data/output/cluster_pred_dhs_indices_gadm2.csv')  #using estimates chris sent on slack Tues Oct 16th
acc <- c()
ts <- seq(0.1,0.5,0.01)
for (thresh in ts) {
    pt <- dta$survey < quantile(dta$survey,probs=c(thresh))
    ll=length(pt)
    # calculate accuracies
    ac1 <- sum((dta$index < quantile(dta$index,probs=c(thresh)))==pt)/ll  #MSNL model
    ac2 <- sum((dta$Resnet.18.RGB.Transfer < quantile(dta$Resnet.18.RGB.Transfer,probs=c(thresh)))==pt)/ll  #transfer learning
    ac3 <- sum((dta$Ridge.NL.mean.scalar < quantile(dta$Ridge.NL.mean.scalar,probs=c(thresh)))==pt)/ll  #scalar NL
    ac4 <- sum((dta$KNN.NL.mean.scalar < quantile(dta$KNN.NL.mean.scalar,probs=c(thresh)))==pt)/ll  #scalar NL
    acc <- rbind(acc,c(thresh,ac1,ac2,ac3,ac4))
}
clz <- c("black","red","blue","orange", "green")
plot(acc[,1],acc[,2],type="l",col=clz[2],xlab="targeting threshold (percentile)",ylab="targeting accuracy",ylim=c(0.5,1),las=1,lwd=2,xaxt="n")
for (i in 3:5) {lines(acc[,1],acc[,i],col=clz[i],lwd=2)}
axis(1,at=seq(0.1,0.5,0.1),paste0(seq(0.1,0.5,0.1)*100,"th"))
dev.off()