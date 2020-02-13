source("00_dependencies.R")


# make Figure 3 for Africa poverty paper
# MB nov 12 2018
# updated AD may 21 2019

mat <- matrix(c(1,2,3,3,1,2,3,3,4,4,5,5,4,4,5,5),nrow = 4,byrow = T)
pdf('../raw_fig/Figure3.pdf',width=7,height=7)
layout(mat)

par(mar=c(5,3,4,0))
# PANEL A + B
mods <- c("Ridge.NL.mean.scalar","Resnet.18.RGB.Transfer","Resnet.18.MS","KNN.NL.mean.scalar","Resnet.18.NL","Resnet.18.MS.NL.concat")
nms <- c("linear scalar NL","CNN transfer","CNN MS","KNN scalar NL","CNN NL","CNN MS + NL")
colz <- c("green","orange","black","pink","blue","red")

mkplot <- function(pred) {
    pred$country_year = paste0(pred$country, "_", pred$year)
    plot(1,xlim=c(0,1),ylim=c(0.8,length(mods)+0.2),type="n",xlab=bquote(r^{2}),ylab="",las=1,axes=F)
    axis(1,at=seq(0,0.8,0.2),seq(0,0.8,0.2))
    for (i in mods) {
        sub <- data.frame(pred[,c("country_year","label")],x=pred[,i] )
        r2 <- summary(lm(label ~ x, data=sub))$r.squared
        out <- sub %>% group_by(country_year) %>% summarise(r2=summary(lm(label ~ x))$r.squared)
        n=which(mods==i)
        points(out$r2,rep(n,dim(out)[1]),pch=124,col="grey90",cex=2)
        points(mean(out$r2),n,pch=124,cex=2.5)
        text(mean(out$r2),n+0.5,round(mean(out$r2),2),cex=0.7)
        points(r2,n,pch=124,cex=2.5,col="red")  #full sample estimate
        text(r2,n+0.5,round(r2,2),cex=0.7,col="red")
        #  abline(v=0.7)
        # text(min(out$r2),n,round(mean(out$r2),2),cex=1)
    }
}

#village level derived indices from aggregated DHS data, no households identifiable
pred <- read.csv('../data/predictions/dhs_ooc_preds_ooc.csv')
mkplot(pred)
text(0.8,1:5,nms,cex=0.5,pos=4)

#village level derived indices from aggregated DHS data, no households identifiable
pred <- read.csv('../data/predictions/dhs_incountry_preds.csv')
pred$country_year = paste(pred$country, pred$year, sep="_")
mkplot(pred)

par(mar=c(5,4,4,2))

# PANEL C:  r2 by % of data kept
modf <- c("Resnet-18 RGB Transfer","Resnet-18 MS","Resnet-18 NL","Resnet-18 MS+NL concat")
nms <- c("CNN transfer","CNN MS","CNN NL","CNN MS + NL")
colz <- c("orange","black","blue","red")
dta <- fread('../data/predictions/r2_keep.csv') #NEEDS TO BE REPLACED

####add the knn 
#village level derived indices from aggregated DHS data, no households identifiable
nl = readRDS('../data/predictions/dhs_ooc_preds_ooc_w_nl.csv')
folds = list(c("angola", "cote_d_ivoire",  "ethiopia", "mali", "rwanda"), 
             c("benin", "burkina_faso", "guinea",  "sierra_leone", "tanzania"),
             c("cameroon", "ghana", "malawi", "zimbabwe"),
             c("democratic_republic_of_congo", "mozambique", "nigeria", "togo", "uganda"),
             c("kenya", "lesotho", "senegal", "zambia"))

dt <- dta %>% group_by(model, keep_frac) %>% summarize(r2 = mean(r2))

plot(1,xlim=c(0,1),ylim=c(0.5,0.75),type="n",xlab="% of data used in training",ylab=bquote(r^{2} ~ "on held out countries"),las=1,axes=F)
axis(1,at=seq(0,1,0.2),seq(0,1,0.2))
axis(2,at=seq(0.5,0.75,0.05),seq(0.5,0.75,0.05),las=1)
for (i in modf) {
    lines(dt$keep_frac[dt$model==i],dt$r2[dt$model==i],col=colz[which(modf==i)])
}
text(0.8,dt$r2[dt$keep_frac==1&dt$model%in%modf],dt$model[dt$keep_frac==1&dt$model%in%modf],cex=0.5)


# PANEL D: rural/urban
pred <- read.csv('../data/predictions/dhs_ooc_preds_ooc.csv')
clz <- c("red","blue")
cll <- apply(sapply(clz, col2rgb)/255, 2, function(x) rgb(x[1], x[2], x[3], alpha=0.03)) 
pred$col <- cll[1]
pred$col[pred$urban=="True"] <- cll[2]
plotreg <- function(x,col="red",lab=-1) {
    mod <- lm(label ~ Resnet.18.MS.NL.concat, data=pred[pred$urban==x,])
    qnt <- round(quantile(pred$Resnet.18.MS.NL.concat[pred$urban==x],probs=c(0.025,0.975)),2)
    yy <- coef(mod)[1] + coef(mod)[2]*qnt
    lines(qnt,yy,col=col,lwd=2)
    # text(1.5,lab,paste0("r2=",round(summary(mod)$r.squared,2)),col=col,cex=0.8)
    # text(1.5,lab,paste0(expression(r^{2}),"=",round(summary(mod)$r.squared,2)),col=col,cex=0.8)
    text(1.5,lab,bquote(r^{2}~.(paste0("=",round(summary(mod)$r.squared,2)))),col=col,cex=0.8)
}


plot(pred$Resnet.18.MS.NL.concat,pred$label,pch=19,col=pred$col,las=1,xlab="predicted wealth",ylab="observed wealth",cex=0.5,axes=F,xlim=c(-1.5,2),ylim=c(-1.8,2.5))
plotreg(x="True",col="blue",lab=-1)
plotreg(x="False",col="darkred",lab=-1.3)
att = -1:2
axis(1,at=att,att)
axis(2,at=att,att,las=1)

# add y-axis densities for surveys
cll <- apply(sapply(clz, col2rgb)/255, 2, function(x) rgb(x[1], x[2], x[3], alpha=0.5)) 
sc = 0.25  #height of densities
lp = -1.5  #left or lower edge
dn <- density(pred$label[pred$urban=="False"],bw=0.1)
dn$y <- dn$y/max(dn$y)*sc + lp
polygon(c(dn$y,rep(lp,length(dn$y))), c(dn$x,rev(dn$x)),col=cll[1],border = NA)
dn <- density(pred$label[pred$urban=="True"],bw=0.1)
dn$y <- dn$y/max(dn$y)*sc + lp
polygon(c(dn$y,rep(lp,length(dn$y))), c(dn$x,rev(dn$x)),col=cll[2],border = NA)

# x-axis densities for predicted
lp = -1.8  #left or lower edge
dn <- density(pred$Resnet.18.MS.NL.concat[pred$urban=="False"],bw=0.1)
dn$y <- dn$y/max(dn$y)*sc + lp
polygon(c(dn$x,rep(lp,length(dn$x))), c(dn$y,rev(dn$y)),col=cll[1],border = NA)
dn <- density(pred$Resnet.18.MS.NL.concat[pred$urban=="True"],bw=0.1)
dn$y <- dn$y/max(dn$y)*sc + lp
polygon(c(dn$x,rep(lp,length(dn$x))), c(dn$y,rev(dn$y)),col=cll[2],border = NA)


# PANEL E: r2 by wealth distribution

mods <- c("KNN.NL.mean.scalar","Ridge.NL.mean.scalar","Resnet.18.RGB.Transfer","Resnet.18.MS","Resnet.18.NL","Resnet.18.MS.NL.concat")
nms <- c("KNN", "scalar NL","CNN transfer","CNN MS","CNN NL","CNN MS + NL")
colz <- c("pink", "green","orange","black","blue","red")
dta <- read.csv('../data/predictions/r2_wealth_ooc.csv')

cumcor = function(x, y) {
    n = 1:length(x)
    res = cumsum(x*y) - cummean(x)*cumsum(y) - cummean(y)*cumsum(x) + n*cummean(x)*cummean(y)
    res = res / (n-1)
    
    var.x = (cumsum(x^2) - n*cummean(x)^2) / (n-1)
    var.y = (cumsum(y^2) - n*cummean(y)^2) / (n-1)
    
    cor = res/sqrt(var.x*var.y)
    cor = cor^2
    
    return(cor)
}
pred = read.csv('../data/predictions/dhs_ooc_preds_ooc.csv')
pred = pred[, c("label", mods)]
pred = pred[order(pred$label),]
pred$Ridge.NL.mean.scalar = cumcor(pred$Ridge.NL.mean.scalar, pred$label)
pred$KNN.NL.mean.scalar = cumcor(pred$KNN.NL.mean.scalar, pred$label)
pred$Resnet.18.RGB.Transfer = cumcor(pred$Resnet.18.RGB.Transfer, pred$label)
pred$Resnet.18.MS = cumcor(pred$Resnet.18.MS, pred$label)
pred$Resnet.18.NL = cumcor(pred$Resnet.18.NL, pred$label)
pred$Resnet.18.MS.NL.concat = cumcor(pred$Resnet.18.MS.NL.concat, pred$label)
dta = pred

#tail(dta)
qnt <- quantile(dta$label,probs=seq(0.1,1,0.1))
nn <- knn(data.frame(dta$label),data.frame(qnt),cl=1:dim(dta)[1])
plot(1,las=1,xlab="deciles of data used",ylab=bquote(r^{2} ~ "on held out countries"),type="n",xlim=c(1,10),ylim=c(0,0.75),axes=F)
axis(1,at=1:10,1:10)
axis(2,at=seq(0,0.7,0.1),seq(0,0.7,0.1),las=1)
for (i in 1:length(mods)) {
    lines(1:10,dta[nn,which(names(dta)==mods[i])],col=colz[i])
    text(9.5,dta[nn[10],which(names(dta)==mods[i])],nms[i],cex=0.5,pos=2)
}
abline(h=seq(0.1,0.7,0.2),lty=2,col="grey",lwd=0.5)


dev.off()