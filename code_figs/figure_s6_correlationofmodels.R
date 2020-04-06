source("00_dependencies.R")

pred = read.csv('../data/predictions/dhs_ooc_preds_ooc.csv')
pred = dplyr::select(pred, label, Resnet.18.MS.NL.concat, Resnet.18.MS, Resnet.18.NL, 
                     Resnet.18.MS.Transfer, Ridge.NL.mean.scalar, KNN.NL.mean.scalar)
names(pred) = c("Survey", "MSNL", "MS", "NL", "Transfer", "Scalar NL", "KNN")

c = round(cor(pred),2)
c[upper.tri(c)] = NA
c = melt(c)

g = ggplot(c, aes(Var2, Var1, fill = value)) + 
    geom_tile(colour="white", size=1.5, stat="identity") + 
    geom_text(aes(Var2, Var1, label = value), color="black", size=rel(3)) +
    scale_fill_gradient(low = "orangered2", high = "dodgerblue", space = "Lab", na.value = "white", guide = "colourbar") +
    scale_x_discrete(expand = c(0, 0), position="top", limits = rev(levels(c$Var2))) +
    scale_y_discrete(expand = c(0, 0)) +
    xlab("") + ylab("") +
    theme_anne(size=12, font="sans") + theme(legend.position="none", 
                                             axis.ticks = element_blank(),
                                             axis.line=element_blank())

ggsave("../processed_fig/FigureS6.pdf", g, width=6.9, height=6.3, units="in", dpi=300)
