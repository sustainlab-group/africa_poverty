source("00_dependencies.R")

################################################################
# make LSMS index of deltas plot
################################################################

lsms = read.csv("../data/predictions/lsms_indexofdelta_preds_incountry.csv")

cross = data.frame(country=c("ethiopia", "malawi", "nigeria", "tanzania", "uganda"),
                   iso3 = c("ETH", "MWI", "NGA", "TZA", "UGA"))
lsms = merge(lsms, cross, by = "country")
lsms = data_to_geolev(lsms, shapefile_loc="../data/shapefiles/gadm/")
lsms_agg = lsms %>% dplyr::group_by(geolev, year1, year2, country) %>% 
    dplyr::summarize(label = mean(label, na.rm=T), preds_ms = mean(preds_ms, na.rm=T), 
                     preds_nl = mean(preds_nl, na.rm=T), 
                     preds_msnl_concat = mean(preds_msnl_concat, na.rm=T), n = n())

r2s = c(cor(lsms$preds_msnl_concat, lsms$label)^2, 
        cor(lsms$preds_nl, lsms$label)^2, 
        cor(lsms$preds_ms, lsms$label)^2)
r2s = data.frame(r2=r2s, type=c("msnl", "nl", "ms"))

r2s_agg = c(cor(lsms_agg$preds_msnl_concat, lsms_agg$label)^2, 
            cor(lsms_agg$preds_nl, lsms_agg$label)^2, 
            cor(lsms_agg$preds_ms, lsms_agg$label)^2)
r2s_agg = data.frame(r2=r2s_agg, type=c("msnl", "nl", "ms"))

rug = ggplot() + geom_rug(aes(x=r2, color=type), r2s, size=1) +  
    theme_anne(font="sans") + xlim(0, 0.43)

rug_agg = ggplot() + geom_rug(aes(x=r2, color=type), r2s_agg, size=1) + 
    theme_anne(font="sans") + xlim(0, 0.43)

point = panel(lsms[, c("preds_ms", "label")], a=0.07, font="sans", size=14, square=T) + 
    ylab("LSMS index of differences") +
    xlab("MS Predicted index of differences") 

point_agg = panel(lsms_agg[, c("preds_ms", "label")], a=0.12, font="sans", size=14, w=lsms_agg$n, square=T) + 
    ylab("LSMS index of differences") +
    xlab("MS Predicted index of differences") 

ggsave("../raw_fig/Figure4_rug.pdf", rug, "pdf", width=5, height=4, dpi=300)
ggsave("../raw_fig/Figure4_rug_agg.pdf", rug_agg, "pdf", width=5, height=4, dpi=300)
ggsave("../raw_fig/Figure4_point.pdf", point, "pdf", width=5, height=5, dpi=300)
ggsave("../raw_fig/Figure4_point_agg.pdf", point_agg, "pdf", width=5, height=5, dpi=300)