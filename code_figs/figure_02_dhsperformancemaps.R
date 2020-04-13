source("00_dependencies.R")

##########################################################################
# Plot r2 at the country  level
##########################################################################

cross = read.csv("../data/crosswalks/crosswalk_countries.csv", na.strings="")
africa = readRDS("../data/shapefiles/africa_gadm.rds")

#village level derived indices from aggregated DHS data, no households identifiable
cluster_gadm = read.csv("../data/output/cluster_pred_dhs_indices_gadm2.csv", stringsAsFactors = F)
country = cluster_gadm %>% dplyr::group_by(svyid, country) %>% 
    dplyr::summarise(r2 = cor(survey, index)^2, n=n(),
                     rmse = sqrt(mean( (survey - index)^2 ))) %>% 
    dplyr::group_by(country) %>% 
    dplyr::summarise(r2 = mean(r2), rmse = mean(rmse), n=sum(n))
country[country=="Côte d'Ivoire"] = "Cote d'Ivoire"
country = merge(country, cross, by.x="country", by.y="country_simp", all.x=T)
africa = merge(africa, country, by.x ="GID_0", by.y="iso3")
q = c(0.52, quantile(country$r2, c(1/6, 2/6, 3/6, 4/6, 5/6)), 1)
j = c(.53, .53+.07, .53+.07*2, .53+.07*3, .53+.07*4, .53+.07*5, 1)
t = c(.5, .6, .7, .75, .8, .85, 1)
t2 = c(.7, .6, .5, .4, .3, .2, .1)
m1 = map(africa, 'r2', '', '', mincol="#d73027", maxcol="#1a9850", 
                       breaks=t, font="sans", size=10) 
m1_rmse = map(africa, 'rmse', '', '', mincol="#1a9850", maxcol="#d73027", 
                            breaks=t2, font="sans", size=10) 
p = plyr::rename(cluster_gadm, c("survey" = "survey-measured asset wealth", 
                                 "index" = "satellite-predicted asset wealth"))
p1 = panel(p[, c(3,2)], font="sans", size=10, square=T)
mean(ddply(cluster_gadm,"svyid",function(x) cor(x$survey,x$index, use="c")^2)$V1)
black = cluster_gadm %>% dplyr::group_by(svyid) %>% dplyr::summarise()


##########################################################################
# Plot r2 at the country level, aggregating to geo2 level
##########################################################################

africa = readRDS("../data/shapefiles/africa_gadm.rds")
#geo2 level derived indices from aggregated DHS data, no households identifiable
g2 = read.csv("../data/output/geolevel2_dhs_indices_gadm2.csv", stringsAsFactors = F)
country = g2 %>% dplyr::group_by(svyid, country) %>% 
    dplyr::summarise(r2 = wtd.cor(dhs, predictions, weight=n.x)[[1]]^2, 
                     rmse = sqrt(mean( (dhs - predictions)^2 ))) %>% 
    dplyr::group_by(country) %>% 
    dplyr::summarise(r2 = mean(r2), 
                     rmse = mean(rmse))
country[country=="Côte d'Ivoire"] = "Cote d'Ivoire"
country = merge(country, cross, by.x="country", by.y="country_simp")
africa = merge(africa, country, by.x ="GID_0", by.y="iso3")
m2 = map(africa, 'r2', '', '', mincol="#d73027", maxcol="#1a9850", 
                       breaks=t, font="sans", size=10)
m2_rmse = map(africa, 'rmse', '', '', mincol="#1a9850", maxcol="#d73027", 
                            breaks=t2, font="sans", size=10) 
p = plyr::rename(g2, c("dhs" = "survey-measured asset wealth", 
                       "predictions" = "satellite-predicted asset wealth"))
p2 = panel(p[, c(5, 3)], w = g2$n.x, font="sans", size=10, square=T)
cor(p[,3], p[,5])^2 #pooled unweighted r2
p %>% dplyr::group_by(svyid) %>% 
    dplyr::summarize(r2 = cor(`survey-measured asset wealth`, 
                              `satellite-predicted asset wealth`)^2) %>% 
    dplyr::summarize(mean(r2)) #average country unweighted r2
mean(ddply(p,"svyid",function(x) wtd.cor(x$`survey-measured asset wealth`,
                                         x$`satellite-predicted asset wealth`, 
                                         weight=x$n.x+x$n.y)^2)$correlation)


##########################################################################
# Comparison with census data
##########################################################################

#derived indices from aggregated DHS data, no households identifiable, combined w census data
geolev = read.csv("../data/output/geolevel2_ipums_dhs_indices_ipums.csv") 
g2 = plyr::rename(geolev, c("dhs" = "survey-measured asset wealth", 
                            "predictions" = "satellite-predicted asset wealth",
                            "index" = "census-measured asset wealth"))
x = panel_set(data=list(g2[,c(4,2)], g2[,c(6, 2)]), name="", w=list(g2$n, g2$n), 
              font='sans', size=10, square=T)


##########################################################################
# Save image files
##########################################################################

ggsave("../raw_fig/Figure2ac_correlationsurvprediction.pdf", grid.arrange(p1, p2, nrow=1), 
       width=8, height=4, units="in", dpi=300)
ggsave("../raw_fig/Figure2b_countryr2.pdf", m1, width=4, height=4, units="in", dpi=300)
ggsave("../raw_fig/Figure2d_aggregatedcountryr2.pdf", m2, 
       width=4, height=4, units="in", dpi=300)

ggsave("../raw_fig/FigureS4a_countryrmse.pdf", m1_rmse, 
       width=4, height=4, units="in", dpi=300)
ggsave("../raw_fig/FigureS4b_aggregatedcountryrmse.pdf", m2_rmse, 
       width=4, height=4, units="in", dpi=300)

ggsave("../raw_fig/Figure2ef_correlationaggregated.pdf", x, width=8, height=4, units="in", dpi=400)
