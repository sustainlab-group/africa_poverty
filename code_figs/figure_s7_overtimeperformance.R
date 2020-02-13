source("00_dependencies.R")


################################################################
# make DHS figure
################################################################
pred = read.csv("../data/predictions/dhs_ooc_preds_ooc.csv")
pred = dplyr::select(pred, lat, lon, country, year, label, Resnet.18.MS.NL.concat, Resnet.18.NL, Resnet.18.MS, KNN.NL.mean.scalar)
names(pred) = c("lat", "lon", "country", "year", "dhs", "msnl", "nl", "ms", "knn_nl")

countries = unique(pred$country)
output = data.frame(country=NA, dhs=NA, nl=NA, msnl=NA, ms=NA, knn_nl=NA, year=NA, year2=NA, distance=NA, lat2=NA, lon2=NA, km=NA)

km_allowed = 10

for (country in 1:length(countries)) {
    
    country = pred[pred$country == countries[country], ]
    country = country[complete.cases(country), ]
    
    years = sort(unique(country$year))
    if (!length(years) > 1) {next}
    
    for(year in 1:(length(years)-1)) {
        
        #subset to correct year
        year1 = country[country$year == years[year], ]
        year2 = country[country$year == years[year+1], ]
        
        #convert to spatial points
        year1_point = SpatialPoints(year1[,c("lat", "lon")])
        year2_point = SpatialPoints(year2[,c("lat", "lon")])
        
        #find nearest neighbor
        tree = createTree(coordinates(year2_point))
        inds = knnLookup(tree, newdat=coordinates(year1_point), k=1)
        
        #combine w nearest neighbor
        year2 = year2[inds, c("dhs", "msnl", "ms", "nl", "knn_nl", "year", "lat", "lon")]
        names(year2) = paste0(names(year2), "2")
        combined_years = cbind(year1, year2)
        
        #find distance between and remove if too far away
        combined_years$distance = sqrt((combined_years$lat - combined_years$lat2)^2 + (combined_years$lon - combined_years$lon2)^2)
        combined_years = combined_years[combined_years$distance < km_allowed/100, ]
        if (nrow(combined_years) < 2) {next}
        
        #get differences
        combined_years$dhs = combined_years$dhs2 - combined_years$dhs
        combined_years$msnl = combined_years$msnl2 - combined_years$msnl
        combined_years$ms = combined_years$ms2 - combined_years$ms
        combined_years$nl = combined_years$nl2 - combined_years$nl
        combined_years = dplyr::select(combined_years, country, dhs, nl, msnl, ms, knn_nl, year, year2, distance, lat2, lon2)
        combined_years$km = km_allowed
        
        output = rbind(output, combined_years)
    }
}

output = output[complete.cases(output), ]

cross = data.frame(country=c("angola", "burkina_faso", "ghana", "kenya", "lesotho", "mali", "mozambique", 
                             "rwanda", "senegal", "zimbabwe", "ethiopia", "nigeria", "tanzania", "uganda"),
                   iso3 = c("AGO", "BFA", "GHA", "KEN", "LSO", "MLI", "MOZ", "RWA", "SEN", "ZWE", "ETH", "NGA", "TZA", "UGA"))
output = merge(output, cross, by = "country")
output = data_to_geolev(output, lat="lat2", lon="lon2", shapefile_loc="../data/shapefiles/gadm/")
output_agg = output %>% dplyr::group_by(geolev, year, year2, country) %>% 
    dplyr::summarize(dhs = mean(dhs, na.rm=T), 
                     msnl = mean(msnl, na.rm=T), 
                     ms = mean(ms, na.rm=T), 
                     nl = mean(nl, na.rm=T), 
                     knn_nl = mean(knn_nl, na.rm=T), 
                     n = n())

c = data.frame(variable = c("msnl", "nl", "ms"), 
               value=c(cor(output$dhs, output$msnl)^2, cor(output$dhs, output$nl)^2, 
               cor(output$dhs, output$ms)^2))
c_agg = data.frame(variable = c("msnl", "nl", "ms"), 
               value=c(cor(output_agg$dhs, output_agg$msnl)^2, cor(output_agg$dhs, output_agg$nl)^2, 
                       cor(output_agg$dhs, output_agg$ms)^2))


rug = ggplot() + geom_rug(aes(x=value, color=variable), c, size=1) + 
    theme_anne(font="sans") + xlim(0, 0.43)

rug_agg = ggplot() + geom_rug(aes(x=value, color=variable), c_agg, size=1) + 
    theme_anne(font="sans") + xlim(0, 0.43)

point = panel(output[, c("msnl", "dhs")], a=0.07, font="sans", size=14, square=T) + 
    ylab("Difference of DHS indexes inx closest DHS clusters") +
    xlab("Difference of MSNL predictions in closest DHS clusters") 

point_agg = panel(output_agg[, c("msnl", "dhs")], a=0.1, font="sans", size=14, w=output_agg$n, square=T) + 
    ylab("Difference of DHS indexes in closest DHS clusters") +
    xlab("Difference of MSNL predictions in closest DHS clusters") 

ggsave("../raw_fig/FigureS7_dhs_rug.pdf", rug, "pdf", width=5, height=4, dpi=300)
ggsave("../raw_fig/FigureS7_dhs_rug_agg.pdf", rug_agg, "pdf", width=5, height=4, dpi=300)
ggsave("../raw_fig/FigureS7_dhs_point.pdf", point, "pdf", width=5, height=5, dpi=300)
ggsave("../raw_fig/FigureS7_dhs_point_agg.pdf", point_agg, "pdf", width=5, height=5, dpi=300)


################################################################
# make LSMS delta of indexes plot
################################################################

lsms = read.csv("../data/predictions/lsms_delta_preds_incountry.csv")

cross = data.frame(country=c("ethiopia", "malawi", "nigeria", "tanzania", "uganda"),
                   iso3 = c("ETH", "MWI", "NGA", "TZA", "UGA"))
lsms = merge(lsms, cross, by = "country")
lsms = data_to_geolev(lsms, shapefile_loc="../data/shapefiles/gadm/")
lsms_agg = lsms %>% dplyr::group_by(geolev, year1, year2, country) %>% 
    dplyr::summarize(label = mean(label, na.rm=T), preds_ms = mean(preds_ms, na.rm=T), 
                     preds_nl = mean(preds_nl, na.rm=T), 
                     preds_msnl_concat = mean(preds_msnl_concat, na.rm=T), n = n())

r2s = c(cor(lsms$preds_msnl_concat, lsms$label)^2, cor(lsms$preds_nl, lsms$label)^2, cor(lsms$preds_ms, lsms$label)^2)
r2s = data.frame(r2=r2s, type=c("msnl", "nl", "ms"))

r2s_agg = c(cor(lsms_agg$preds_msnl_concat, lsms_agg$label)^2, cor(lsms_agg$preds_nl, lsms_agg$label)^2, cor(lsms_agg$preds_ms, lsms_agg$label)^2)
r2s_agg = data.frame(r2=r2s_agg, type=c("msnl", "nl", "ms"))

rug = ggplot() + geom_rug(aes(x=r2, color=type), r2s, size=1) + 
    theme_anne(font="sans") + xlim(0, 0.43)

rug_agg = ggplot() + geom_rug(aes(x=r2, color=type), r2s_agg, size=1) + 
    theme_anne(font="sans") + xlim(0, 0.43)

point = panel(lsms[, c("preds_ms", "label")], a=0.07, font="sans", size=14, square=T) + 
    ylab("LSMS difference of indexes") +
    xlab("MS Predicted difference of indexes") 

point_agg = panel(lsms_agg[, c("preds_ms", "label")], a=0.12, font="sans", size=14, w=lsms_agg$n, square=T) + 
    ylab("LSMS difference of indexes") +
    xlab("MS Predicted difference of indexes") 

ggsave("../raw_fig/FigureS7_diff_rug.pdf", rug, "pdf", width=5, height=4, dpi=300)
ggsave("../raw_fig/FigureS7_diff_rug_agg.pdf", rug_agg, "pdf", width=5, height=4, dpi=300)
ggsave("../raw_fig/FigureS7_diff_point.pdf", point, "pdf", width=5, height=5, dpi=300)
ggsave("../raw_fig/FigureS7_diff_point_agg.pdf", point_agg, "pdf", width=5, height=5, dpi=300)
