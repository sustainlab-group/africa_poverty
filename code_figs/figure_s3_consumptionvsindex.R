source("00_dependencies.R")

###################################################################
# Scatterplot of Consumption ~ Assets
###################################################################   

full = read.csv("../data/output/lsms_labels_index.csv")
full = full[!is.na(full$total_cons_ann), ]

wdi = read.csv("../data/wdi.csv", na.strings = "..", stringsAsFactors=F)
wdi = wdi[wdi$Series.Name == "PPP conversion factor, GDP (LCU per international $)" &
              wdi$Country.Name %in% c("Ethiopia", "Nigeria", "Tanzania"), ]
wdi = gather(wdi, year, val, `X2000..YR2000.`:`X2019..YR2019.`, factor_key=TRUE)[, c(1, 5:6)]
wdi$year = substr(wdi$year, 2, 5)
wdi$Country.Name = tolower(wdi$Country.Name)
wdi$country_year = paste0(wdi$Country.Name,  "_", wdi$year)

conv = data.frame(country = c("et", "ng", "tz"), country_conv = c("ethiopia", "nigeria", "tanzania"))
t =  merge(full, conv, by="country")
t$country_year =  paste0(t$country_conv, "_", t$year)
t = merge(t, wdi[, c("country_year", "val")], by="country_year", all.x=T)
t$consumption = (t$total_cons_ann/t$val)/365
t = t %>% dplyr::group_by(country, year, country_year, ea_id) %>% 
    dplyr::summarize(consumption=mean(consumption), index=mean(index), n=n())
t$`log(consumption)` = log(t$consumption)

pdf("../processed_fig/FigureS3.pdf", width=6, height=6)
panel(t[, c("index", "log(consumption)")], font="sans", size=16, w=t$n)
dev.off()