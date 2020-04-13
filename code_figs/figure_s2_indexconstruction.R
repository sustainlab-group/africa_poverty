source("00_dependencies.R")


###################################################################
# Correlation of various asset ownership things (DHS DATA)
###################################################################   

dhs = readRDS("../data/DHS_output_2020-04-09_aggregatedforidentification.RDS")

range01 = function(x){(x-min(x))/(max(x)-min(x))}
index_vars = c("roomspp", "electric", "phone", "radio", "tv", "car", "fridge",
               "motorcycle", "floor_qual", "toilet_qual", "water_qual")

i = dhs
i$roomspp = range01(i$roomspp)
i$floor_qual = range01(i$floor_qual)
i$toilet_qual = range01(i$toilet_qual)
i$water_qual = range01(i$water_qual)
dhs$sum_index = rowSums(i[, index_vars])

dhs$country_index =  NA
dhs$country_year_index =  NA
for (c in unique(dhs$cname)) {
    dhs$country_index[dhs$cname == c] = stata_index(dhs[dhs$cname == c, index_vars])
    for (y in unique(dhs[dhs$cname == c,]$year)) {
        dhs$country_year_index[dhs$cname==c & dhs$year==y] = 
            stata_index(dhs[dhs$cname==c & dhs$year==y, index_vars])
    }
}

flip = dhs$cname=="MW"&dhs$year==2010
dhs[flip, ]$country_year_index = -dhs[flip, ]$country_year_index
dhs$object_index = stata_index(dhs[, c("electric", "phone", "radio", "tv", "car")])

dhs = dplyr::select(dhs, cname, year, cluster, index, object_index, country_year_index, 
                    sum_index, n)
names(dhs) = c("cname", "year", "cluster", "Index", 
               "Index constructed only using object ownership variables", 
               "Index constructed within country-years", 
               "Index constructed by summing assets owned, no PCA", "n")

panel_set(list(dhs[, c("Index", "Index constructed by summing assets owned, no PCA")], 
               dhs[, c("Index", "Index constructed only using object ownership variables")], 
               dhs[, c("Index", "Index constructed within country-years")]), 
          name="", unif_bounds=F, font="sans",
          subtitles=c("Raw count of assets owned",
                      "Only using objects - tv, radio, etc", 
                      "Index created within each country-year"), 
          save_path="../processed_fig/FigureS2.pdf")
