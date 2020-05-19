source("00_dependencies.R")


indexdelta = read.csv("../data/output/lsms_labels_index_of_diffs.csv")
#indexdelta = filter(indexdelta, year.x==2005 & year.y==2009)
indexdelta = dplyr::select(indexdelta, lat, lon, index_diff, index, x, year.x, year.y)
indexdelta$lat = round(indexdelta$lat, 3)
indexdelta$lon = round(indexdelta$lon, 3)

lsms = read.csv("../data/predictions/lsms_nls.csv")
lsms = merge(lsms, lsms, by = c("lat", "lon", "country"))
lsms = dplyr::filter(lsms, year.x < year.y & year.y-year.x < 7)
lsms$nl_mean = lsms$nls_mean.y - lsms$nls_mean.x
#lsms = lsms[lsms$nl_mean > -7, ] #remove 2 outliers to make graph clearer
lsms = dplyr::select(lsms, lat, lon, nl_mean, year.x, year.y)
lsms$lat = round(lsms$lat, 3)
lsms$lon = round(lsms$lon, 3)

deltas = merge(lsms, indexdelta, by=c("lat", "lon", "year.x", "year.y"))
names(deltas) = c("lat", "lon", "year.x", "year.y",
                  "Change in mean NL", "Index of changes", "Change of index", "n")

lsms = read.csv("../data/predictions/lsms_nls.csv")
lsms = dplyr::filter(lsms, year == 2005 & country=="uganda")
lsms$lat = round(lsms$lat, 4)
lsms$lon = round(lsms$lon, 4)
labels = read.csv("../data/output/lsms_labels_index_agg.csv")
labels = dplyr::filter(labels, year == 2005 & country=="ug")
labels$lat = round(labels$lat, 4)
labels$lon = round(labels$lon, 4)
lsms = merge(lsms, labels, by=c("lat", "lon"))
lsms = dplyr::select(lsms, index, nls_mean, n)
names(lsms) = c("Index", "Mean NL", "n")

panel_set(list(lsms[, 1:2], deltas[, c(7, 5)], deltas[, c(6, 5)]), 
          w=list(lsms$n, deltas$n, deltas$n), lm=rep(F, 3), unif_bounds=F, 
          name="", font="sans", dot_size=F, save_path="../processed_fig/FigureS8.pdf")
