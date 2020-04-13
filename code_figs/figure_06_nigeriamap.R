source("00_dependencies.R")

xmin = 596700
xmax = xmin+(7650*20)
ymin = 474300
ymax = ymin+(7650*13)
w = ((xmax-xmin)/7650)
h = ((ymax-ymin)/7650)
bounding = data.frame(x = c(xmin, xmax, xmax, xmin, xmin), y = c(ymax, ymax, ymin, ymin, ymax))

################################################################
#read in data needed
################################################################
country = readOGR("../data/shapefiles/gadm/gadm36_NGA_shp", "gadm36_NGA_0")
c = crs(country)
r = raster(paste0("../data/predictions/nigeria_2012_2014_msnl_vals.tif"))
country = spTransform(country, projection(r))
country_df = tidy(gSimplify(country, .005))
bounding_points = SpatialPointsDataFrame(bounding, data=bounding)
crs(bounding_points) = crs(r)
pal = colorRampPalette(rev(brewer.pal(10,"RdYlBu")))
preds = read.csv("../data/predictions/dhs_preds_incountry.csv")
preds = preds[preds$country=="nigeria",]
preds$wealthpooled = preds$labels
preds[preds$wealthpooled>1.3, "wealthpooled"] = 1.3
preds[preds$wealthpooled<-1, "wealthpooled"] = -1
country = spTransform(country, projection(r))
country_df = tidy(gSimplify(country, .005))
preds = SpatialPointsDataFrame(preds[,c("lat", "lon")], data=as.data.frame(preds[,c("wealthpooled")]))
crs(preds) = c
preds = spTransform(preds, projection(r))
preds = cbind(preds@coords, preds@data)
names(preds) =c("lat", "lon", "wealthpooled")

################################################################
#make the raster figs for ms, msnl, and nl
################################################################
#loop through each type and create the image for it 
for (mode in c("nl", "ms", "msnl")) {
    r = raster(paste0("../data/predictions/nigeria_2012_2014_", mode, "_vals.tif"))
    r[r>1.3] = 1.3
    pdf(paste0("../raw_fig/Figure6_predictions", mode, ".pdf"), width=w, height=h)
    gplot(r) + geom_tile(aes(fill=value)) + 
        scale_fill_gradientn(colors=pal(11), limits=c(-1, 1.3), na.value="transparent") + 
        theme_blank() + scale_x_continuous(limits = c(xmin, xmax), expand = c(0, 0)) + 
        scale_y_continuous(limits = c(ymin, ymax), expand = c(0, 0))  
    dev.off()
    if (mode=="msnl"){
        pdf(paste0("../raw_fig/Figure6_predictions", mode, "_outline.pdf"), width=176, height=143)
        gplot(r) + geom_tile(aes(fill=value)) + 
            scale_fill_gradientn(colors=pal(11), limits=c(-1, 1.3), na.value="transparent") + 
            geom_path(aes(x=long, y=lat, group=group), data = country_df, color = "black", size = 2) +
            theme_blank() + geom_path(aes(x, y), data = bounding, size=3)
        dev.off()
    }
}

################################################################
#make the scatterplot of groundtruths from dhs
################################################################
pdf("../raw_fig/Figure6_nigeriagt.pdf", width=w, height=h)
ggplot() + 
    geom_path(aes(x=long, y=lat, group=group), data = country_df, color = "black", size = 1.5) +
    geom_point(aes(lon, lat, color=wealthpooled), data=preds, size=35, alpha=0.9) + 
    theme_blank() + scale_color_gradientn(colors=pal(11), limits=c(-1, 1.3)) +
    scale_x_continuous(limits = c(xmin, xmax), expand = c(0, 0)) + scale_y_continuous(limits = c(ymin, ymax), expand = c(0, 0))
dev.off()

################################################################
# saving raw nl image
################################################################
nl = raster(paste0("../data/predictions/nigeria_nl_5_8_4_6_polygon.tif"))
nl = log(nl)
nl[nl>2.5] = 2.5
pdf("../raw_fig/Figure6_nigerianl.pdf", width=w, height=h)
ggplot(nl) + geom_tile(aes(fill=value)) + theme_blank() +
    scale_fill_gradient(low = "black", high = "white", limits=c(-3, 2.5), na.value = "transparent") +
    scale_x_continuous(limits = c(xmin, xmax), expand = c(0, 0)) + scale_y_continuous(limits = c(ymin, ymax), expand = c(0, 0))
dev.off()


################################################################ 
#make the population weighted aggregated estimates
################################################################

#convert projections and deal w rasters
p = raster(paste0("../data/predictions/nigeria_pop.tif"))
r = raster(paste0("../data/predictions/nigeria_2012_2014_msnl_vals.tif"))
p_w = p*r #weight by population
shape = readOGR("../data/shapefiles/gadm/gadm36_NGA_shp", "gadm36_NGA_2")
shape = spTransform(shape, projection(p))
weighted_wealth = raster::extract(p_w, shape) #get weighted predictions by district
weighted_wealth = lapply(weighted_wealth, FUN=sum)
population = raster::extract(p, shape)
population = lapply(population, FUN=sum)
wealth = unlist(weighted_wealth)/unlist(population)

#get shapefiles in the right format
shape@data$wealth = wealth
shape@data$wealth_tile = ntile(shape@data$wealth, 10)
shape_df = tidy(gSimplify(shape, .005))
shape$polyID = sapply(slot(shape, "polygons"), function(x) slot(x, "ID"))
shape_df = merge(shape_df, shape, by.x = "id", by.y="polyID")
country = spTransform(country, projection(r))
country_df = tidy(gSimplify(country, .005))

#save fig
pdf("../raw_fig/Figure6_predictionsmsnlagg.pdf", width=176, height=143)
ggplot() + aes(x=long, y=lat, group=group) +
    geom_path(color = "black", size = 1, data = shape_df) +
    geom_polygon(aes(fill=wealth_tile), data = shape_df) +
    geom_path(color = "black", size = 2, data = country_df) +
    scale_fill_gradientn(colors=rev(brewer.pal(10,"RdYlBu")), limits=c(0, 10)) + theme_blank()
dev.off()

################################################################
# saving legend
################################################################
#village level derived indices from aggregated DHS data, no households identifiable
preds = read.csv("../data/predictions/dhs_preds_incountry.csv")
preds = preds[preds$country=="nigeria",]
preds = as.data.frame(preds)
preds[preds$label > 1.3, "label"] = 1.3
preds[preds$label < -1, "label"] = -1
g = ggplot() + 
    geom_point(aes(lon, lat, color=label), data=preds, size=3.5) + 
    scale_color_gradientn(guide="legend", colors=pal(11), limits=c(-1, 1.3)) +
    theme(legend.position="bottom") + 
    guides(color=guide_colourbar(barwidth=10, barheight=.8, ticks=F, nbin=100, title=NULL))

legend = census.tools::get_legend(g) 
png(filename="../raw_fig/Figure6_legend.png", width=2000, height=800, res=600)
grid.newpage()
grid.draw(legend) 
dev.off() 