packages = c("RColorBrewer", "SearchTrees", "broom", "class", "colorspace", "data.table", 
             "extrafont", "ggplot2", "ggthemes", "grDevices", "graphics", "grid", 
             "gridExtra", "jcolors", "lazyeval", "lfe", "plyr", "png", "raster", 
             "rasterVis", "reshape2", "rgdal", "rgeos", "scales", "sf", "sp", "splines", 
             "tidyr", "weights", "dplyr")
packages = packages[which(!packages %in% rownames(installed.packages()))]
if(length(packages)>0){install.packages(packages)}

library(RColorBrewer, quietly=T)
library(SearchTrees, quietly=T)
library(broom, quietly=T)
library(class, quietly=T)
library(colorspace, quietly=T)
library(data.table, quietly=T)
library(extrafont, quietly=T)
library(ggplot2, quietly=T)
library(ggthemes, quietly=T)
library(grDevices, quietly=T)
library(graphics, quietly=T)
library(grid, quietly=T)
library(gridExtra, quietly=T)
library(jcolors, quietly=T)
library(lazyeval, quietly=T)
library(lfe, quietly=T)
library(plyr, quietly=T)
library(png, quietly=T)
library(raster, quietly=T)
library(rasterVis, quietly=T)
library(reshape2, quietly=T)
library(rgdal, quietly=T)
library(rgeos, quietly=T)
library(scales, quietly=T)
library(sf, quietly=T)
library(sp, quietly=T)
library(splines, quietly=T)
library(tidyr, quietly=T)
library(weights, quietly=T)
library(dplyr, quietly=T)

theme_anne = function(font="sans", size=10) {
    theme_tufte(base_size=size, base_family=font) %+replace% 
        theme(
            panel.background  = element_blank(),
            plot.background = element_rect(fill="transparent", colour=NA), 
            axis.line.x = element_line(color="black", size = .2), 
            axis.line.y = element_line(color="black", size = .2), 
            plot.title = element_text(hjust = 0.5)
        )
}


map = function(data, col, title, legend_title, black=0, color=NULL,
               breaks=find_breaks(data@data[, col], 7), mincol="#2D35E4", maxcol="#E4302D", 
               font='sans', size=10, simp=0.05, na.color="#696969") {
    
    data_df = broom::tidy(rgeos::gSimplify(data, simp))
    data$polyID = sapply(slot(data, "polygons"), function(x) slot(x, "ID"))
    if (length(unique(data$polyID)) != length(unique(data_df$id))) {
        stop("gSimplify has removed some of the polygons entirely, meaning matching will fail. Enter a lower simp value.")
    }
    data_df = merge(data_df, data, by.x = "id", by.y="polyID")
    
    if (is.null(color)) {color = scales::seq_gradient_pal(mincol, maxcol, "Lab")(seq(0,1,length.out=length(breaks)-1))}
    
    gg = ggplot() + aes(x = long, y = lat, group = group) +
        geom_polygon(data = data_df, aes_string(fill = cut(data_df[,col], breaks))) +
        geom_polygon(data = data_df[data_df$country=="Lesotho",], 
                     aes_string(fill = cut(data_df[data_df$country=="Lesotho",col], breaks))) +
        geom_path(data = data_df, color = "white", size = 0.08) +
        coord_equal() + 
        ggtitle(title) + 
        labs(fill = legend_title) +
        theme_anne(font, size=size) + theme(line = element_blank(),                          
                                            axis.text=element_blank(),                       
                                            axis.title=element_blank(),                      
                                            panel.background = element_blank(),
                                            text=element_text(size=size,  family=font)) +
        scale_fill_manual(values=color, drop=F, na.value=na.color)
    
    gg = gg + geom_polygon(data = data_df[data_df[, col] == black, ], fill="black") +
        geom_path(data = data_df, color = "white", size = 0.1)
    
    gg
}


find_common_bounds = function(x, y, square) {
    if (square) {
        all = c(x, y)
        ret = c(min(all, na.rm=T), max(all, na.rm=T), min(all, na.rm=T), max(all, na.rm=T))
    } else {
        ret = c(min(x, na.rm=T), max(x, na.rm=T), min(y, na.rm=T), max(y, na.rm=T))
    }
    return(ret)
}


panel = function(data, xbounds=NULL, ybounds=NULL, laby=NULL, labx=NULL, annotate=T, 
                 annotation=NULL, betal=F, lm=T, a=0.2, w=NULL, square=F, font="sans", 
                 size=10, dot_size=T) {
    
    # save names for axes
    labels = names(data)
    names(data) = c("x", "y")
    
    # define bounds and annotation location
    if (is.null(xbounds)) {xbounds=data$x}
    if (is.null(ybounds)) {ybounds=data$y}
    bounds = find_common_bounds(xbounds, ybounds, square)
    minx = bounds[1]
    maxx = bounds[2]
    miny = bounds[3]
    maxy = bounds[4]
    if (is.null(labx)) {labx=minx} 
    if (is.null(laby)) {laby=maxy} 
    
    # plot!
    plot = ggplot(data, aes(x, y)) + xlab(labels[1]) + ylab(labels[2]) + 
        xlim(minx, maxx) + ylim(miny, maxy) + theme_anne(font=font, size=size)
    if (!is.null(w) & dot_size) {plot = plot+geom_point(aes(size=w), alpha = a)+guides(size=FALSE)
    } else {plot = plot+geom_point(alpha = a)}
    
    if (lm) {
        plot = plot + geom_smooth(method="lm", se=FALSE, alpha=0.4, size=.5)
        if (annotate) {
            # model data to get r2 and Beta for annotation
            if (!is.null(w)) {model = lm(y~x, weights=w, data=data)
            } else {model = lm(y~x, data=data)}
            r2 = format(round(summary(model)$r.squared, 2), nsmall = 2)
            beta = round(summary(model)$coefficients[2], 2)
            
            if (!is.null(annotation) & betal) {lab = paste(annotation, "\nr^2 =='", r2, "'\nBeta ==", beta)}
            else if (!is.null(annotation) & !betal) {lab = paste(annotation, "\nr^2 =='", r2, "'")}
            else if (betal) {lab = paste("r^2 =='", r2, "'\nBeta ==", beta)}
            else if (!betal) {lab = paste("r^2 =='", r2, "'")}
            
            #add annotation to plot
            plot = plot + annotate("text", x=labx, y=laby, label=lab, color="grey46", 
                                   hjust=0, vjust=1, size=size/2.5, family=font, parse=T)
        } 
    }
    
    if (!lm & annotate) {
        if(!is.null(w)) {
            r2 = round(wtd.cor(data$x, data$y, w=w)[1]^2, 2)
        } else {
            r2 = round(cor(data$x, data$y, use ="c")^2, 2)
        }
        lab = paste("r^2 == ", r2)
        if (!is.null(annotation)) {lab = paste(annotation, "\nr^2 == ", r2)}
        plot = plot + annotate("text", x=labx, y=laby, label=lab, color="grey46", 
                               hjust=0, vjust=1, size=size/2.5, family=font, parse=T)
    }
    
    print(lab)
    return(plot)
}


arrange_panels = function(plots, n, save_path, name=NULL, font="sans") {
    
    if(!is.null(save_path)) {
        w = ceiling(length(plots)/n)*4
        h = n*4
        ggsave(save_path, do.call("grid.arrange", c(plots, nrow=n)), 
               width=w, height=h, units="in", dpi=600)
        return(do.call("grid.arrange", c(plots, nrow=n)))
    }
    else {
        temp = do.call("grid.arrange", c(plots, nrow=n))
        x = grid.arrange(temp, top=textGrob(name,gp=gpar(fontsize=17,fontfamily=font)))
        return(x)
    }
}


panel_set = function(data, name, subtitles=NULL, n=1, w=NULL, save_path=NULL, font="sans", 
                     lm=rep(T, length(data)), annotate=T, betal=F, a=0.2, square=F, 
                     size=10, unif_bounds=T, dot_size=T) {
    plots = list(rep(NA, length(data)))
    
    xbounds = unlist(sapply(data, function(x) x[, 1]))
    xbounds = c(min(xbounds, na.rm=T)-.1, max(xbounds, na.rm=T)+.1)
    ybounds = unlist(sapply(data, function(x) x[, 2]))
    ybounds = c(min(ybounds, na.rm=T)-.1, max(ybounds, na.rm=T)+.1)
    
    #for each set of x/y passed in 
    for (i in 1:length(data)) {
        #data[[i]] needs to be a df with two columns, named as they should be 
        cur = data[[i]]
        weights = w[[i]]
        if (!unif_bounds) {
            xbounds = c(min(cur[, 1], na.rm=T)-abs(min(cur[, 1], na.rm=T)*.1), 
                        max(cur[, 1], na.rm=T)+abs(min(cur[, 1], na.rm=T)*.1))
            ybounds = c(min(cur[, 2], na.rm=T)-abs(min(cur[, 2], na.rm=T)*.1),
                        max(cur[, 2], na.rm=T)+abs(min(cur[, 2], na.rm=T)*.1))
        }
        gg = panel(cur, xbounds, ybounds, w=weights, font=font, lm=lm[i], size=size,
                   annotate=annotate, betal=betal, a=a, square=square, dot_size=dot_size) + ggtitle(subtitles[i])
        plots[[i]] = gg
    }
    
    arrange_panels(plots, n, save_path, name, font)
}


data_to_geolev = function(data, shapefile_loc, lon='lon', lat='lat', complete=F, 
                          country='iso3', geolev='2', gadm=T) {
    
    #save the original columns
    columns = names(data)
    
    if ( !(lon %in% columns) || !(lat %in% columns) ) {
        stop("Data input must contain lat and lon columns as specified.")
    }
    
    #turn data in to spatial points
    xy = data[, c(lon, lat)]
    utm = "+proj=utm +zone=32 +ellps=intl +towgs84=-87,-98,-121,0,0,0,0 +units=km +no_defs"
    wgs_84 = "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0 +units=m"
    data = SpatialPointsDataFrame(coords=xy, data = data, proj4string = CRS(wgs_84)) #data is in wgs84
    data = spTransform(data, CRS(utm)) #need it in utm for distance calculation
    data$geolev = NA
    
    #get countries to loop through
    countries = as.character(unique(data@data[,country]))
    
    for (c in countries) {
        c = as.character(c)
        #open the file specific to the country, get subset of data
        shapefile = read_shapefile(shapefile_loc, c, geolev, gadm)
        if (is.null(shapefile)) {
            warning(sprintf("Country %s shapefile not found. Skipping and continuing.", c))
            next
        }
        shapefile_geolev_col = shapefile[[2]]
        shapefile = shapefile[[1]]
        shapefile = spTransform(shapefile, CRS(utm))
        t = data[as.vector(data@data[,country]==c),]
        
        #find distance to polygons from each point, find smallest distance, save
        dists = gDistance(t, shapefile, byid=T)
        dists = apply(dists, 2, which.min)
        geolocated = as.character(shapefile@data[dists, shapefile_geolev_col])
        
        #save to original data file
        data[data@data[,country]==c,'geolev'] = geolocated
    }
    
    if (complete) {
        #meaning only return the rows that mapped to places
        #defaults to not this because FEFO
        data = data[!is.na(data@data$geolev), ]
    }
    
    data = data@data
    
    return(data)
}


read_shapefile = function(shapefile_loc, country, geolev, gadm) {
    
    if (gadm) {
        folder = paste0(shapefile_loc, "gadm36_", country, "_shp")
        files = list.files(folder)
        file = paste0("gadm36_", country, "_", geolev)
        shapefile_geolev_col = paste0("GID_", geolev)
    } else {
        folder = paste0(shapefile_loc, "geo", geolev, "_", country, "")
        files = list.files(folder)
        file = paste0("geo", geolev, "_", country)
        shapefile_geolev_col = paste0("GEOLEVEL", geolev)
    }
    
    if (paste0(file, ".shp") %in% files) {
        shapefile = rgdal::readOGR(folder, file)
        if (gadm) {
            shapefile_geolev_col = paste0("GID_", geolev)
        } else {
            shapefile_geolev_col = paste0("GEOLEVEL", geolev)
        }
    } else {
        file = gsub(geolev, "1", file)
        if (paste0(file, ".shp") %in% files) {
            shapefile = rgdal::readOGR(folder, gsub(geolev, "1", file))
            shapefile_geolev_col = gsub(geolev, "1", shapefile_geolev_col)
        } else {
            return(NULL)
        }
    }
    
    return(list(shapefile, shapefile_geolev_col))
}


stata_index = function(data_index, train_index=data_index, show_loadings=F) {
    
    correlations = cor(train_index)
    e = eigen(correlations)
    loadings = (e$vectors[, 1] * e$values[1])
    loadings = loadings/sqrt(e$values[1])
    loadings = -loadings
    
    if (show_loadings){
        print(loadings)
    }
    
    index = scores(data_index, loadings)
    
    return(index)
}


scores = function(data, loadings) {
    z = as.matrix(data)
    zz = scale(z, T, T)
    cv = cov(data) #covmat$cov
    sds = sqrt(diag(cv))
    cv = cv/(sds %o% sds)
    sc =  zz %*% solve(cv, loadings)
    return(sc)
}
