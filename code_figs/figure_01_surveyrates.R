source("00_dependencies.R")

######################################################
# Create DF with % pop surveyed for each year in the world
######################################################

#get country by year # of people surveyed in povcal listed SURVEY data
povcal = read.csv("../data/surveys/povcal_time_pop.csv")
povcal = melt(povcal, id="country")
names(povcal) = c("country", "year", "povcal")
povcal[is.na(povcal$povcal) | povcal$povcal == "", "povcal"] = 0
#povcal$povcal = as.numeric(povcal$povcal)

#get country by year # of people surveyed in DHS SURVEY data
dhs = read.csv("../data/surveys/dhs_time.csv", stringsAsFactors = F)
dhs = melt(dhs)
names(dhs) = c("country", "year", "dhs")

#get country by year population
pop = read.csv("../data/surveys/population_time.csv", stringsAsFactors = F)
pop = melt(pop)
names(pop) = c("country", "iso3", "year", "pop")
pop = dplyr::select(pop, -country)

#crosswalk country ids
cross = read.csv("../data/crosswalks/crosswalk_countries.csv", stringsAsFactors = F)

#merge povcal, population, and crosswalk country names
dhs = merge(dhs, cross, by.y="country_simp", by.x="country", all.x=T)[, -6]
povcal = merge(povcal, cross, by.y="country_simp", by.x="country", all.x=T)[, -6]
full = merge(dhs, povcal, by=c("country", "year", "iso2", "iso3", "country_pred", "country_wb"), all=T)
full = merge(full, pop, by=c("iso3", "year"), all.x=T)
full = dplyr::select(full, year, country, iso2, dhs, pop, povcal)

#sub in the years of povcal without  sample sizes
samp = as.numeric(full$povcal)/full$pop
samp = quantile(samp[samp != 0], 0.9, na.rm=T)
full$povcal[full$povcal == "x" & !is.na(full$povcal)] = samp*full$pop[full$povcal == "x" & !is.na(full$povcal)]
full$povcal = as.numeric(full$povcal)

#get year in better format
full$year = full$year %>% as.character() %>% substr(2, 6) %>% as.numeric() 

#get the US and filter the rest to relevant countries
full = full[full$country %in% 
                c("Algeria", "Angola",  "Benin", "Botswana", "Burkina Faso", "Burundi",
                  "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros", 
                  "Democratic Republic of the Congo", "Djibouti", "Egypt", 
                  "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon", "Gambia", "Ghana", 
                  "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", 
                  "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", 
                  "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", 
                  "Republic of the Congo", "Rwanda", "Sao Tome and Principe", "Senegal", 
                  "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", 
                  "Sudan", "Swaziland", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia",  
                  "Zimbabwe"), ]
                        
#get the aggregated yearly revisit rate over all the countries in full
year = aggregate(full[, c('dhs', 'pop', 'povcal')], by=list(full$year), sum, na.rm=TRUE)
names(year) = c("year", 'dhs', 'pop', 'povcal')
year$surv_perc = (year$pop*365)/(year$dhs + year$povcal)

#combine the us dhs and povcal with ACS etc
us_pop = pop[pop$iso3=="USA",]
us = read.csv('../data/surveys/us_surveys_time.csv')
us = melt(us)
us = aggregate(us$value, by=list(us$variable), FUN=sum, na.rm=T)
us$year = us$Group.1 %>% as.character() %>% substr(2, 6) %>% as.numeric() 
us = merge(us, us_pop, by.x="Group.1", by.y="year")
us$survey_perc = (us$pop*365)/(us$x)

######################################################
# Create DF with % pop imaged for each year in the world
######################################################

gee = read.csv("../data/overpass/dhs_sample_GEE.csv")
gee = gee %>% 
    group_by(year) %>% #multiply modis *8 because its an 8 day composite
    dplyr::summarize(l5 = 365/(sum(num_l5, na.rm=T)/500), 
                     l7 = 365/(sum(num_l7, na.rm=T)/500), 
                     l8 = 365/(sum(num_l8, na.rm=T)/500),
                     s2 = 365/(sum(num_s2, na.rm=T)/500), 
                     s1 = 365/(sum(num_s1, na.rm=T)/500), 
                     all_s = 365/(sum(num_s1, num_s2, na.rm=T)/500),
                     all_l = 365/(sum(num_l5, num_l8, num_l7, na.rm=T)/500), 
                     modis = 365/((sum(num_modis, na.rm=T)*8)/500))
                         
planet = read.csv("../data/overpass/dhs_sample_Planet.csv")
planet = dplyr::select(planet, year, cluster_id, count_PlanetScope, count_RapidEye)
planet = planet %>% 
    group_by(year) %>% 
    dplyr::summarize(planetscope = 365/(sum(count_PlanetScope, na.rm=T)/500), 
              rapideye = 365/(sum(count_RapidEye, na.rm=T)/500), 
              all_planet = 365/(sum(count_RapidEye, count_PlanetScope, na.rm=T)/500))

quickbird = read.csv("../data/overpass/landinfo_dhs_sample_nocatalog.csv")
quickbird$year = paste0("20", substr(as.character(quickbird$date),
                                     nchar(as.character(quickbird$date))-1, 
                                     nchar(as.character(quickbird$date))))
quickbird$cloud = substr(as.character(quickbird$cloud), 1, nchar(as.character(quickbird$cloud))-1)
quickbird = quickbird[quickbird$cloud <= 30 & quickbird$off.nadir <= 20 &
                          quickbird$sensor %in% c("WorldView-1", "WorldView-2", "WorldView-3", 
                                                  "WorldView-4", "GeoEye-1", "QuickBird-2", "IKONOS"), ]
quickbird =  dcast(quickbird, year ~ sensor)
quickbird$dg = 365/(rowSums(quickbird[, 2:ncol(quickbird)])/500)
quickbird = dplyr::select(quickbird, year, dg)

overpass = merge(gee, planet, by="year")
overpass = merge(overpass, quickbird, by="year")
overpass[overpass==Inf] = NA
overpass = melt(overpass, id.vars="year")

overpass = overpass[overpass$variable %in% c("s2", "all_l", "planetscope", "dg", "rapideye"),]

resolution = data.frame(variable=c("s2", "all_l", "planetscope", "dg", "rapideye"), 
                        res = c(10, 30, 3, .6, 5))
overpass = merge(overpass, resolution, by="variable")

######################################################
# Make line plot of revisit rate over time
######################################################

reverselog_trans = function(base = 10) {
    trans <- function(x) -log(x, base)
    inv <- function(x) base^(-x)
    trans_new(paste0("reverselog-", format(base)), trans, inv, 
              log_breaks(base = base), 
              domain = c(1e-100, Inf))
}

year$variable = "africa_surveys"
year = year[, c("year", "variable", "surv_perc")]
names(year) = c("year", "variable", "value")
us$variable = "us_surveys"
us = us[, c("year", "variable", "survey_perc")]
names(us) = c("year", "variable", "value")

year = rbind(us, year)
year = year[year$year <= 2016,]
options(scipen=100)

p = ggplot() + 
    geom_hline(yintercept=1, size=0.7, alpha=0.5, color="grey") + 
    geom_hline(yintercept=7,size=0.7, alpha=0.5, color="grey") + 
    geom_hline(yintercept=30, size=0.7, alpha=0.5, color="grey") +
    geom_hline(yintercept=365, size=0.7, alpha=0.5, color="grey") +
    geom_hline(yintercept=3650, size=0.7, alpha=0.5, color="grey") +
    geom_hline(yintercept=36500, size=0.7, alpha=0.5, color="grey") +
    geom_hline(yintercept=365000, size=0.7, alpha=0.5, color="grey") +
    geom_hline(yintercept=4927500, size=0.7, alpha=0.5, color="grey") +
    
    geom_line(aes(year, value, group=variable, color=as.factor(res)), size=0.8, overpass) + #, color=variable
    geom_line(aes(year, value, group=variable), size=0.8, linetype="1232", year) +
    
    scale_y_continuous(trans=reverselog_trans(10), limits = c(4927500, 0.5),
                       breaks = c(1, 7, 30, 100, 365, 3650, 10000, 36500, 365000, 1000000, 4927500), 
                       labels = c(1, "", "", 100, "", "",   10000, "",    "",     1000000, "")) + #13500 years
    scale_x_continuous(limits = c(2000, 2018), breaks=seq(2000, 2018, 2)) + 
    
    ylab("Avg. household revisit interval (days)") + xlab("Year") + theme_anne("sans", size=25) +
    scale_colour_manual(values = colorRampPalette(c("#06276E", "#9BC7FF"), bias=2)(5)) + theme(legend.position = "none")

ggsave("../raw_fig/Figure1b_revisitrate.pdf", plot=p,  width=11.5, height=7.4)


######################################################
# Create DF with % pop surveyed for each country in the world
######################################################

country = full[full$year <= 2014 & full$year >= 2000, ]
country = aggregate(country[, c('dhs', 'pop', 'povcal')], by=list(country$country), sum, na.rm=TRUE)
names(country) = c("country", 'dhs', 'pop', 'povcal')
country$perc = (country$dhs + country$povcal)/(country$pop*(365))
country = merge(country, cross, by.x="country", by.y="country_simp")

cols = 2:18

povcal = read.csv("../data/surveys/povcal_time_pop.csv")
povcal[povcal==""] = 0
povcal[, -1] = !is.na(povcal[, -1])
povcal$povsum = rowSums(povcal[, cols], na.rm=T)

dhs = read.csv("../data/surveys/dhs_time.csv")
dhs[, -1] = dhs[, -1] > 0
dhs$dhssum = rowSums(dhs[, cols], na.rm=T)

census = merge(dhs, povcal, by='country', all=T)
census = merge(census, cross, by.x="country", by.y="country_simp")
census = dplyr::select(census, country, iso3, povsum, dhssum)

census[is.na(census)] = 0
survt = (census$povsum + census$dhssum)!=0
census[survt, "survsum"] = (2018-1999)/(census[survt, "dhssum"] + census[survt, "povsum"])
census[!survt, "survsum"] = 0
africa = readRDS("../data/shapefiles/africa_gadm.rds")
africa = merge(africa, census, by.x ="GID_0", by.y="iso3")
africa$survsum[is.na(africa$survsum)] = 0

######################################################
# Make maps
######################################################

surv = map(africa, 'survsum', '', 'years btwn surveys', color=c("#F7F7B8", "#F7EB7D", "#F7E04C", "#F7CA23", "#FFA94D", "#FF4D4D"), #mincol="#FFFFA7", maxcol="#FF0000", 
           breaks=c(1, 2, 3, 4, 5, 10, 19), font="sans")
ggsave("../raw_fig/Figure1a_mapsurveysovertime.pdf", plot=surv,  width=7, height=7)
