source("00_dependencies.R")

###################################################################
# Correlation of R2 and World Dev Indicators
###################################################################  

#read in dhs data and convert codes to quality codes
dhs = readRDS("../data/DHS_output_2019-12-18_aggregatedforidentification.RDS")
range01 = function(x){(x-min(x))/(max(x)-min(x))}
index_vars = c("rooms", "electric", "phone", "radio", "tv", "car", "floor_qual", 
               "toilet_qual", "water_qual")
i = dhs
i$rooms = range01(i$rooms)
i$floor_qual = range01(i$floor_qual)
i$toilet_qual = range01(i$toilet_qual)
i$water_qual = range01(i$water_qual)
dhs$sum_index = rowSums(i[, index_vars])
dhs$object_index = stata_index(dhs[, c("electric", "phone", "radio", "tv", "car")])
dhs$index = -stata_index(dhs[, index_vars])

#get wdi and convert to long format for merging
wdi = read.csv("../data/wdi.csv", na.strings = "..", stringsAsFactors=F)
wdi[wdi=="Population, total"] = "pop"
wdi[wdi=="GDP per capita (constant 2010 US$)"] = "gdp"
wdi[wdi=="Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)"] = "poverty"
wdi[wdi=="Urban population (% of total population)"] = "urban"
wdi[wdi=="Agriculture, forestry, and fishing, value added (% of GDP)"] = "ag"
wdi[wdi=="GINI index (World Bank estimate)"] = "gini"
wdi[wdi=="PPP conversion factor, GDP (LCU per international $)"] = "ppp"
wdi = wdi[, c(1,  3, 5:ncol(wdi))]
wdi = gather(wdi, year, val, `X2000..YR2000.`:`X2019..YR2019.`, factor_key=TRUE)
wdi[wdi==""] = NA
wdi = wdi[complete.cases(wdi), ]
wdi = spread(wdi, Series.Name, val)
wdi$year = substr(wdi$year, 2, 5)
names(wdi)[1] = "country"
wdi$country = tolower(wdi$country)

#change names so it will merge
wdi[wdi=="burkina faso"] = "burkina_faso"
wdi[wdi=="congo, dem. rep."] = "democratic_republic_of_congo"
wdi[wdi=="cote d'ivoire"] = "cote_d_ivoire"
wdi[wdi=="sierra leone"] = "sierra_leone"

#linearly interpolate missing values
for (c in unique(wdi$country)) {
    temp = wdi[wdi$country==c, ]
    temp = temp[complete.cases(temp), ]
    gini = approx(temp$year, temp$gini, xout=unique(wdi$year), rule=2)
    pov = approx(temp$year, temp$poverty, xout=unique(wdi$year), rule=2)
    wdi[wdi$country==c, "gini"] = gini$y
    wdi[wdi$country==c, "poverty"] = pov$y
}
#for country where only one value, just set to that value
wdi[wdi$country=="zimbabwe", "gini"] = mean(wdi[wdi$country=="zimbabwe", "gini"], na.rm=T)
wdi[wdi$country=="zimbabwe", "poverty"] = mean(wdi[wdi$country=="zimbabwe", "poverty"], na.rm=T)

wdi$country_year = paste0(wdi$country, "_", wdi$year)

vars = readRDS("../data/DHS_output_2019-12-18_countryvariance.RDS") 
conv = data.frame(cname = unique(dhs$cname), 
                  country=c("angola", "burkina_faso", "benin", 
                            "democratic_republic_of_congo", "cote_d_ivoire", "cameroon", 
                            "", "ghana", "guinea", "kenya", "","lesotho", "mali", "malawi", 
                            "mozambique", "nigeria", "rwanda", "sierra_leone", "senegal", 
                            "", "togo", "tanzania", "uganda", "zambia", "zimbabwe"))
vars = merge(vars, conv)

preds = read.csv("../data/predictions/preds.csv")
preds =  preds %>% dplyr::group_by(country_year) %>% 
    dplyr::summarize(r2 = cor(wealthpooled,  Resnet.18.MS.NL.concat)^2, 
              rmse = sqrt(mean((Resnet.18.MS.NL.concat - wealthpooled)^2)))
preds = merge(preds, wdi, by="country_year")
preds$gdp = log(preds$gdp)
preds$pop = log(preds$pop)
names(preds)[c(6:10, 12)] = c("Agriculture", "GDP per capita", "GINI", "Population", 
                              "% under $1.90", "% Urban")
preds = merge(preds, vars[, c("country", "year", "diff", "btw")], by=c("country", "year"), all.x=T)
names(preds)[13:14] = c("Within village variance", "Between village variance")

panel_set(list(preds[, c("GINI", "r2")], preds[, c("% Urban", "r2")],
               preds[, c("% under $1.90", "r2")], preds[, c("GDP per capita", "r2")],
               preds[, c("Agriculture", "r2")],  preds[, c("Population", "r2")], 
               preds[, c("Within village variance", "r2")],  
               preds[, c("Between village variance", "r2")]), 
          subtitles=c("GINI Index", "Urban Population, % of total", 
                      "Poverty headcount ratio at $1.90 a day", "GDP per capita (2010 $)",
                      "Agriculture, forestry, and fishing, value added (% of GDP)", 
                      "Total Population", "DHS within village wealth variance", 
                      "DHS between village wealth variance"),
          name="", unif_bounds=F, n=4,
          save_path="../processed_fig/FigureS11.pdf", font="sans")
