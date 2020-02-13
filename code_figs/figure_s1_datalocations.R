source("00_dependencies.R")

######################################################
# Read in geo stuff
######################################################

africa = readRDS("../data/shapefiles/africa_gadm.rds")

######################################################
# Read in DHS locations
######################################################

dhs = read.csv("../data/predictions/dhs_preds_incountry.csv")
dhs = dplyr::select(dhs, lat, lon, labels)

######################################################
# Read in LSMS locations
######################################################

lsms = read.csv("../data/predictions/lsms_preds_ooc.csv")
lsms = dplyr::select(lsms, lat, lon, wealthpooled)

######################################################
# Make maps
######################################################

data_df = broom::tidy(rgeos::gSimplify(africa, 0.05))
countries_df = africa[africa@data$NAME_0 %in% c("Sierra Leone", "Togo", "Benin", "Nigeria", "Senegal", 
                                                "Tanzania", "Malawi", "Ethiopia", "Rwanda", "Kenya", 
                                                "Zambia", "Mozambique", "Angola", "Democratic Republic of the Congo", 
                                                "Guinea", "Côte d'Ivoire", "Gambia", "Ghana", "Uganda",
                                                "Burkina Faso", "Mali", "Cameroon", "Lesotho", "Zimbabwe"),]
countries_df = broom::tidy(rgeos::gSimplify(countries_df, 0.05))

a = ggplot() +
  geom_polygon(aes(x=long, y=lat, group=group), data=data_df, fill="grey") + 
  geom_path(aes(x=long, y=lat, group=group), data=data_df, color="white", size = 0.08) + 
  geom_point(aes(lon, lat, color=labels), size=0.2, data=dhs) +
  scale_color_gradientn(colors = rainbow(3)) +
  geom_path(aes(x=long, y=lat, group=group), data=countries_df, color="black", size = 0.3) + 
  coord_equal() + ggtitle('') +
  theme_anne('sans', size = 10) +
  theme(
    line = element_blank(),
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.background = element_blank(),
    text = element_text(size = 10, family = 'sans'),
    legend.text = element_blank(), 
    legend.title = element_blank()
  )

countries_df = africa[africa@data$NAME_0 %in% c("Malawi", "Nigeria", "Ethiopia", "Tanzania", 
                                                "Uganda"),]
countries_df = broom::tidy(rgeos::gSimplify(countries_df, 0.05))

b = ggplot() + 
  geom_polygon(aes(x=long, y=lat, group=group), data=data_df, fill="grey") + 
  geom_path(aes(x=long, y=lat, group=group), data=data_df, color="white", size = 0.08) + 
  geom_point(aes(lon, lat, color=wealthpooled), size=0.2, data=lsms) +
  scale_color_gradientn(colors = rainbow(3)) +
  geom_path(aes(x=long, y=lat, group=group), data=countries_df, color="black", size = 0.3) + 
  coord_equal() + ggtitle('') +
  theme_anne('sans', size = 10) +
  theme(
    line = element_blank(),
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.background = element_blank(),
    text = element_text(size = 10, family = 'sans'),
    legend.text = element_blank(), 
    legend.title = element_blank()
  )

ggsave("../raw_fig/FigureS1_a.pdf", plot=a,  width=7, height=7)
ggsave("../raw_fig/FigureS1_b.pdf", plot=b,  width=7, height=7)
