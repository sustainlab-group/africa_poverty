source("00_dependencies.R")

#read in the data of jittered results
jitter = read.csv("../data/dhs_jitter_results_knn_jitter.csv")
x = jitter %>% dplyr::group_by(jitter) %>%
    dplyr::summarise(r2_train=mean(r2_train), r2_test=mean(r2_test), r2_true=mean(r2_true))

#plot
y = ggplot() + 
    geom_jitter(aes(jitter, r2_test, color="jittered test"), alpha=0.09, width=.55, data=jitter) + 
    geom_jitter(aes(jitter, r2_true, color="unjittered test"), alpha=0.09, width=.55, data=jitter) + 
    geom_line(aes(jitter, r2_test, color="jittered test"), data=x) +
    geom_line(aes(jitter, r2_true, color="unjittered test"), data=x) +
    geom_smooth(aes(jitter, r2_test, color="jittered test"), method = "lm", 
                formula = y ~ poly(x, 2), data=x, se=F, fullrange=T, linetype=2, size=0.5) +
    theme_anne(font="sans") + labs(color = "") + xlab("Mean jitter (km)") + 
    ylab("r2") + ylim(0.43, 0.75) + xlim(0, 11.5)
ggsave(filename="../processed_fig/FigureS12.pdf", plot=y, device="pdf", width=7, height=4)
