#!/bin/bash

Echo Starting to run startup scripts

Rscript 00_dependencies.R

#StataSE -e do work/02_create_wealth_index_for_ipums.do &

Echo Running the scripts to create main figures

Rscript figure_01_surveyrates.R
Rscript figure_02_dhsperformancemaps.R
Rscript figure_03_modelperformance.R
Rscript figure_04_performancelsmsovertime.R
Rscript figure_05_policywealthtemp.R
Rscript figure_06_nigeriamap.R
wait

Echo Running the scripts to create supplemental figures

Rscript figure_s1_datalocations.R
Rscript figure_s2_consumptionvsindex.R
Rscript figure_s3_indexconstruction.R
Rscript figure_s4_correlationofmodels.R
Rscript figure_s6_performancevsWDI.R
Rscript figure_s7_jitter.R
Rscript figure_s8_overtimeperformance.R
Rscript figure_s9_simulationovertime.R
Rscript figure_s10_NLovertime.R
wait

Echo Finished running
