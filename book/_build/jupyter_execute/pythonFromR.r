library(reticulate)
pd = import("pandas")
url = "https://raw.githubusercontent.com/bcaffo/ds4bme_intro/master/data/kirby127a_3_1_ax_283Labels_M2_corrected_stats.csv"
dat = pd$read_csv(url)
head(dat)

npr = import("numpy.random")
normalGenerator = npr$normal
normalGenerator(size=as.integer(5))
