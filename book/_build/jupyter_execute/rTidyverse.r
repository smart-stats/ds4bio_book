library(tidyverse)

dat = read_csv("https://raw.githubusercontent.com/bcaffo/ds4bme_intro/master/data/kirby127a_3_1_ax_283Labels_M2_corrected_stats.csv")
head(dat)

dat = dat %>% select(-X1, -rawid)
dat %>% head

t1l1 = dat %>% filter(type == 1, level == 1)
t1l1

## Set the base plot
g = ggplot(data = t1l1, aes(x = roi, y = volume, fill = roi)) 
## Add the bar graphs
g = g + geom_col()
## My fonts weren't rendering correctly, so changing to a different one
g = g + theme(text=element_text(family="Consolas"))
## The x axis labels are long and overlap if you don't rotate them
g = g + theme(axis.text.x = element_text(angle = 45))
## Show the plot
g
