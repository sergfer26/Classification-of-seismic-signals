# install.packages('RSEIS')
library(RSEIS)

# Exhalation
dir_exh <- 'C:/Users/julio/Documents/ITT2020/5. CENAPRED- TOPIC A/exhalation14072019/20190714121358'
exhalation_names <- list.files(dir_exh)

exhalation_series <- lapply(paste0(dir_exh,'/',exhalation_names), read1sac, Iendian= 2)

par(mfrow = c(3,6))
for (i in 1:18) {
  DOsgram(exhalation_series[[i]]$amp,exhalation_series[[i]]$HEAD$delta)
}
# Explosion
dir_exp <- 'C:/Users/julio/Documents/ITT2020/5. CENAPRED- TOPIC A/explosion14032019/20190314202837'
explotion_names <- list.files(dir_exp)

explotion_series <- lapply(paste0(dir_exp,'/',explotion_names), read1sac, Iendian= 2)
dev.off()
par(mfrow = c(2,6))
for (i in 1:12) {
  DOsgram(explotion_series[[i]]$amp,explotion_series[[i]]$HEAD$delta)
}

# Tremor
dir_trem <- 'C:/Users/julio/Documents/ITT2020/5. CENAPRED- TOPIC A/tremor08072019/20190708005546'
tremor_names <- list.files(dir_trem)

tremor_series <- lapply(paste0(dir_trem,'/',tremor_names), read1sac, Iendian= 2)
dev.off()
par(mfrow = c(2,6))
for (i in 1:12) {
  DOsgram(tremor_series[[i]]$amp,tremor_series[[i]]$HEAD$delta)
}

# Volcano tectonic
dir_vt <- 'C:/Users/julio/Documents/ITT2020/5. CENAPRED- TOPIC A/vt30072019/20190730070713'
vt_names <- list.files(dir_vt)

vt_series <- lapply(paste0(dir_vt,'/',vt_names), read1sac, Iendian= 2)
dev.off()
par(mfrow = c(3,7))
for (i in 1:21) {
  DOsgram(vt_series[[i]]$amp,vt_series[[i]]$HEAD$delta)
}

# Interactive example
SPECT.drive(vt_series[[1]]$amp,vt_series[[1]]$HEAD$delta)
