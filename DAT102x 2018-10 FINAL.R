###############################################################################################
# DAT102x October 2018 - Predicting Prevalence of Undernourishment
#
# Final Script - 6.87 RMSE in DrivenData competition
#
###############################################################################################
# Script assumes that input files are in /Input folder within the working folder.
# Prediction CSV files will be saved to /Submissions folder.
###############################################################################################

library(caret)
library(caTools)
library(doParallel)
library(gbm)
library(ggplot2)
library(dplyr)
library(mice)
library(Amelia)
library(data.table)

    # Function to replace values in dataframe
    replaceDfValue = function(dataframe, cols, rows, newValue = NA) {
      if (any(rows)) {
        set(dataframe, rows, cols, newValue)
      }
    }
    
    # Function to Find and Remove Outliers
    remove_outliers <- function(x, na.rm = TRUE, ...) {
      qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
      H <- 1.5 * IQR(x, na.rm = na.rm)
      y <- x
      y[x < (qnt[1] - H)] <- qnt[1]
      y[x > (qnt[2] + H)] <- qnt[2]
      y
    }
    
    #Function for Scatter Plot of Predicted vs Actual for Test Dataset
    scatterplot <- function(tst, prd, res) {
        filename <- paste("./Plots/",trimws(res),trimws(".jpeg"))
        filename <- trimws(gsub(" \\.",".", filename))
        filename <- gsub("/ ", "/", filename)
        print(filename)
        #To Screen
        plot(tst, prd, main = res, sub = "Predicted vs Actual", type = "p", xlab="Actual", ylab="Predicted")
        ##To Jpeg File
        jpeg(file = filename)
        plot(tst, prd, main = res, sub = "Predicted vs Actual", type = "p", xlab="Actual", ylab="Predicted")
        dev.off()
    }

rm(.Random.seed, envir=globalenv())    
set.seed(55)

train_values <- read.csv("./input/train_Values.csv")
train_labels <- read.csv("./input/train_labels.csv")
df.Train <- merge(train_values, train_labels,by="row_id")
rm(train_values)
rm(train_labels)
df.Score <- read.csv("./input/test_values.csv")

## Remove rows with more than 30% NA
df.Train <- df.Train[-which(rowMeans(is.na(df.Train)) > 0.3),]

## Remove columns with more than 30% NA in train data, then drop same columns from score data
df.Train <- df.Train[, -which(colMeans(is.na(df.Train)) > 0.3)]
df.Score$prevalence_of_undernourishment <- 0
df.Score <- df.Score[, colnames(df.Train)]

#Create dataframe of original obesity and animal_protein values for use in second model after running first model predictions
df.Train2 <- as.data.frame(cbind(df.Train$row_id, df.Train$obesity_prevalence, df.Train$avg_supply_of_protein_of_animal_origin, df.Train$prevalence_of_undernourishment, df.Train$country_code))
names(df.Train2) <- c("row_id","obesity_prevalence","avg_supply_of_protein_of_animal_origin","prevalence_of_undernourishment","country_code")
df.Score2 <- as.data.frame(cbind(df.Score$row_id, df.Score$obesity_prevalence, df.Score$avg_supply_of_protein_of_animal_origin, df.Score$country_code))
names(df.Score2) <- c("row_id","obesity_prevalence","avg_supply_of_protein_of_animal_origin","country_code")

#Combine Train and Test sets, impute missing then split out again
df.Merged <- rbind(df.Train, df.Score)

# Replace values within country_codes with mean of other values from same country_code
df.Merged <- df.Merged %>% group_by(country_code) %>% mutate(avg_supply_of_protein_of_animal_origin = ifelse(is.na(avg_supply_of_protein_of_animal_origin),mean(avg_supply_of_protein_of_animal_origin, na.rm = T), avg_supply_of_protein_of_animal_origin))
df.Merged <- df.Merged %>% group_by(country_code) %>% mutate(caloric_energy_from_cereals_roots_tubers = ifelse(is.na(caloric_energy_from_cereals_roots_tubers),mean(caloric_energy_from_cereals_roots_tubers, na.rm = T), caloric_energy_from_cereals_roots_tubers))

#Replace outlier percentage values with 0, 95, Mean/Median (or value between mean and median)
replaceDfValue(df.Merged, "percentage_of_arable_land_equipped_for_irrigation", which(df.Merged$percentage_of_arable_land_equipped_for_irrigation > 100), 30)
replaceDfValue(df.Merged, "net_oda_received_percent_gni", which(df.Merged$net_oda_received_percent_gni > 100), 95)
replaceDfValue(df.Merged, "net_oda_received_percent_gni", which(df.Merged$net_oda_received_percent_gni < 0), 0)
replaceDfValue(df.Merged, "access_to_improved_sanitation", which(df.Merged$access_to_improved_sanitation > 100), 65)
replaceDfValue(df.Merged, "access_to_improved_water_sources", which(df.Merged$access_to_improved_water_sources > 100), 83.5)
replaceDfValue(df.Merged, "access_to_electricity", which(df.Merged$access_to_electricity > 100), 78)

# Replace outliers in other columns based on quartile values using function defined previously
df.Merged$agricultural_land_area <- remove_outliers(df.Merged$agricultural_land_area)
df.Merged$avg_value_of_food_production <- remove_outliers(df.Merged$avg_value_of_food_production)
df.Merged$cereal_yield <- remove_outliers(df.Merged$cereal_yield)
df.Merged$forest_area <- remove_outliers(df.Merged$forest_area)
df.Merged$population_growth <- remove_outliers(df.Merged$population_growth)
df.Merged$rural_population <- remove_outliers(df.Merged$rural_population)
df.Merged$total_land_area <- remove_outliers(df.Merged$total_land_area)
df.Merged$total_population <- remove_outliers(df.Merged$total_population)

# Create engineered features (land shares, population shares, etc.)
df.Merged$agricultural_area_share <- df.Merged$agricultural_land_area / df.Merged$total_land_area
df.Merged$agricultural_area_per_capita <- df.Merged$agricultural_land_area / df.Merged$total_population
df.Merged$co2_emissions_by_area <- df.Merged$co2_emissions / df.Merged$total_land_area
df.Merged$co2_emissions_per_capita <- df.Merged$co2_emissions / df.Merged$total_population
df.Merged$forest_area_share <- df.Merged$forest_area / df.Merged$total_land_area
df.Merged$forest_area_per_capita <- df.Merged$forest_area / df.Merged$total_population
df.Merged$total_area_per_capita <- df.Merged$total_land_area / df.Merged$total_population
df.Merged$urban_pop_share <- df.Merged$urban_population / df.Merged$total_population
df.Merged$rural_pop_share <- df.Merged$rural_population / df.Merged$total_population
df.Merged$total_labor_force_proportion <- df.Merged$total_labor_force / df.Merged$total_population
df.Merged$country_code <- as.numeric(df.Merged$country_code)

## Amelia missing values imputation

  # First create bounds matrix from min and max values for each column (so imputed values can't be outside min or max values)
  bds <- matrix (ncol = 3)
  for (r in 1:ncol(df.Merged)) {
    if (r == 2) {colbd <- matrix(c(2,0,0), ncol = 3)}
    else {colbd <- matrix(c(r, min(df.Merged[,r], na.rm=TRUE), max(df.Merged[,r], na.rm=TRUE)), ncol = 3)}
    bds <- rbind(bds, colbd)
  }
  bds <- bds[-1,]
  
  # Imputation step
  ncpus = detectCores() - 1
  m = ncpus * 10
  idvars = c('row_id', 'country_code', 'prevalence_of_undernourishment')
  a.out = amelia(df.Merged, m = 1, idvars = idvars, parallel = 'multicore', ncpus = ncpus, bounds = bds) #, noms = '' )
  df.Merged <- a.out$imputations[[1]]

# Create log value columns
df.Merged$ln_food_imports_as_share_of_merch_exports <- log(df.Merged$food_imports_as_share_of_merch_exports)
df.Merged$ln_gross_domestic_product_per_capita_ppp <- log(df.Merged$gross_domestic_product_per_capita_ppp)
df.Merged$ln_net_oda_received_percent_gni <- log(df.Merged$net_oda_received_percent_gni)
df.Merged$ln_net_oda_received_per_capita <- log(df.Merged$net_oda_received_per_capita)
df.Merged$ln_per_capita_food_production_variability <- log(df.Merged$per_capita_food_production_variability)
df.Merged$ln_hiv_incidence <- log(df.Merged$hiv_incidence)
df.Merged$ln_co2_emissions <- log(df.Merged$co2_emissions)
df.Merged$ln_total_labor_force <- log(df.Merged$total_labor_force)
df.Merged$ln_agricultural_area_per_capita <- log(df.Merged$agricultural_area_per_capita)
df.Merged$ln_forest_area_per_capita <- log(df.Merged$forest_area_per_capita)
df.Merged$ln_total_area_per_capita <- log(df.Merged$total_area_per_capita)
df.Merged$ln_co2_emissions_by_area <- log(df.Merged$co2_emissions_by_area)
df.Merged$ln_co2_emissions_per_capita <- log(df.Merged$co2_emissions_per_capita)

df.Merged$ln_net_oda_received_percent_gni <-replace(df.Merged$ln_net_oda_received_percent_gni, is.infinite(df.Merged$ln_net_oda_received_percent_gni), NA)

# Replace any NaN from log calcs with NA
is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))
df.Merged[is.nan(df.Merged)] <- NA

# MICE imputation for any NA values resulting from log conversions
imputed_Data <- mice(df.Merged, m=1, maxit = 1, method = 'cart', seed = 500)
df.Merged <- complete(imputed_Data, 1)

# Drop selected columns (including those which have been converted to log values)
drops <- c("food_imports_as_share_of_merch_exports","gross_domestic_product_per_capita_ppp","net_oda_received_percent_gni","net_oda_received_per_capita",
           "per_capita_food_production_variability","hiv_incidence", "co2_emissions","total_labor_force","agricultural_area_per_capita","forest_area_per_capita",
           "total_area_per_capita","co2_emissions_by_area","co2_emissions_per_capita")

df.Merged <- df.Merged[ , !(names(df.Merged) %in% drops)]

# Split back out in to TRAIN+TEST and SCORE datasets
df.Train <- df.Merged[which(df.Merged$prevalence_of_undernourishment != 0),]
df.Score <- df.Merged[which(df.Merged$prevalence_of_undernourishment == 0),]

## Split TRAIN dataset in to TRAIN and TEST, ensuring Country_Codes kept together
n <- as.data.frame(unique(df.Train$country_code))
n$ind <- sample(c("TRAIN","TEST"), nrow(n), replace=TRUE, prob=c(0.8, 0.2))
names(n) <- c('ccode', 'ind')
data1 <- n[n$ind == "TRAIN", ]
data2 <- n[n$ind == "TEST", ]

df.ForSplit <- df.Train
df.Train <- df.ForSplit %>% filter(country_code %in% data1$ccode)
df.test <- df.ForSplit %>% filter(country_code %in% data2$ccode)

# Clean up objects from data processing
rm(drops, data1, data2, df.Merged, df.ForSplit, imputed_Data, n, bds, ncpus, m, idvars, a.out, r, colbd)

#Create empty dataframe for metrics
df.Results <- data.frame("Model" = character(0), "mse" = numeric(0), "rmse" = numeric(0), "coef" = numeric(0))

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

nc <- ncol(df.Train)

###############################
# GBM
###############################

rm(.Random.seed, envir=globalenv())
set.seed(16)

fitControl <- trainControl(method = "cv", number = 5)
formula <- prevalence_of_undernourishment ~ .
Grid <- expand.grid(n.trees = c(1500), interaction.depth = c(30), shrinkage = c(0.1), n.minobsinnode = c(10))
fit <- train(formula, data=df.Train, method = 'gbm', trControl=fitControl, tuneGrid=Grid, metric='RMSE', maximize=FALSE, distribution="gaussian")
mdl.predictions <- predict(fit, df.test)
results <- cbind(mdl.predictions, df.test$prevalence_of_undernourishment) 
colnames(results) <- c('pred','real')
results <- as.data.frame(results)
mse <- round(mean((results$real-results$pred)^2), 3)
coef <- round((cor(results$real, results$pred) ^ 2), 3)
r_mse <- round(mse^0.5, 3)
print(c(mse, r_mse, coef))
df.Results <- rbind(df.Results, data.frame("Model" = "GBM (gbm)", "mse" = mse, "rmse" = r_mse, "coef" = coef))
scatterplot(results$real, results$pred, df.Results[(nrow(df.Results)), 1])

#################################################################################
#Output Scoring Predictions from 1st model only (no 2nd model fix applied)
#################################################################################

score_prediction <- as.data.frame((predict(fit, df.Score)))
score_prediction <- cbind(df.Score$row_id, score_prediction)
names(score_prediction) <- c("row_id","prevalence_of_undernourishment")
write.csv(score_prediction, paste0("./Submissions/", Sys.Date(), " submission 0x GBM without fix.csv"), row.names = F)

#########################################################################################
# Plots showing outliers to be fixed by second model
#########################################################################################

plot_columns <- as.data.frame(cbind(df.Score$obesity_prevalence, df.Score$avg_supply_of_protein_of_animal_origin, score_prediction$prevalence_of_undernourishment, df.Score$country_code))
names(plot_columns) <- c("obesity_prevalence","avg_supply_of_protein_of_animal_origin","prevalence_of_undernourishment", "country_code")

p <- ggplot(plot_columns) +
  geom_point(aes(x=obesity_prevalence, y=prevalence_of_undernourishment, col=as.factor(country_code), size = 2), show.legend = FALSE) + 
  xlab("Obesity Prevalence") + ylab("Prevalance of Undernourishment") + ggtitle("Undernourishment vs Obesity - 1st model output") +
  scale_x_continuous(labels=function(n){format(n, scientific = FALSE)}) +
  theme(plot.title = element_text(size = 11, face = "bold"))
print(p)

p <- ggplot(plot_columns) +
  geom_point(aes(x=avg_supply_of_protein_of_animal_origin, y=prevalence_of_undernourishment, col=as.factor(country_code), size = 2), show.legend = FALSE) + 
  xlab("avg_supply_of_protein_of_animal_origin") + ylab("Prevalance of Undernourishment") + ggtitle("Undernourishment vs avg_supply_of_protein_of_animal_origin - 1st model output") +
  scale_x_continuous(labels=function(n){format(n, scientific = FALSE)}) +
  theme(plot.title = element_text(size = 11, face = "bold"))
print(p)

#########################################################################################
# Apply fix to outliers in obesity and animal_protein plots by training a second
# model using these two columns + country_code only. Replace using dplyr mutate.
#########################################################################################

# Bind score predictions to df.Score2 table and impute missing values using mean of values for each country_code
df.Score2 <- cbind (score_prediction, df.Score2$obesity_prevalence, df.Score2$avg_supply_of_protein_of_animal_origin, df.Score2$country_code)
names(df.Score2) <- c("row_id","prevalence_of_undernourishment","obesity_prevalence","avg_supply_of_protein_of_animal_origin","country_code")
df.Score2 <- df.Score2 %>% group_by(country_code) %>% mutate(avg_supply_of_protein_of_animal_origin = ifelse(is.na(avg_supply_of_protein_of_animal_origin),mean(avg_supply_of_protein_of_animal_origin, na.rm = T), avg_supply_of_protein_of_animal_origin))
df.Score2 <- df.Score2 %>% group_by(country_code) %>% mutate(obesity_prevalence = ifelse(is.na(obesity_prevalence),mean(obesity_prevalence, na.rm = T), obesity_prevalence))

# Impute missing values in reduced Train dataset
df.Train2 <- df.Train2 %>% group_by(country_code) %>% mutate(avg_supply_of_protein_of_animal_origin = ifelse(is.na(avg_supply_of_protein_of_animal_origin),mean(avg_supply_of_protein_of_animal_origin, na.rm = T), avg_supply_of_protein_of_animal_origin))
df.Train2 <- df.Train2 %>% group_by(country_code) %>% mutate(obesity_prevalence = ifelse(is.na(obesity_prevalence),mean(obesity_prevalence, na.rm = T), obesity_prevalence))
df.Train2 <- df.Train2[!is.na(df.Train2$obesity_prevalence),]

# Train 2nd model
Grid2 <- expand.grid(n.trees = c(1500), interaction.depth = c(30), shrinkage = c(0.01), n.minobsinnode = c(5))
fit2 <- train(formula, data=df.Train2, method = 'gbm', trControl=fitControl, tuneGrid=Grid2, metric='RMSE', maximize=FALSE, distribution="gaussian")

# Predict Score labels using 2nd model and bind column to df.Score2 table
mdl.predictions2 <- predict(fit2, df.Score2)
mdl.predictions2 <- as.data.frame(cbind(df.Score2$row_id, mdl.predictions2))
names(mdl.predictions2) <- cbind("row_id","prevalence_of_undernourishment")
df.Score2$prediction2 <- mdl.predictions2$prevalence_of_undernourishment

# Use dplyr mutate function to replace outlier predictions from 1st model with predicted values from 2nd model
df.Score2 <- df.Score2 %>% mutate(prevalence_of_undernourishment = ifelse(obesity_prevalence > 25 & prevalence_of_undernourishment > 15 &
                                                                      prevalence_of_undernourishment > 1.5 * prediction2,
                                                                      prediction2, prevalence_of_undernourishment))


df.Score2 <- df.Score2 %>% mutate(prevalence_of_undernourishment = ifelse(avg_supply_of_protein_of_animal_origin > 25 & prevalence_of_undernourishment > 24 &
                                                                    prevalence_of_undernourishment > 1.5 * prediction2,
                                                                    prediction2, prevalence_of_undernourishment))


submit <- df.Score2[,c(1,2)]

write.csv(submit, paste0("./Submissions/", Sys.Date(), " submission 0x GBM with 2nd model fix.csv"), row.names = F)

#########################################################################################
# Plots of predictions after fixing by second model
#########################################################################################

plot_columns <- as.data.frame(cbind(df.Score2$obesity_prevalence, df.Score2$avg_supply_of_protein_of_animal_origin, df.Score2$prevalence_of_undernourishment, df.Score$country_code))
names(plot_columns) <- c("obesity_prevalence","avg_supply_of_protein_of_animal_origin","prevalence_of_undernourishment", "country_code")

p <- ggplot(plot_columns) +
  geom_point(aes(x=obesity_prevalence, y=prevalence_of_undernourishment, col=as.factor(country_code), size = 2), show.legend = FALSE) + 
  xlab("Obesity Prevalence") + ylab("Prevalance of Undernourishment") + ggtitle("Undernourishment vs Obesity - 2nd model output") +
  scale_x_continuous(labels=function(n){format(n, scientific = FALSE)}) +
  theme(plot.title = element_text(size = 11, face = "bold"))
print(p)

p <- ggplot(plot_columns) +
  geom_point(aes(x=avg_supply_of_protein_of_animal_origin, y=prevalence_of_undernourishment, col=as.factor(country_code), size = 2), show.legend = FALSE) + 
  xlab("avg_supply_of_protein_of_animal_origin") + ylab("Prevalance of Undernourishment") + ggtitle("Undernourishment vs avg_supply_of_protein_of_animal_origin - 2nd model output") +
  scale_x_continuous(labels=function(n){format(n, scientific = FALSE)}) +
  theme(plot.title = element_text(size = 11, face = "bold"))
print(p)
             