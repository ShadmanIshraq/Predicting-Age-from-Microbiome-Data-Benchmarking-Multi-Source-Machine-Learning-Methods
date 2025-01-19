# Load required libraries
library(curatedMetagenomicData)
library(mia)


# Load relative abundance dataset
tse.relab <- curatedMetagenomicData("LifeLinesDeep.+.relative_abundance", dryrun = FALSE) |>
  mergeData()

# Compute relative abundance
tse.relab <- transformAssay(tse.relab, assay.type = "relative_abundance", method = "relabundance")



# Agglomerate data
tse <- mergeFeaturesByRank(tse.relab, rank = "order")

# Apply CLR transform
tse <- transformAssay(tse.relab, assay.type = "relative_abundance", method = "clr",
                      MARGIN="samples", pseudocount=1)

# Get assay
assay <- assay(tse, "clr")
# Transpose assay
assay <- t(assay)

# Convert into data.frame
df <- as.data.frame(assay)

# Add labels to assay
labels <- colData(tse)$age
labels <- as.numeric(labels)
df$age <- labels 



#Dataset for diversity and indicator species analysis based on age category

# Agglomerate data
tse_acg <- mergeFeaturesByRank(tse.relab, rank = "order")

#Apply CLR transform
tse_acg <- transformAssay(tse.relab, assay.type = "relative_abundance", method = "clr",
                          MARGIN="samples", pseudocount=1)

# Get assay
assay <- assay(tse_acg)
# Transpose assay
assay <- t(assay)

# Convert into data.frame
df_acg <- as.data.frame(assay)

# Add labels to assay
labels <- colData(tse_acg)$age_category
df_acg$age_category <- labels 




# Load marker abundance dataset

tse_markab<-curatedMetagenomicData("LifeLinesDeep.+.marker_abundance",dryrun = FALSE) |>
  mergeData()


# Apply CLR transform
tse1 <- transformAssay(tse_markab, assay.type = "marker_abundance", method = "log10",
                       MARGIN="samples", pseudocount=1)

# Get assay
assay <- assay(tse1, "log10")
# Transpose assay
assay <- t(assay)

# Convert into data.frame
df_markab <- as.data.frame(assay)

# Add labels to assay
labels <- colData(tse_markab)$age
labels <- as.numeric(labels)
df_markab$age <- labels 




# Load relative abundance dataset

tse_pathab<-curatedMetagenomicData("LifeLinesDeep.+.pathway_abundance",dryrun = FALSE) |>
  mergeData()

# Apply CLR transform
tse2 <- transformAssay(tse_pathab, assay.type = "pathway_abundance", method = "log10",
                       MARGIN="samples", pseudocount=1)

# Get assay
assay <- assay(tse2, "log10")
# Transpose assay
assay <- t(assay)

# Convert into data.frame
df_pathab <- as.data.frame(assay)

# Add labels to assay
labels <- colData(tse_pathab)$age
labels <- as.numeric(labels)
df_pathab$age <- labels 









############## Single-omic dataset preparation





# Calculate the coefficient of variation for each numeric feature in the dataset
cv_values <- apply(df[, sapply(df, is.numeric)], 2, function(x) {
  if (mean(x) != 0) {
    sd(x) / mean(x)
  } else {
    NA  # Avoid division by zero
  }
})

# Define a threshold for the coefficient of variation
cv_threshold <- 0.1  # Adjust this threshold as needed

# Filter columns based on the threshold
filtered_df <- df[, names(cv_values)[!is.na(cv_values) & cv_values > cv_threshold]]








################# Multi-omics dataset preparation



# Ensure that 'age' is a numeric column
df$age <- as.numeric(df$age)

# Calculate Spearman correlation for each feature with 'age'
# Apply the correlation only to numeric columns (excluding 'age' itself)
correlations <- apply(df[, -ncol(df)], 2, function(x) cor(x, df$age, method = "spearman"))




# Ensure that 'age' is a numeric column in df_markab
df_markab$age <- as.numeric(df_markab$age)

# Calculate Spearman correlation for each feature with 'age'
# Apply the correlation only to numeric columns (excluding 'age' itself)
correlations_markab <- apply(df_markab[, -ncol(df_markab)], 2, function(x) cor(x, df_markab$age, method = "spearman"))





# Ensure that 'age' is a numeric column in df_markab
df_pathab$age <- as.numeric(df_pathab$age)

# Calculate Spearman correlation for each feature with 'age'
# Apply the correlation only to numeric columns (excluding 'age' itself)
correlations_pathab <- apply(df_pathab[, -ncol(df_pathab)], 2, function(x) cor(x, df_pathab$age, method = "spearman"))




# Set correlation threshold
threshold <- 0.16

# Filter `df` dataset
filtered_features_df <- names(correlations[abs(correlations) >= threshold])
df_filtered <- df[, c(filtered_features_df, "age")]  # Keep filtered features and 'age'

# Filter `df_markab` dataset
filtered_features_markab <- names(correlations_markab[abs(correlations_markab) >= threshold])
df_markab_filtered <- df_markab[, filtered_features_markab]  # Exclude 'age' here

# Filter `df_pathab` dataset
filtered_features_pathab <- names(correlations_pathab[abs(correlations_pathab) >= threshold])
df_pathab_filtered <- df_pathab[, filtered_features_pathab]  # Exclude 'age' here

# Combine the filtered datasets using cbind
# Exclude the 'age' column from df_filtered during cbind
combined_filtered_df <- cbind(df_filtered[, filtered_features_df, drop = FALSE], 
                              df_markab_filtered, 
                              df_pathab_filtered)

# Add the 'age' column back to the combined dataset
combined_filtered_df <- cbind(age = df_filtered$age, combined_filtered_df)

# Ensure no column is named `df_filtered` or similar
colnames(combined_filtered_df) <- make.unique(colnames(combined_filtered_df))








############## IntegratedLearner dataset preparation




# Create the sample_metadata dataframe
sample_metadata <- data.frame(
  Y = combined_filtered_df$age,
  subjectID = rownames(combined_filtered_df),
  row.names = rownames(combined_filtered_df) 
)

# Ensure Y is continuous, convert it to numeric 
sample_metadata$Y <- as.numeric(sample_metadata$Y)




# Function to remove the 'age' column and transpose the dataset
process_dataset <- function(dataset) {
  dataset_no_age <- dataset[ , !names(dataset) %in% "age"]  
  transposed_dataset <- t(dataset_no_age)  
  transposed_dataset <- as.data.frame(transposed_dataset)  
  return(transposed_dataset)
}

# Process each dataset
transposed_relab <- process_dataset(df_filtered)
transposed_patab <- process_dataset(df_pathab_filtered)
transposed_markab <- process_dataset(df_markab_filtered)



# Add featureType column to each dataset as specified
transposed_relab$featureType <- 'species'
transposed_patab$featureType <- 'pathway'
transposed_markab$featureType <- 'biomarker'



# Merging the datasets using rbind
t_filtered_merged_dataset <- rbind(transposed_relab, transposed_patab, transposed_markab)




# Extracting featureType from t_filtered_merged_dataset
featureType <- t_filtered_merged_dataset$featureType

# Create feature_metadata dataset with row names
feature_metadata <- data.frame(featureID = rownames(t_filtered_merged_dataset), featureType = featureType)

# Set row names of feature_metadata to be the same as t_filtered_merged_dataset
rownames(feature_metadata) <- rownames(t_filtered_merged_dataset)



feature_table <- subset(t_filtered_merged_dataset, select = -featureType)






