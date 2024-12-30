library(vegan)
library(labdsv)  
library(dplyr)
library(mia)
library(miaViz)
library(ggsignif)
library(ggplot2)
library(phyloseq)
library(ggpubr)



######### Indicator Species Analysis


# Separate the microbiome data (without age_category) and the age categories
abundance_data <- df_acg %>% select(-age_category)
age_category <- df_acg$age_category

# Normalize the data (choose one method)
abundance_data_norm <- decostand(abundance_data, method = "total")  # TSS normalization

# Perform Indicator Species Analysis using `indval`
indval_result <- indval(abundance_data_norm, age_category)


# To extract significant indicator species (taxa) for specific age groups:
significant_taxa <- indval_result$sign[which(indval_result$pval <= 0.05)]
print(significant_taxa)



# Extract taxa names and significant p-values
significant_taxa <- rownames(indval_result$indval)[indval_result$pval <= 0.05]
indval_significant <- indval_result$indval[significant_taxa, ]
pval_significant <- indval_result$pval[significant_taxa]


# Extract the top 20 significant taxa
top_taxa_count <- 20  # Set the number of top taxa you want
significant_taxa <- significant_taxa[1:min(top_taxa_count, length(significant_taxa))]

# Create a data frame 
indval_df <- data.frame(
  Taxa = rep(significant_taxa, times = ncol(indval_significant)),  # Adjusted to the number of columns
  AgeCategory = rep(colnames(indval_significant), each = length(significant_taxa)),
  IndVal = as.vector(as.matrix(indval_significant[significant_taxa, ])),  # Ensure you subset here too
  PValue = rep(pval_significant[significant_taxa], times = ncol(indval_significant))
)



# Keeping only the species name
indval_df$Taxa <- gsub(".*\\|s__(.*)", "\\1", indval_df$Taxa)

# Print the data frame
print(indval_df)






############ Top prevalent species accross the samples

head(getPrevalence(tse.relab, rank= "species",assay.type = "relative_abundance", detection = 1/100, sort = TRUE, as_relative = TRUE),20)





############ Phylum abundance



#Extracting the abundance of the phylum accross the samples
# Agglomerating data by "phylum" rank
tse_phylum <- agglomerateByRank(tse.relab, rank = "phylum", onRankOnly = TRUE)

# Transforming the assay type to relative abundance
tse_phylum <- transformAssay(tse_phylum, assay.type = "relative_abundance", method = "relabundance")

# Getting the top 5 taxa
top_taxa <- getTop(tse_phylum, top = 5, assay.type = "relative_abundance")

# Renaming the "Phylum" rank to keep only top taxa and the rest as "Other"
phylum_renamed <- lapply(rowData(tse_phylum)$phylum,
                         function(x) { if (x %in% top_taxa) { x } else { "Other" } })

# Assigning the renamed phyla back to the rowData
rowData(tse_phylum)$phylum <- as.character(phylum_renamed)

# Calculating relative abundance percentages
assay_data <- assay(tse_phylum, assay.type = "relative_abundance")
phylum_sums <- rowSums(assay_data)
phylum_percentages <- phylum_sums / sum(phylum_sums) * 100

# Creating a mapping of phylum names to percentages
phylum_labels <- sapply(unique(rowData(tse_phylum)$phylum), function(phylum) {
  if (phylum == "Other") {
    percentage <- sum(phylum_percentages[!names(phylum_percentages) %in% top_taxa])
  } else {
    percentage <- sum(phylum_percentages[names(phylum_percentages) == phylum])
  }
  paste0(phylum, " (", round(percentage, 1), "%)")
})

# Updating phylum names in rowData for plotting with percentages in the legend
rowData(tse_phylum)$phylum <- phylum_labels[rowData(tse_phylum)$phylum]

# Plotting with y-axis as percentages and updated legend
plotAbundance(tse_phylum, assay.type = "relative_abundance", rank = "phylum",
              order_rank_by = "abund",
              order_sample_by = tse_phylum$Sample_ID) +
  scale_y_continuous(labels = scales::percent_format(scale = 100)) +
  labs(y = "Relative Abundance (%)", fill = "Phylum (Percentage)")








############## Shannon diversity




# Extracting only abundance data (excluding `age_category`)
abundance_data <- df_acg[, !names(df_acg) %in% "age_category"]

# Compute Shannon diversity index
shannon_values <- diversity(abundance_data, index = "shannon")

# Add Shannon values and age categories back to a data frame
shannon_df <- data.frame(Sample = rownames(df_acg), 
                         Shannon = shannon_values,
                         Age_Category = df_acg$age_category)





# Create a boxplot with significance annotations
ggplot(shannon_df, aes(x = Age_Category, y = Shannon, fill = Age_Category)) +
  geom_boxplot() +
  labs(
    x = "Age Category", 
    y = "Shannon Diversity Index") +
  theme_minimal() +
  geom_signif(
    comparisons = list(c("adult", "schoolage"), c("adult", "senior"), c("schoolage", "senior")), # Replace with your actual age categories
    map_signif_level = TRUE, # Automatically maps p-values to significance stars
    test = "wilcox.test", # Use Wilcoxon as a default for post-hoc tests
    step_increase = 0.1, # Space between annotations
    textsize = 3 # Size of significance stars
  ) +
  scale_fill_brewer(palette = "Set3")  # Apply the Set3 color palette







################ Beta diversity





# Extract only abundance data (excluding `age_category`)
abundance_data <- df_acg[, !names(df_acg) %in% "age_category"]

# Confirm that abundance data is numeric
abundance_data <- as.data.frame(lapply(abundance_data, as.numeric))


# Calculate Bray-Curtis distance matrix
dist_matrix <- vegdist(abundance_data, method = "bray")


# Perform PCoA
pcoa_result <- cmdscale(dist_matrix, k = 2, eig = TRUE)

# Extract coordinates for plotting
pcoa_df <- as.data.frame(pcoa_result$points)
colnames(pcoa_df) <- c("PCoA1", "PCoA2")
pcoa_df$age_category <- df_acg$age_category  # Add age categories for plotting



ggplot(pcoa_df, aes(x = PCoA1, y = PCoA2, color = age_category)) +
  geom_point(size = 1.8) +
  labs(
    x = "PCoA1", y = "PCoA2") +
  theme_minimal()


# Run PERMANOVA
permanova_result <- adonis2(dist_matrix ~ age_category, data = df_acg)
print(permanova_result)







################ Correlation analysis





# Assuming the correlation data frames have one column with correlation values
# Add dataset labels to each data frame
cor_df$Dataset <- "Relative Abundance"
cor_markab_df$Dataset <- "Marker Abundance"
cor_pathab_df$Dataset <- "Pathway Abundance"

# Combine the datasets into a single data frame
all_correlations <- rbind(
  data.frame(Correlation = cor_df[,1], Dataset = cor_df$Dataset),
  data.frame(Correlation = cor_markab_df[,1], Dataset = cor_markab_df$Dataset),
  data.frame(Correlation = cor_pathab_df[,1], Dataset = cor_pathab_df$Dataset)
)





# Define comparisons for significance testing
comparisons <- list(
  c("Relative Abundance", "Marker Abundance"),
  c("Relative Abundance", "Pathway Abundance"),
  c("Marker Abundance", "Pathway Abundance")
)

# Create the boxplot with significance bars
ggplot(all_correlations, aes(x = Dataset, y = Correlation, fill = Dataset)) +
  geom_boxplot() +
  labs(
    x = "Dataset",
    y = "Correlation") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3") +  # Optional color palette
  stat_compare_means(comparisons = comparisons, 
                     method = "wilcox.test", 
                     label = "p.signif",   # Use "p.signif" for significance stars or "p.format" for p-values
                     p.adjust.method = "bonferroni")  # Adjust for multiple comparisons








