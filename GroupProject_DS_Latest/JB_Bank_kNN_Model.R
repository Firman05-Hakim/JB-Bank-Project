# =========================================================================
# STEP 1: TOOL SETUP & DATA IMPORT
# =========================================================================

# A. Install the necessary packages 
# (Highlight these 4 lines and run them ONLY ONCE to download the tools)
install.packages("readxl")
install.packages("caret")
install.packages("corrplot")
install.packages("kknn")

# B. Load the toolboxes into RStudio
# (Highlight these 4 lines and run them every time you open this project)
library(readxl)     # Required to read the .xls file
library(caret)      # Required for data splitting and cross-validation
library(corrplot)   # Required for feature selection visualization
library(kknn)       # Required for the weighted distance metric

# C. Import the dataset
# We tell R to look at the "Data" sheet specifically, 
# and skip the first 2 rows so it grabs the correct column names.
bank_data <- read_excel("HypotheticalBank.xls", sheet = "Data", skip = 2)

# D. View the first 6 rows to confirm it loaded correctly
head(bank_data)



# =========================================================================
# STEP 2: FEATURE SELECTION & VISUALIZATION (Requirement a)
# =========================================================================

# 1. Clean the Data (Logic-based removal)
# We remove 'ID' because it's just a label. 
# We remove 'ZIP Code' because the assignment strictly said "Do not use".
bank_clean <- bank_data[ , !(names(bank_data) %in% c("ID", "ZIP Code"))]

# 2. Calculate Correlation
# This builds a table showing how every variable relates to the "Personal Loan"
cor_matrix <- cor(bank_clean)

# 3. Visualization (Requirement: "Apply good visualization techniques")
# This creates a color-coded chart of the correlations.
corrplot(cor_matrix, method = "number", type = "lower", 
         tl.col = "black", tl.srt = 45, tl.cex = 0.8,
         title = "Correlation Matrix of Bank Customer Predictors",
         mar = c(0,0,2,0))


# =========================================================================
# STEP 3: DATA PARTITIONING & NORMALIZATION (Strictly 60/40)
# =========================================================================

# 1. Final Attribute Elimination (Requirement a)
# Removing Experience due to 0.99 correlation with Age.
bank_final <- bank_clean[ , !(names(bank_clean) %in% c("Experience"))]

# 2. Ensure Target is a Factor
bank_final$`Personal Loan` <- as.factor(bank_final$`Personal Loan`)

# 3. Data Partitioning (60% Training / 40% Validation)
set.seed(123)
train_index <- createDataPartition(bank_final$`Personal Loan`, p = 0.60, list = FALSE)

train_data_raw <- bank_final[train_index, ]
valid_data_raw <- bank_final[-train_index, ]

# 4. Normalization (Scaling)
# We calculate the scale on training data and apply to both sets.
norm_model <- preProcess(train_data_raw, method = c("center", "scale"))
train_data <- predict(norm_model, train_data_raw)
valid_data <- predict(norm_model, valid_data_raw)

# =========================================================================
# STEP 4: OPTIMAL K & EUCLIDEAN MODEL (Requirement b & c)
# =========================================================================

# 1. 10-Fold Cross-Validation Setup
train_control <- trainControl(method = "cv", number = 10)

# 2. Find Best K (Testing 1 to 20)
set.seed(123)
knn_model_euclidean <- train(`Personal Loan` ~ ., 
                             data = train_data, 
                             method = "knn",
                             trControl = train_control,
                             tuneGrid = expand.grid(k = 1:20))

# 3. Visualization for Justification
plot(knn_model_euclidean, main="Selection of Optimal k using Cross-Validation")

# 4. Final Euclidean Accuracy and Matrix
best_k <- knn_model_euclidean$bestTune$k
euclidean_preds <- predict(knn_model_euclidean, newdata = valid_data)

print("--- EUCLIDEAN MODEL RESULTS ---")
confusionMatrix(euclidean_preds, valid_data$`Personal Loan`, positive = "1")

# =========================================================================
# STEP 5: WEIGHTED DISTANCE MODEL (Requirement d)
# =========================================================================

# 1. Weighted k-NN using kknn package
set.seed(123)
knn_weighted <- kknn(`Personal Loan` ~ ., 
                     train = train_data, 
                     test = valid_data, 
                     k = best_k, 
                     kernel = "optimal")

weighted_preds <- fitted(knn_weighted)

print("--- WEIGHTED MODEL RESULTS ---")
confusionMatrix(as.factor(weighted_preds), valid_data$`Personal Loan`, positive = "1")

