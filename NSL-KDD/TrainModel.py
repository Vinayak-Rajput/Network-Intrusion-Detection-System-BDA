# TrainModel.py (Fixed Version)

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pathlib import Path  # <-- IMPORT THIS

# --- 0. Setup Spark Session ---
print("Starting Spark session...")
spark = SparkSession.builder.appName("NIDS_Case_Study").getOrCreate()
spark.sparkContext.setLogLevel("ERROR") # Hides all the INFO spam

# --- 1. Define File Paths (The Fix) ---
# Get the directory where this Python script is located
script_dir = Path(__file__).parent

# Define file paths *relative to the script's directory*
train_csv = script_dir / 'KDDTrain_with_headers.csv'
test_csv = script_dir / 'KDDTest_with_headers.csv' # Make sure you've created this file!

# --- 2. Load Data ---
print(f"Loading data from '{train_csv}' and '{test_csv}'...")

# Use pandas to get column names first
try:
    pdf = pd.read_csv(train_csv)
    all_columns = pdf.columns.tolist()
except FileNotFoundError:
    print(f"ERROR: '{train_csv}' not found.")
    print("Did you run CsvConversion.py on 'KDDTrain+.TXT'?")
    spark.stop()
    exit()

# Load data into Spark, inferring the schema (column types)
train_df = spark.read.csv(str(train_csv), header=True, inferSchema=True)

try:
    test_df = spark.read.csv(str(test_csv), header=True, inferSchema=True)
except Exception as e:
    print(f"ERROR: Could not load '{test_csv}'.")
    print("Please make sure you have run CsvConversion.py on 'KDDTest+.TXT'.")
    spark.stop()
    exit()

print("Data loaded into Spark.")

# --- 3. Preprocessing / Feature Engineering ---
# (The rest of the script is the same as before)
print("Starting data preprocessing...")

# Identify categorical and numeric columns (excluding labels)
categorical_cols = ['protocol_type', 'service', 'flag']
label_cols = ['label', 'difficulty_score'] 
numeric_cols = [c for c in all_columns if c not in categorical_cols + label_cols]

# A. Create binary label (1.0 for 'attack', 0.0 for 'normal')
train_df = train_df.withColumn("binary_label", 
                               when(col("label") == "normal", 0.0).otherwise(1.0))
test_df = test_df.withColumn("binary_label", 
                             when(col("label") == "normal", 0.0).otherwise(1.0))

# B. Create StringIndexer stages for categorical columns
index_output_cols = [c + "_index" for c in categorical_cols]
string_indexers = [
    StringIndexer(inputCol=col, outputCol=out_col, handleInvalid="keep")
    for col, out_col in zip(categorical_cols, index_output_cols)
]

# C. Assemble all features into a single vector
feature_assembler_inputs = numeric_cols + index_output_cols
vector_assembler = VectorAssembler(
    inputCols=feature_assembler_inputs,
    outputCol="raw_features"
)

# D. Scale the features
scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withStd=True,
    withMean=True
)

# --- 4. Define the Model (Analytical Method) ---
rf_classifier = RandomForestClassifier(
    labelCol="binary_label",
    featuresCol="features",
    numTrees=100
)

# --- 5. Create the BDA Pipeline ---
pipeline = Pipeline(stages=string_indexers + [vector_assembler, scaler, rf_classifier])

# --- 6. Train the Model ---
print("Starting model training... (This may take a few minutes)")
model = pipeline.fit(train_df)
print("Model training complete.")

# --- 7. Evaluate the Model (Expected Outcomes / KPIs) ---
print("Evaluating model on test data...")
predictions = model.transform(test_df)

# Show a sample of predictions
print("\n--- Sample Predictions ---")
predictions.select("label", "binary_label", "prediction", "probability").show(10)

# --- Calculate KPIs ---
print("\n--- Model Performance (KPIs) ---")

# 1. Accuracy
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="binary_label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_acc.evaluate(predictions)
print(f"Accuracy = {accuracy * 100:.2f}%")

# 2. Precision (for the 'attack' class, label 1.0)
evaluator_pr = MulticlassClassificationEvaluator(
    labelCol="binary_label", predictionCol="prediction", metricName="precisionByLabel", metricLabel=1.0)
precision = evaluator_pr.evaluate(predictions)
print(f"Precision (for attacks) = {precision:.4f}")

# 3. Recall (for the 'attack' class, label 1.0)
evaluator_re = MulticlassClassificationEvaluator(
    labelCol="binary_label", predictionCol="prediction", metricName="recallByLabel", metricLabel=1.0)
recall = evaluator_re.evaluate(predictions)
print(f"Recall (for attacks) = {recall:.4f}")

# 4. F1-Score
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="binary_label", predictionCol="prediction", metricName="f1", metricLabel=1.0)
f1 = evaluator_f1.evaluate(predictions)
print(f"F1-Score (for attacks) = {f1:.4f}")

print("\nCase study model pipeline is complete.")
spark.stop()