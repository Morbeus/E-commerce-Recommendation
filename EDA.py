import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Create EDA folder if it doesn't exist
if not os.path.exists('EDA'):
    os.makedirs('EDA')

# Read the JSON data (each line is a separate JSON object)
data = []
for line in open('Dataset for DS Case Study.json', 'r'):
    data.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(data)

# Basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nBasic Information:")
print(df.info())

# Basic statistics for numeric columns
print("\nBasic Statistics:")
print(df.describe())

# Count unique values (excluding 'helpful' column)
print("\nUnique Values per Column:")
for col in df.columns:
    if col != 'helpful':  # Skip the 'helpful' column
        print(f"{col}: {df[col].nunique()} unique values")

# Rating distribution
plt.figure(figsize=(10, 6))
df['overall'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('EDA/rating_distribution.png', bbox_inches='tight')
plt.close()

# Average rating by product (ASIN)
avg_rating_by_product = df.groupby('asin')['overall'].agg(['mean', 'count']).sort_values('count', ascending=False)
print("\nAverage Rating by Product (Top 5 most reviewed):")
print(avg_rating_by_product.head())

# Calculate review lengths first
df['review_length'] = df['reviewText'].str.len()

# Analyze review lengths
plt.figure(figsize=(10, 6))
sns.boxplot(x='overall', y='review_length', data=df)
plt.title('Review Length by Rating')
plt.xlabel('Rating')
plt.ylabel('Review Length (characters)')
plt.savefig('EDA/review_length_by_rating.png', bbox_inches='tight')
plt.close()

# Helpful votes analysis (with safe division)
df['helpful_ratio'] = df['helpful'].apply(lambda x: x[0]/x[1] if x[1] != 0 else 0)
plt.figure(figsize=(10, 6))
sns.boxplot(x='overall', y='helpful_ratio', data=df)
plt.title('Helpfulness Ratio by Rating')
plt.xlabel('Rating')
plt.ylabel('Helpful Ratio')
plt.savefig('EDA/helpfulness_by_rating.png', bbox_inches='tight')
plt.close()

# Time series analysis
df['review_date'] = pd.to_datetime(df['unixReviewTime'], unit='s')
reviews_over_time = df.groupby('review_date')['overall'].count().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(reviews_over_time['review_date'], reviews_over_time['overall'])
plt.title('Number of Reviews Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.savefig('EDA/reviews_over_time.png', bbox_inches='tight')
plt.close()

# Save the statistical information to a text file
with open('EDA/statistical_summary.txt', 'w') as f:
    f.write("Dataset Shape: {}\n\n".format(df.shape))
    f.write("Columns: {}\n\n".format(df.columns.tolist()))
    f.write("Basic Statistics:\n")
    f.write(df.describe().to_string())
    f.write("\n\nUnique Values per Column:\n")
    for col in df.columns:
        if col != 'helpful':
            f.write(f"{col}: {df[col].nunique()} unique values\n")
    f.write("\nTop 5 Most Reviewed Products:\n")
    f.write(avg_rating_by_product.head().to_string())
