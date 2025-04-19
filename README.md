# AWS SageMaker MLOps POC: House Price Prediction for Kitchener and Waterloo, Ontario

## Objective
Predict house sale prices specifically for the Kitchener and Waterloo regions in Ontario, Canada using AWS SageMaker and MLOps best practices, including:
Data wrangling
Feature engineering with SageMaker Feature Store
Model training & registration
Real-time & batch inference

## AWS & SageMaker Setup

### 1. Set up AWS Environment

Create an AWS account if you don’t already have one.
Sign in to the AWS Console and go to Amazon SageMaker.
In the SageMaker Console, go to SageMaker Studio > Domains > click Create domain.
Choose SageMaker Studio as the interface.
Select IAM as the authentication method.
Name your domain (e.g., mlops-kitchener-housing).
Create a new user profile (e.g., mluser).
Choose or create an IAM execution role with the following permissions:
AmazonSageMakerFullAccess
AmazonS3FullAccess
AmazonSageMakerFeatureStoreAccess
AmazonSageMakerModelRegistryAccess
CloudWatchFullAccess

Click Submit and wait for the domain to be created.

### 2. Launch SageMaker Studio

After domain creation, go to SageMaker Studio > User Profiles > click Launch next to your user.
This will open SageMaker Studio in a new browser tab.

### 3. Create an S3 Bucket

Go to the S3 console
Create a new bucket: sagemaker-kitchener-waterloo-housing
Enable versioning (optional but recommended)

### 4. Download & Upload Dataset

Download Kitchener-Waterloo real estate data:
From portals like Realtor.ca or City of Kitchener Open Data / City of Waterloo Open Data
Or use sample data such as the Ames Housing dataset
Upload CSV to S3 bucket under the raw/ folder:
s3://sagemaker-kitchener-waterloo-housing/raw/kitchener_waterloo_housing.csv

### 5. Load Dataset in SageMaker Studio

In a new Jupyter notebook inside SageMaker Studio, run the following code to load the dataset from S3:
```
import pandas as pd
import boto3
from io import StringIO

# Define S3 bucket and key
bucket = 'sagemaker-kitchener-waterloo-housing'
key = 'raw/kitchener_waterloo_housing.csv'

# Create S3 client and get object
s3_client = boto3.client('s3')
response = s3_client.get_object(Bucket=bucket, Key=key)

# Read the CSV into a DataFrame
body = response['Body'].read().decode('utf-8')
df = pd.read_csv(StringIO(body))

# Preview the dataset
df.head()
```
### Step-by-Step Instructions

1. Data Wrangling (01_data_wrangling.ipynb)
```
Load Data
import pandas as pd

# Load Ames Housing dataset
df = pd.read_csv("AmesHousing.csv")
df.head()

Clean Missing Values

# Identify missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# Drop columns with too many missing values
df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], inplace=True)

# Fill numeric columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = df[cat_cols].fillna('None')
```

Feature Engineering
```
# Encode categorical features
from sklearn.preprocessing import LabelEncoder

for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Normalize selected numerical features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numeric_features = ['GrLivArea', 'LotArea', 'YearBuilt', 'TotalBsmtSF']
df[numeric_features] = scaler.fit_transform(df[numeric_features])
```
Save Cleaned Dataset
```
import boto3
import io

csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
s3_resource = boto3.resource('s3')
s3_resource.Object('sagemaker-kitchener-waterloo-housing', 'processed/kitchener_waterloo_cleaned.csv').put(Body=csv_buffer.getvalue())
```

2. Feature Store (02_feature_store_ingestion.ipynb)

Define a feature group using property_id and timestamp
Load the processed data from S3
Use FeatureGroup class from sagemaker.feature_store
Store features in both online and offline Feature Store
