## Background:
Telecom XYZ's Marketing Division has a budget constraint for promoting a new consumer offer. They've requested the Customer Value Management (CVM) team to strategically drive the promotion.

## Current Situation:
The CVM team tested the offer on a random sample of 2,240 customers.

## Objective:
As the Machine Learning Manager, develop a machine learning model using the experiment results to identify the most receptive customers from the full customer database for the new offer. 
This will enable the team to maximize campaign effectiveness by targeting the most likely customers to adopt the offer and optimize marketing spend by focusing on a high-potential segment.

## Dataset Attributes:
-    `Id :` Customer’s unique identifier
-    `Year_birth:` Customer’s birth year
-   ``Consumer_seg:`` Market segment fo Customer
-    ``Region:`` Customer’s geographical location
-    ``Cltv:`` Customer lifetime Value
-    ``Activation_dt:`` Date customer joined the Telco Operator
-    ``Recency:`` Number of days since last purchase.
-    ``Device_type :`` Smartphone or Non Smartphone flag
-    ``Data_rev_6mths:`` Data Revenue in last 6 months.
-    ``Voice_rev_6mths:`` Voice Revenue in last 6 months
-    ``Tot_rev_6mths:`` Total Revenue in last 6 months .
-    ``Total_purchases:`` Total purchase count of all products in the last 6 months
-    ``Total_engagements:`` Count of engagements across all channels whether offer was suuscessfully purchased on not
-    ``Digital_channel_purchases:`` Count of purchases done via digital channels
-    ``Ussd_channel_purchases:`` Count of purchases done via USSD Channel
-    ``Other_channels_purchases:`` Count of purchases done via Other Channels
-    ``PrevResponseCnt :`` Count of responses for previous promotional campaign.
-    ``Complaint_cnt:`` Count of previous complaints
-    ``Response(Target Label) :`` Purchase or Not based on the new offer.


`I choose Random Forest because of it a versatile and powerful algorithm that is commonly used for classification tasks due to its ability to handle various data types, robustness to overfitting, and interpretability through feature importance analysis.`
