# NutriScan

## Overview
NutriScan is a web application that identifies food from images and provides detailed nutritional information. Simply upload a photo of your food, and the app will predict what it is and display a comprehensive nutritional breakdown.

## Features
- Food identification using deep learning via Azure ML endpoints
- Nutritional information retrieval from MongoDB database
- Detailed breakdown of calories, macronutrients, and micronutrients
- Visual representation of macronutrient distribution via pie charts
- User-friendly interface built with Streamlit

## Current Capabilities
- Recognizes 6 food categories:
  - Biryani
  - Milk Tea (Chai)
  - Chapati
  - Chicken Tikka
  - Paratha
  - Samosa
- Provides serving size information
- Displays comprehensive nutritional information including:
  - Calories
  - Protein, Fat, Carbohydrates
  - Sugars, Dietary Fiber
  - Vitamins & Minerals (Sodium, Iron, Calcium)
- Confidence score for food identification predictions

## Technical Implementation
- Built with Streamlit for the user interface
- Uses Azure ML endpoints for image classification model inference
- MongoDB backend for nutritional data storage
- Matplotlib for data visualization (pie charts for macronutrients)

## Configuration
To use this application, you'll need to set up the following environment variables:
```
MONGO_URI=your_mongo_connection_string
DB_NAME=your_database_name
COLLECTION_NAME=your_collection_name
AZURE_ENDPOINT_URI=your_azure_endpoint_uri
AZURE_PRIMARY_KEY=your_azure_primary_key
AZURE_DEPLOYMENT_NAME=your_deployment_name
```

## Azure Integration
The machine learning model has been deployed on Azure, making it accessible through Azure ML endpoints. This provides:
- Scalable image classification capabilities
- Efficient inference through Azure's machine learning infrastructure
- Streamlined integration with the Streamlit frontend

## How It Works
1. Upload an image of supported food types
2. The image is sent to Azure ML endpoint for classification
3. Results are displayed with confidence score
4. Nutritional information is retrieved from MongoDB using text search
5. Detailed nutritional breakdown and visualizations are generated

## Future Development
- Implementing object detection for multiple foods in one image
- Expanding the food database to include more categories
- Adding personalized nutritional recommendations