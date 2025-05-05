# NutriScan

## Overview
NutriScan is a web application that identifies food from images and provides detailed nutritional information. Simply upload a photo of your food, and the app will predict what it is and display a comprehensive nutritional breakdown.

## Features
- Food identification using deep learning (ResNet18 model)
- Nutritional information retrieval from MongoDB database
- Detailed breakdown of calories, macronutrients, and micronutrients
- Visual representation of macronutrient distribution via pie charts
- User-friendly interface built with Streamlit

## Current Capabilities
- Recognizes 3 food categories: Biryani, Fries, and BBQ
- Provides serving size information
- Displays comprehensive nutritional information including:
  - Calories
  - Protein, Fat, Carbohydrates
  - Sugars, Dietary Fiber
  - Vitamins & Minerals (Sodium, Iron, Calcium)

## Technical Implementation
- Built with PyTorch and Streamlit
- Uses a fine-tuned ResNet18 architecture
- MongoDB backend for nutritional data storage
- Matplotlib for data visualization

## Installation
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Set up environment variables in `.env`:
   ```
   MONGO_URI=your_mongo_connection_string
   DB_NAME=your_database_name
   COLLECTION_NAME=your_collection_name
   ```
4. Run the app: `streamlit run app.py`

## Future Development
- Implementing object detection for multiple foods in one image
- Expanding the food database to include more categories
- Adding personalized nutritional recommendations
