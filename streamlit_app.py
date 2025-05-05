import streamlit as st
from PIL import Image
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import certifi
import io
import requests
import base64

# Load environment variables
load_dotenv(override=True)

# MongoDB connection
@st.cache_resource
def get_mongo_connection():
    client = AsyncIOMotorClient(os.getenv("MONGO_URI") , tlsCAFile=certifi.where())
    return client

async def get_nutrition_data(food_name):
    client = await get_mongo_connection()
    db = client[os.getenv("DB_NAME")]
    collection = db[os.getenv("COLLECTION_NAME")]

    # Text search query using the index
    query = {
        "$text": {
            "$search": food_name,
            "$caseSensitive": False,
            "$diacriticSensitive": False
        }
    }
    
    # Get results sorted by relevance score
    results = list(collection.find(query, {"score": {"$meta": "textScore"}})
                         .sort([("score", {"$meta": "textScore"})])
                         .limit(5))

    # Clean documents and add similarity score
    cleaned = []
    for doc in results:
        doc.pop('_id', None)
        doc.pop('__v', None)
        cleaned.append({
            **doc,
            "match_score": doc.get("score", 0)
        })
    
    return cleaned

# Azure ML endpoint configuration
@st.cache_data
def get_endpoint_info():
    """Get endpoint information from environment variables or config file"""
    endpoint_uri = os.getenv("AZURE_ENDPOINT_URI")
    primary_key = os.getenv("AZURE_PRIMARY_KEY")
    
    return endpoint_uri, primary_key

class_names = ['Biryani', 'Milk Tea', 'Chapati', 'Chicken tikka', 'Paratha', 'Samosa']


# Function to predict using Azure ML endpoint
def predict_with_endpoint(image):
    endpoint_uri, primary_key = get_endpoint_info()
    
    if not endpoint_uri or not primary_key:
        st.error("Azure ML endpoint configuration is missing. Please set AZURE_ENDPOINT_URI and AZURE_PRIMARY_KEY environment variables.")
        return None, 0
    
    # Load and convert the image
    image = image.convert("RGB")  # Ensure no alpha channel for JPEG

    # Convert to bytes and then base64 encode
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    image_bytes = img_byte_arr.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')  # Encode as base64 string
    
    # Prepare headers and structured JSON payload
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {primary_key}",
        "azureml-model-deployment": os.getenv("AZURE_DEPLOYMENT_NAME")
    }
    
    payload = {
        "input_data": {
            "data": image_base64
        }
    }
    
    try:
        # Send request to the endpoint with JSON payload
        response = requests.post(
            endpoint_uri, 
            headers=headers, 
            json=payload  # Use json parameter to properly serialize
        )
        
        # Check if request was successful
        if response.status_code == 200:
            try:
                result = response.json()
                if "predicted_class" in result and "confidence" in result:
                    return result["predicted_class"], result["confidence"]
                else:
                    st.error(f"Unexpected response format: {result}")
                    return None, 0
            except Exception as e:
                st.error(f"Failed to parse response JSON: {e}")
                return None, 0
        else:
            st.error(f"Error from endpoint: {response.status_code} - {response.text}")
            return None, 0
    except Exception as e:
        st.error(f"Error connecting to endpoint: {e}")
        return None, 0


def display_nutrition(data):
    # Display header information
    st.header(f"🍴 {data['name']}")
    st.subheader(f"Serving Size: {data['servingSize']}")
    
    # Create columns layout
    col1, col2, col3 = st.columns(3)
    
    with st.expander("View Nutrition Breakdown!"):
    # Calories metric
        col1.metric(
            label="Calories",
            value=data['calories']['amount'],
            help="Total calories per serving"
        )
        
        # Key Nutrients
        with col2:
            st.metric(
                label="Total Fat",
                value=data['nutrients']['Total Fat']['amount'],
                help="Daily Value percentage"
            )
            
        with col3:
            st.metric(
                label="Protein",
                value=data['nutrients']['Protein']['amount'],
                help="Protein content"
            )
        
        # Second row of metrics
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.metric(
                label="Carbohydrates",
                value=data['nutrients']['Total Carbohydrates']['amount'],
            )
            
        with col5:
            st.metric(
                label="Sugars",
                value=data['nutrients']['Sugars']['amount'],
            )
            
        with col6:
            st.metric(
                label="Dietary Fiber",
                value=data['nutrients']['Dietary Fiber']['amount'],
               
            )
        
        # Third row for vitamins/minerals
        st.subheader("Vitamins & Minerals")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            st.metric("Sodium", 
                    data['nutrients']['Sodium']['amount'],
                    )
            
        with col8:
            st.metric("Iron",
                    data['nutrients']['Iron']['amount'],
                    )
            
        with col9:
            st.metric("Calcium",
                    data['nutrients']['Calcium']['amount'],
                    )
    
    # Add visual separator
    st.markdown("---")


def create_nutrition_pie(data):
    # Extract values
    protein = float(data['nutrients']['Protein']['amount'].replace('g', ''))
    carbs = float(data['nutrients']['Total Carbohydrates']['amount'].replace('g', ''))
    fat = float(data['nutrients']['Total Fat']['amount'].replace('g', ''))
    
    # Calculate percentages
    total = protein + carbs + fat
    sizes = [protein/total*100, carbs/total*100, fat/total*100]
    
    # Create chart
    fig, ax = plt.subplots()
    ax.pie(sizes, 
           labels=['Protein', 'Carbs', 'Fat'],
           colors=['#ff9999','#66b3ff','#99ff99'],
           autopct='%1.1f%%',
           startangle=90)
    ax.axis('equal')
    
    return fig

# Streamlit app
def main():
    st.title("NutriScan")
    st.info("Upload a food image, get its name and nutritional breakdown, and visualize macronutrients with an interactive pie chart, all in one place!")
    
    st.sidebar.caption("Made with 💖 by Shaheer Jamal")

    st.sidebar.header("Currently Supported Food Types")
    st.sidebar.write("""
                      1. Biryani
                      2. Chai
                      3. Chapati
                      4. Chicken Tikka
                      5. Paratha
                      6. Samosa
                     """)
    
    uploaded_image = st.sidebar.file_uploader("Upload Image Currently Supported Food Types!",type=["jpg", "jpeg", "png"])
                
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.sidebar.image(image,caption="Uploaded image", width=300) 
        # Preprocess and predict
        predicted_class, confidence = predict_with_endpoint(image)

        if predicted_class is None:
            st.error("Prediction failed! Please upload image of supported food types.")
            return
        with st.spinner('Analyzing Image...'):
            st.caption("Predicted Food Name")
            st.success(f"**{predicted_class}**")
            st.progress(confidence, text=f"Confidence Score {confidence:.1%}")

        with st.spinner("Searching database..."):
            results = get_nutrition_data(predicted_class)
            
            if not results:
                st.error("Food not found in database!")
            else:
                if results:
                    display_nutrition(results[0])
        
                    with st.expander("Visualize Macro Nutrients!"):
                        st.pyplot(create_nutrition_pie(results[0]))
               
            
if __name__ == '__main__':
    food = main()