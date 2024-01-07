from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise import accuracy
import requests
from bs4 import BeautifulSoup

# Sample dataset (replace it with your own data)
data = {
    'user_ids': ['User1', 'User2', 'User3'],
    'item_ids': ['Item1', 'Item2', 'Item3'],
}

# Create a Pandas DataFrame from the data
import pandas as pd
df = pd.DataFrame(data)

# Web scraping to obtain additional item information (weather ratings)
def scrape_weather_rating(city_name, api_key):
    # Make a request to the OpenWeatherMap API
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}'
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        weather_data = response.json()

        # Access weather information from the response
        temperature = weather_data['main']['temp']
        # Use temperature (or any other weather-related parameter) as the rating
        weather_rating = temperature
        return {'city_name': city_name, 'weather_rating': weather_rating}
    else:
        print(f'Failed to retrieve weather data for {city_name}. Status code: {response.status_code}')
        return None

# Example: Scraping weather ratings for cities
weather_ratings = []
api_key = 'ecac894b998a828ffffa26df26a5756d'  # Replace with your actual OpenWeatherMap API key

for index, row in df.iterrows():
    city_name = row['item_ids']
    weather_info = scrape_weather_rating(city_name, api_key)
    
    if weather_info:
        weather_ratings.append(weather_info)

# Convert the scraped data to a DataFrame
weather_df = pd.DataFrame(weather_ratings)

# Check if 'weather_rating' column exists in the DataFrame
if 'weather_rating' in weather_df.columns:
    # Define the Reader
    reader = Reader(rating_scale=(weather_df['weather_rating'].min(), weather_df['weather_rating'].max()))

    # Load the dataset
    dataset = Dataset.load_from_df(weather_df[['city_name', 'weather_rating']], reader)

    # Split the dataset into training and testing sets
    trainset, testset = train_test_split(dataset, test_size=0.2)

    # Build the recommendation model (SVD algorithm)
    model = SVD()
    model.fit(trainset)

    # Make predictions on the test set
    predictions = model.test(testset)

    # Evaluate the accuracy of the model
    accuracy.rmse(predictions)

    # Example: Get top-N recommendations for a city
    city_name = 'Item1'

    # Predict the weather rating for the city
    prediction = model.predict(city_name, None)
    print(f"Predicted weather rating for {city_name}: {prediction.est:.2f}")
else:
    print("The 'weather_rating' column is missing in the DataFrame.")
