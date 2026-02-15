# Customer Segmentation Predictor

A simple and interactive web application that predicts which **customer segment** a person belongs to using a pre-trained **K-Means clustering** model.

The model classifies customers based on behavioral and demographic features such as income, recency, spending, purchase channels, campaign response, etc.

![Customer Segmentation Demo](https://customersegmentation-ksdnnnczmvangmiymqmujx.streamlit.app/)  
*(Replace this placeholder with a real screenshot of your deployed app after uploading one)*

## Features

- Interactive input form for 10 key customer features
- Real-time prediction using scaled K-Means model
- Clear segment name display (e.g. "High Engagement Customers")
- Shows cluster ID and input summary
- Error handling for invalid inputs
- Clean, modern UI built with Streamlit

## Live Demo

👉 https://your-app-name.streamlit.app/  
*(Update this link once you deploy it on Streamlit Cloud)*

## Tech Stack

- **Frontend / Framework**: Streamlit
- **Machine Learning**: scikit-learn (K-Means)
- **Data Processing**: pandas, numpy
- **Model & Scaler Persistence**: joblib (.pkl files)

## Project Structure
