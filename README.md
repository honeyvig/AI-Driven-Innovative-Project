# AI-Driven-Innovative-Project
We are a small but growing company operating two online shops: dobroje.rs and dodatnaoprema.com. We sell a wide range of products, but the competition in the local market is intense. To stay competitive, we are pivoting towards a more focused strategy, promoting five key items each week on sale.

We are looking to develop an AI-driven solution that will help us make smarter, data-informed decisions and streamline our promotional activities.

Analyze local market trends, prices, and competition
Understand our business dynamics through historical data (10 years of data will be provided that need to be stored in DB)
Conduct niche and logistics analysis
Recommend products to prioritize for weekly promotions based on data-driven insights
Continuously improve the AI agent’s suggestions based on ongoing sales data and performance metrics
===================
Here’s a Python implementation of an AI-driven system for your needs. The solution involves:

    Database setup and storage: Store historical data, market trends, and other relevant details.
    Data analysis and modeling: Use machine learning for trend analysis, pricing strategies, and weekly recommendations.
    Recommendation engine: Recommend products for weekly promotions.
    Feedback loop: Continuously update the model based on sales performance.

Key Components
1. Database Setup (SQLite for simplicity, can scale to MySQL/PostgreSQL)

import sqlite3

def setup_database():
    connection = sqlite3.connect("business_data.db")
    cursor = connection.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            product_name TEXT,
            sales INTEGER,
            revenue FLOAT,
            date DATE
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            competitor_price FLOAT,
            demand_index FLOAT,
            date DATE
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weekly_promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start DATE,
            product_id INTEGER,
            product_name TEXT,
            promotion_reason TEXT
        )
    """)
    
    connection.commit()
    connection.close()

setup_database()

2. Load Historical Data into the Database

import pandas as pd

def load_data(file_path, table_name):
    connection = sqlite3.connect("business_data.db")
    df = pd.read_csv(file_path)
    df.to_sql(table_name, connection, if_exists='append', index=False)
    connection.close()

# Example: load historical sales data
# load_data("historical_sales.csv", "historical_sales")

3. Analyze Trends and Competition

import pandas as pd

def analyze_market_trends():
    connection = sqlite3.connect("business_data.db")
    
    # Load data
    sales_data = pd.read_sql("SELECT * FROM historical_sales", connection)
    trends_data = pd.read_sql("SELECT * FROM market_trends", connection)
    
    # Merge and analyze
    merged_data = pd.merge(sales_data, trends_data, on=["product_id", "date"])
    merged_data['price_differential'] = merged_data['competitor_price'] - merged_data['revenue'] / merged_data['sales']
    
    # Add demand prediction (simplistic trend analysis)
    merged_data['demand_forecast'] = merged_data['demand_index'] * 1.05  # Adjust based on trends
    
    connection.close()
    return merged_data

trend_analysis = analyze_market_trends()
print(trend_analysis.head())

4. Recommendation Engine

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def recommend_products():
    connection = sqlite3.connect("business_data.db")
    data = pd.read_sql("SELECT * FROM historical_sales", connection)
    
    # Feature engineering
    data['month'] = pd.to_datetime(data['date']).dt.month
    features = data[['sales', 'revenue', 'month']]
    target = (data['sales'] > data['sales'].mean()).astype(int)  # 1 for high sales, 0 for low
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Test model
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    
    # Recommendations
    feature_importances = pd.Series(model.feature_importances_, index=features.columns)
    top_features = feature_importances.sort_values(ascending=False)
    print("Top features influencing sales:", top_features)
    
    connection.close()

recommend_products()

5. Weekly Promotions and Feedback Loop

def update_promotions():
    connection = sqlite3.connect("business_data.db")
    cursor = connection.cursor()
    
    # Use trend analysis and predictions to recommend products
    trend_analysis = analyze_market_trends()
    top_products = trend_analysis.sort_values(by='demand_forecast', ascending=False).head(5)
    
    # Insert weekly promotions
    for _, product in top_products.iterrows():
        cursor.execute("""
            INSERT INTO weekly_promotions (week_start, product_id, product_name, promotion_reason)
            VALUES (?, ?, ?, ?)
        """, (pd.Timestamp.now(), product['product_id'], product['product_name'], "High demand forecast"))
    
    connection.commit()
    connection.close()

update_promotions()

6. Feedback Loop for Model Improvement

def feedback_loop():
    connection = sqlite3.connect("business_data.db")
    
    # Load recent promotion data
    promotions = pd.read_sql("SELECT * FROM weekly_promotions", connection)
    sales = pd.read_sql("SELECT * FROM historical_sales", connection)
    
    # Merge data and analyze effectiveness
    merged = pd.merge(promotions, sales, on="product_id")
    merged['promotion_effectiveness'] = merged['sales'] / merged['sales'].mean()
    
    # Update model inputs or feature importance
    print(merged[['product_name', 'promotion_effectiveness']])
    
    connection.close()

feedback_loop()

Deployment Notes

    Historical Data: Load your 10 years of sales and market trends into the database.
    Scheduled Automation: Use tools like cron jobs or Celery for weekly analysis and promotion updates.
    User Interface: Consider a simple web app with frameworks like Flask or Django to view and adjust weekly recommendations.
    Scalability: For larger datasets, consider using PostgreSQL or MongoDB.

This code provides a comprehensive framework for automating your promotional decisions using AI-driven insights. 
