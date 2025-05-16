# Apple stocks predict
![Image](https://github.com/user-attachments/assets/1e843a30-f7dd-415a-a303-56e94796cb5b)

---

## I. Introduction
The stock market plays a critical role in the global economy. Predicting stock prices is a valuable tool that supports investors in making informed decisions and helps institutions manage financial risks effectively.
This project focuses on the collection, processing, and analysis of Apple Inc. (AAPL) stock price data from 2015 to the present using PySpark for scalable data processing. The goal is to clean and prepare the data, compute key technical indicators, and visualize price trends as a foundation for future modeling and prediction.

---

## II. Project Objectives

- ✅ Collect historical stock price data for AAPL from Yahoo Finance.
- ✅ Process the data using PySpark to ensure scalability and efficiency.
- ✅ Calculate and visualize basic technical indicators (e.g., moving averages).
- ✅ Develop predictive models for stock price forecasting (LSTM, Gradient Boost Tree Regressor)

---

## III. Technologies Used

| Tool        | Purpose                                |
|-------------|----------------------------------------|
| `Python`    | Main programming language              |
| `PySpark`   | Big data processing                    |
| `yfinance`  | Download historical data from Yahoo Finance |
| `matplotlib`| Data visualization                     |
| `pandas`    | Lightweight data manipulation          |

---

## IV. Comprehensive Technical and Statistical Analysis of AAPL Stock
### 🔍  Data Source and Collection
The historical stock data of Apple is retrieved from Yahoo Finance, starting from January 1, 2015 to the current date. The dataset includes daily stock metrics such as Open, High, Low, Close, Volume, and Adjusted Close. After downloading, the data is structured into a tabular format with a clean and standardized schema, ready for transformation and analysis.
### ⚙️ Data Preparation and Cleaning
Before computing indicators, the data undergoes preprocessing steps to ensure compatibility and efficiency within the PySpark environment. This includes:
  - Converting the Date column to a proper date type to enable time-based operations.
  - Selecting only the relevant columns (Date and Close) to reduce memory usage and simplify computations.
  - Sorting the dataset chronologically to maintain time series integrity.
This preparation ensures that large volumes of stock data can be processed efficiently, maintaining accuracy and speed.
### 📊 Exploratory Data Analysis (EDA) on AAPL Stock Data
#### 1.Objective
This exploratory analysis focuses on historical stock data for Apple Inc. (AAPL), spanning from 2012 to 2025. The aim is to clean and transform the data, inspect basic statistics, and generate insightful visualizations to better understand the behavior and distribution of key features such as closing price, daily return, and feature correlations.
#### 2. Data Cleaning and Preparation
- MultiIndex Flattening: The original dataset (from Yahoo Finance) sometimes includes a multi-level column index. This is flattened to retain only the top-level column names for simplicity and clarity.
- Column Filtering: Only essential columns are retained: Date, Open, High, Low, Close, and Volume.
- Type Conversion: After loading the data into a Spark DataFrame, all numeric fields are explicitly cast to DoubleType to ensure numerical operations run smoothly. The Date column is parsed into proper date format (yyyy-MM-dd) for sorting and time-series operations.
#### 3. Schema and Sample Data
- The schema inspection confirms that all numeric fields (Open, High, Low, Close, Volume) are of type Double, while the Date column is correctly formatted as DateType.
- Sample data preview shows five rows of the dataset with daily stock prices and trading volume, verifying that the data has been correctly formatted and loaded.
#### 4. Summary Statistics
![Image](https://github.com/user-attachments/assets/6ff030df-a1ff-4223-a0de-be5d6cd1fbde)

![Image](https://github.com/user-attachments/assets/8263dd2e-0865-46ca-8556-0882406c7cd9)

A summary of the dataset's descriptive statistics reveals:
  - Price Ranges: The minimum and maximum values for Open, High, Low, and Close give insights into how AAPL's stock price has evolved over time.
  - Volume Insight: Daily trading volume varies significantly, indicating fluctuating investor activity.
  - Missing Values: Initial summary checks indicate no major missing values, ensuring readiness for downstream modeling.
#### 5. Feature Engineering
Daily Return Calculation: A new column Return is created, representing the percentage change in closing price from the previous day. This is essential for understanding stock volatility and behavior.
#### 6. Data Visualization
##### a. Closing Price Over Time
![Image](https://github.com/user-attachments/assets/fbf74652-9b98-402f-a66e-852562da904e)

- Chart Description: A time-series line plot shows the evolution of AAPL’s closing stock price from 2012 to 2025.
- Insight: The chart highlights clear upward trends with some notable corrections, aligning with real-world financial events. The long-term growth trajectory reflects Apple’s strong performance.
##### b. Histogram of Daily Returns
![Image](https://github.com/user-attachments/assets/1c474492-1877-4ddf-a36c-d8ad7a977351)

- Chart Description: A histogram depicts the distribution of daily returns.
- Insight: The distribution is approximately centered around 0 with a slight right skew. Most returns fall within a small range, but the tails indicate occasional high volatility (e.g., earnings announcements or market shocks).
##### c. Correlation Heatmap
![Image](https://github.com/user-attachments/assets/56b5d9ca-0328-4db1-922f-d3541137c18e)

- Chart Description: A heatmap showing pairwise Pearson correlations between the key numeric variables (Open, High, Low, Close, Volume, and Return).
- Insight:
  - Strong positive correlations between Open, High, Low, and Close prices suggest these features move together, which is typical for stock data.
  - Volume shows weak correlation with prices, indicating that price changes do not necessarily depend on the number of shares traded.
  - Return shows relatively low correlation with price levels, emphasizing its independence as a feature capturing short-term movement.

### 📊 Exploratory Feature Analysis on AAPL Stock
This section provides a deeper look into Apple (AAPL) stock behavior by engineering additional features and visualizing key relationships. The goal is to uncover patterns in returns, trading volume, and trend movements that may help in understanding stock dynamics and investor behavior.
#### 1. Monthly Distribution of Daily Returns
![Image](https://github.com/user-attachments/assets/056a91b2-3806-4104-9dd1-10b8c87612c1)

A boxplot was generated to show the distribution of daily returns for each month across the dataset. Key observations include:
  - November and December tend to show higher positive returns, possibly due to end-of-year market optimism or earnings season.
  - March and September display wider ranges and more outliers, indicating increased volatility.
  - The chart suggests potential seasonality in market returns.
##### 2. Average Daily Volume by Year
![Image](https://github.com/user-attachments/assets/86247035-2a55-4fdc-8254-e1e469da9b83)

A bar chart was created to visualize the average trading volume per year. Notable trends:
  - A sharp increase in average volume occurred in 2020, coinciding with the COVID-19 pandemic and a rise in retail trading.
  - After 2021, volume began to stabilize, though remaining above pre-2020 levels.
  - Volume patterns may reflect changes in market sentiment, economic events, or broader trading behavior shifts.
#### 3. Autocorrelation of Daily Returns
![Image](https://github.com/user-attachments/assets/58b4611d-fc24-46dd-bc44-2cab80e2bb40)

An autocorrelation plot of daily returns was used to examine persistence in returns over time:
  - The chart shows low to no autocorrelation, especially in short lags, supporting the Efficient Market Hypothesis (EMH).
  - This indicates that past returns have little predictive power over future returns in the short term.
#### 4. Volume vs. Absolute Return
![Image](https://github.com/user-attachments/assets/a92a5440-0a13-490b-8fed-1590174b176c)

A scatter plot of trading volume against the absolute value of daily returns (on a log scale) revealed:
  - A positive relationship between large price movements and high trading volume.
  - This supports the idea that volatility attracts trading activity, potentially due to news events, earnings reports, or speculation.
  - Log scaling helps highlight subtle differences at lower volume levels.
#### 5. Moving Averages Crossover (MA50 vs MA200)
![Image](https://github.com/user-attachments/assets/c60f679e-6fe5-4e0f-9f62-122b51830266)

A crossover chart was plotted using:
  - 50-day Moving Average (MA50): Captures short-to-medium term trends.
  - 200-day Moving Average (MA200): Captures long-term trends.
Insights from the chart:
  -Golden Cross (MA50 crossing above MA200) may indicate bullish momentum.
  - Death Cross (MA50 falling below MA200) may signal bearish momentum.
  - These crossover patterns are widely used in technical analysis to identify trend shifts.

#### 6. Price vs. MA30 Chart
📈 30-Day Simple Moving Average (MA30)
The 30-day Simple Moving Average (SMA) is a fundamental trend-following indicator in technical analysis. It reflects the average closing price of the stock over the past 30 trading days. This smoothing technique helps eliminate short-term volatility and highlights the underlying trend.
In this analysis, MA30 is calculated using a rolling window method. The indicator is then visualized alongside the actual closing prices to show how the market behaves in relation to its recent average.
Insight:
When the closing price crosses above the MA30 line, it may suggest an upward momentum. Conversely, when it drops below, it could indicate a bearish trend.

![Image](https://github.com/user-attachments/assets/e4085327-e9ea-477e-9b46-4e0ffd928f9c)
Displays the closing price of AAPL stock over time with the 30-day moving average (MA30).
  - MA30 smooths short-term fluctuations and highlights trends.
  - Price above MA30 → uptrend; price below MA30 → possible downtrend.
  - Useful for identifying trend direction and potential entry/exit points.

#### 7. RSI14 Chart
📊 14-Day Relative Strength Index (RSI14)
The Relative Strength Index (RSI) is a momentum oscillator used to evaluate whether a stock is overbought or oversold based on recent price changes. In this project, a 14-day RSI is calculated.
  - RSI > 70 typically signals that the stock may be overbought.
  - RSI < 30 indicates a potential oversold condition.
The RSI is derived by averaging recent gains and losses over a 14-day window and applying the RSI formula. It provides a normalized value between 0 and 100 that helps investors assess market sentiment and potential reversal points.

![Image](https://github.com/user-attachments/assets/711dead9-dcac-42a2-bfe0-01e9da7d9936)
Shows the 14-day Relative Strength Index (RSI), a momentum indicator.
  - RSI > 70 → overbought zone, potential price correction.
  - RSI < 30 → oversold zone, potential rebound.
  - Helps assess market momentum and detect possible reversals.

---

## VI. Model Training Process: LSTM for Stock Price Prediction

### Data Preparation

- Extracted the **closing price** data from the preprocessed dataset for use as the target variable in prediction.
- Scaled the closing prices to the range **[0, 1]** using Min-Max scaling to normalize the input, which helps improve neural network training.
- Created input sequences of length **60 days** (window size), where each sample consists of the closing prices of the previous 60 days, and the corresponding target is the closing price on the next day.
- Reshaped the input data to match the expected LSTM input shape: **(samples, time steps, features)**.
- Split the dataset into **training (80%)** and **testing (20%)** sets for model validation.

### Model Architecture

- Constructed an LSTM neural network with:
  - Two LSTM layers (each with 50 units), the first returning sequences for stacking.
  - Dropout layers (20%) to reduce overfitting.
  - Dense layers for output prediction.
- The model is designed to learn temporal dependencies in stock price movements for better prediction accuracy.

### Model Training

- Compiled the model with **Adam optimizer** and **mean squared error (MSE)** loss function, appropriate for regression tasks.
- Trained the model over **50 epochs** with validation on the test set to monitor performance.

### Additional Notes

- The closing prices were inverse-transformed after training to recover predictions in original scale, allowing for meaningful comparison with actual prices.
- This approach leverages historical price patterns to forecast future closing prices, aiding investment decision-making.

--- 

## VII. Test Data Preparation for LSTM Model
To evaluate the performance of the trained model, the testing dataset is prepared by creating sequences of historical data using a sliding window approach with a window size of 100 days. Specifically:
  - Input sequences (x_test) are constructed by taking consecutive 100-day segments from the scaled test data.
  - Corresponding target values (y_test) represent the actual closing price on the day immediately following each 100-day input sequence.
  - Both x_test and y_test are then converted to NumPy arrays to be compatible with the model's expected input format.
This setup allows the model to predict future stock prices based on the previous 100 days of data and facilitates the evaluation of model accuracy on unseen data.

---

## VIII. Making prediction and plotting the graph of predicted vs actual values: LSTM Model
- In this step, the trained model is used to predict the closing stock prices based on the prepared test dataset. The predicted values are stored in the variable y_pred, which contains the forecasted prices corresponding to each day in the test set.
- Both the predicted values (y_pred) and the actual values (y_test) are then inverse transformed by multiplying with a scaling factor. This step restores the data back to the original price scale (in USD), allowing direct comparison between predicted and true prices.

![Image](https://github.com/user-attachments/assets/26b33557-f90d-4074-abba-f8d2a22199e2)

- Finally, the actual and predicted prices are plotted over time. On the plot:
  - The blue line represents the actual closing prices.
  - The red line shows the predicted prices generated by the model.
This visualization helps to intuitively assess the model’s performance in capturing the trend and fluctuations of the stock price over time. Close alignment between the two lines indicates good predictive accuracy, while significant deviations may point to prediction errors or unusual market volatility.

---
## IX. Defining the final dataset for testing by including last 100 coloums of the Model Training with GBTRegressor 
We keep just the six core columns. ("Date","Open","High","Low","Close","Volume")
yfinance pulls historical OHLC+Volume for AAPL from Jan 1, 2020 to Jan 1, 2025.

then we Reads the CSV into a Spark DataFrame.

We create three derived features for each day 
𝑡
:

  PrevClose: yesterday’s Close (lag 1).

  MA5: 5-day moving average of Close.

  MA10: 10-day moving average of Close.

Dropping nulls removes the first 10 rows where these can’t be computed.

VectorAssembler packs all seven numeric inputs into a single features vector column.

GBTRegressor builds 100 decision-tree boosters (maxIter=100), each up to depth 5.

We chain them in a Pipeline for convenience (assembler → gbt) and train on the full engineered DataFrame.

Why GBTRegressor?
Gradient Boosting builds an ensemble of weak learners (small trees) sequentially, each one correcting its predecessor’s errors.

It often outperforms single-tree models and can naturally handle nonlinear interactions between features (e.g. how Volume and MA5 jointly influence tomorrow’s price).

The hyperparameters (maxIter, maxDepth) control complexity vs. overfitting.

and finally Rolls forward to predict the next 100 days using actual daily inputs,

Evaluates and visualizes performance vs. real market data.

---

## X. Model evaluation: Gradient-Boost Tree Regressor

![image](https://github.com/user-attachments/assets/2bd6b725-e7de-4698-b03d-26dd3dcd3809)

✅ General Observations:
- The predicted line (orange, dashed) closely follows the actual line (blue) for most of the 100-day window.
- The model seems to capture trend directionality well, especially during moderate price movements.

⚠️ Notable Divergences:
- Late March to mid-April 2025 shows a sharp drop in actual prices (from ~$220 to ~$170) that the model partially misses — the predicted prices are less volatile, underestimating both the depth of the dip and the magnitude of the rebound.
- This suggests the model might be:
  - Over-smoothed or conservative in high-volatility regimes.
  - Not responsive enough to sudden shocks (e.g., earnings, news).

📊 Statistical Metrics:
- RMSE (Root Mean Squared Error) = 7.3024
  - Penalizes larger errors more than MAE.
  - A value of 7.3 USD indicates on average, predictions deviate by about 3–4% in a price range of $170–$250.
  - This is quite reasonable for stock prediction, where high volatility is the norm.

- MAE (Mean Absolute Error) = 5.4684
  - Less sensitive to outliers than RMSE.
  - Indicates a consistent deviation of ~$5.47, suggesting solid average-case accuracy.
  - Acceptable in financial time series with sharp fluctuations.
 
- R² (R-squared) = 0.8219
  - Explains 82.2% of the variance in the actual closing price.
  - This is a strong result for stock prediction — even models with R² in the 0.7–0.8 range are often considered robust for financial applications.

=> Brief Summary:
- ✔️ The model provides a solid foundation, performs well by industry standards and can capture trend directionality pretty well.
- ❗ However, it may benefit from further tuning or applying more effective method/model , especially for volatile conditions and event-driven dips or rallies.
