# 📈 Advanced Quant & AI Sentiment Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)
![yfinance](https://img.shields.io/badge/yfinance-Data-00aa00.svg)
![Gemini AI](https://img.shields.io/badge/Google_Gemini-AI-orange.svg)

An interactive, full-stack quantitative financial dashboard built for hedge-fund level equity research. It combines real-time technical price action, fundamental valuation metrics, and cutting-edge **Generative AI & NLP** to analyze market sentiment dynamically.

[👉 **View the Live Dashboard Here**](https://quant-dashboard.streamlit.app/)

## 🚀 Key Features

*   **Algorithmic Sentiment Analysis (NLP):** Scrapes real-time financial news headlines and uses `TextBlob` NLP to score market sentiment (Bullish/Bearish) between -1.0 and 1.0.
*   **Generative AI Macro Analyst:** Integrates the `Google Gemini 2.5 Flash` API. The dashboard feeds live stock prices, P/E ratios, and recent news into the LLM to dynamically generate a 2-paragraph "Hedge Fund Analyst Brief" summarizing market conditions.
*   **Pearson Correlation Engine:** Uses `scipy.stats` to calculate the mathematical correlation between daily NLP sentiment scores and actual stock price returns, proving whether the news is a lagging or leading indicator for specific tickers.
*   **Advanced Charting:** Utilizes `Plotly Graph Objects` to render interactive, dark-mode technical charts featuring Price Action, 20-Day Simple Moving Averages (SMA), Volume bars, and overlaid sentiment markers that reveal exact news headlines on hover.
*   **Fundamental Valuation Dashboard:** Tracks comprehensive balance sheet metrics including Market Cap, PEG Ratio, Price-to-Book, Profit Margins, Return on Equity (ROE), and Free Cash Flow.

## 🧠 Architecture & Stack

*   **Frontend / UI:** `Streamlit` (Interactive Web App)
*   **Data Pipeline:** `yfinance` (Real-time Market Data & News)
*   **Data Science / Math:** `Pandas`, `NumPy`, `SciPy`
*   **Visualization:** `Plotly`
*   **Artificial Intelligence:** `google-generativeai` (Gemini API), `TextBlob` (NLP)

## 🛠️ Local Installation

If you want to run the dashboard locally to bypass aggressive cloud API rate limits:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/quant-dashboard.git
    cd quant-dashboard
    ```

2.  **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**
    Create a `.env` file in the root directory and add your Google Gemini API key:
    ```env
    GEMINI_API_KEY="your_api_key_here"
    ```

5.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## ⚠️ Notes on Cloud Deployment (yfinance)
When running on cloud providers like Streamlit Community Cloud or Vercel, the `yfinance` library is occasionally subjected to aggressive rate-limiting by Yahoo's servers, which can temporarily result in `N/A` values on the Fundamental Analytics tab. The app utilizes a `get_fast_info()` fallback mechanism to mitigate this, but local execution remains the most stable environment for deep fundamental scraping.
