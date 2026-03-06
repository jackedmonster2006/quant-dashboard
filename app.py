import streamlit as st
import yfinance as yf
import pandas as pd
from textblob import TextBlob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from scipy.stats import pearsonr
import os
from dotenv import load_dotenv

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# We will try to use the Google Gemini SDK for the macro AI summary
try:
    import google.generativeai as genai
    
    # Streamlit Cloud puts secrets in st.secrets, but local uses os.getenv
    try:
        gemini_key = st.secrets["GEMINI_API_KEY"]
    except:
        gemini_key = os.getenv("GEMINI_API_KEY")
        
    if gemini_key:
        genai.configure(api_key=gemini_key)
        # Verify it works by checking the model
        model = genai.GenerativeModel('gemini-2.5-flash')
        ai_client = True
    else:
        ai_client = False
except Exception as e:
    print("AI Client Error:", e)
    ai_client = False

# --- PAGE SETUP ---
st.set_page_config(page_title="Quant Dashboard | FinBERT & Fundamentals", layout="wide", page_icon="📈")
st.title("📈 Advanced Quant & Sentiment Dashboard")
st.markdown("Combines Technicals, Fundamental Valuation, and AI-driven Alternative Data.")

# --- SIDEBAR & NAVIGATION ---
st.sidebar.header("⚙️ Configuration")

SECTORS = {
    "Technology 💻": {
        "AI & Semiconductors": {"NVIDIA": "NVDA", "AMD": "AMD", "Broadcom": "AVGO", "Taiwan Semi": "TSM"},
        "Software & Cloud": {"Microsoft": "MSFT", "Palantir": "PLTR", "Salesforce": "CRM", "Snowflake": "SNOW"},
        "Consumer Tech": {"Apple": "AAPL", "Alphabet (Google)": "GOOGL", "Meta": "META"}
    },
    "E-Commerce & Retail 🛒": {
        "Global Marketplaces": {"Amazon": "AMZN", "Shopify": "SHOP", "Alibaba": "BABA"},
        "Retail Chains": {"Walmart": "WMT", "Costco": "COST", "Target": "TGT"}
    },
    "Finance & Crypto 🏦": {
        "Traditional Banks": {"JPMorgan": "JPM", "Bank of America": "BAC", "Goldman Sachs": "GS"},
        "Fintech & Crypto": {"Coinbase": "COIN", "Block": "SQ", "PayPal": "PYPL", "Robinhood": "HOOD"}
    },
    "Automotive & EV 🚗": {
        "Electric Vehicles": {"Tesla": "TSLA", "Rivian": "RIVN", "Lucid": "LCID"},
        "Legacy Auto": {"Ford": "F", "General Motors": "GM", "Toyota": "TM"}
    },
    "Custom Ticker 🔍": {}
}

selected_sector = st.sidebar.selectbox("Select Sector", list(SECTORS.keys()))

if selected_sector == "Custom Ticker 🔍":
    ticker_input = st.sidebar.text_input("Enter Any Ticker Symbol", "NVDA").upper()
    company_name = ticker_input
else:
    sub_sectors = SECTORS[selected_sector]
    selected_sub = st.sidebar.selectbox("Select Industry", list(sub_sectors.keys()))
    companies = sub_sectors[selected_sub]
    selected_company = st.sidebar.selectbox("Select Company", list(companies.keys()))
    ticker_input = companies[selected_company]
    company_name = selected_company

period_input = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y"])

# --- CACHED DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_data(ticker, period):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    # Try to grab fundamentals
    try:
        info = stock.info
    except:
        info = {}
        
    if hist.empty:
        return None, None, info
        
    hist.index = hist.index.tz_localize(None)
    
    # Fetch news
    try:
        news = stock.news
    except:
        news = []
        
    news_data = []
    for article in news:
        # Check if article is valid
        if not article: continue
        
        # yfinance changed their news API response structure recently
        # It's now nested under a 'content' key
        content = article.get('content', article) # Fallback for old structure
        
        # If content is None for some reason, fallback to empty dict
        if content is None: content = {}
        
        title = content.get('title', '')
        
        # Get publisher safely
        provider = content.get('provider') or {}
        publisher = provider.get('displayName', 'Unknown')
        
        # Get link safely
        click_url = content.get('clickThroughUrl') or {}
        link = click_url.get('url', '')
        
        # Timestamps are now ISO 8601 strings, not unix epoch
        pub_date_str = content.get('pubDate')
        
        if not pub_date_str or not title: 
            continue
            
        try:
            # Parse ISO string "2026-03-01T22:49:00Z"
            date = pd.to_datetime(pub_date_str).tz_localize(None)
        except:
            continue
        
        # NLP Sentiment Analysis
        analysis = TextBlob(title)
        score = analysis.sentiment.polarity
        
        if score > 0.05:
            label = 'Bullish 🟢'
            color = 'green'
        elif score < -0.05:
            label = 'Bearish 🔴'
            color = 'red'
        else:
            label = 'Neutral ⚪'
            color = 'gray'
            
        news_data.append({
            'Date': pd.to_datetime(date.date()),
            'Published': date.strftime("%Y-%m-%d %H:%M"),
            'Title': title,
            'Publisher': publisher,
            'Score': score,
            'Sentiment': label,
            'Color': color,
            'Link': link
        })
        
    return hist, pd.DataFrame(news_data), info

# --- MAIN APP LOGIC ---
if ticker_input:
    with st.spinner(f"Fetching market data and calculating Quant metrics for {ticker_input}..."):
        stock_df, news_df, stock_info = fetch_data(ticker_input, period_input)
        
    if stock_df is None or stock_df.empty:
        st.error(f"❌ Could not find market data for ticker: {ticker_input}.")
    else:
        # Create Tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Technicals & Sentiment", "💼 Fundamental Analytics", "🤖 AI Macro Summary", "📰 Live News Feed"])
        
        with tab1:
            # --- METRICS ROW ---
            current_price = stock_df['Close'].iloc[-1]
            prev_price = stock_df['Close'].iloc[-2] if len(stock_df) > 1 else current_price
            pct_change = ((current_price - prev_price) / prev_price) * 100
            
            volume_today = stock_df['Volume'].iloc[-1]
            avg_volume_10d = stock_df['Volume'].tail(10).mean() if len(stock_df) >= 10 else volume_today
            vol_ratio = (volume_today / avg_volume_10d) * 100 if avg_volume_10d > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price (Close)", f"${current_price:.2f}", f"{pct_change:.2f}%")
            
            vol_delta = f"{vol_ratio - 100:.1f}% vs 10D Avg"
            col2.metric("Trading Volume", f"{volume_today:,.0f}", vol_delta, delta_color="normal" if vol_ratio > 100 else "inverse")
            
            # --- CORRELATION CALCULATION ---
            # Do sentiment scores actually move the price?
            corr_text = "Need more news data"
            if not news_df.empty:
                # Calculate daily returns
                stock_df['Daily_Return'] = stock_df['Close'].pct_change()
                
                # We need to map the news date to the closest valid trading day
                # Since news can break on weekends, we merge using merge_asof
                
                # Sort both dataframes by date first (required for merge_asof)
                news_df_sorted = news_df.sort_values('Date')
                stock_df_sorted = stock_df.reset_index().sort_values('Date')
                
                # Group sentiment by date first so we don't have duplicate news days
                daily_sentiment = news_df_sorted.groupby('Date')['Score'].mean().reset_index()
                
                # Merge: find the closest future trading day for each piece of news
                # (e.g. Saturday news affects Monday's return)
                merged_corr = pd.merge_asof(
                    daily_sentiment, 
                    stock_df_sorted[['Date', 'Daily_Return']].dropna(), 
                    on='Date', 
                    direction='forward',
                    tolerance=pd.Timedelta(days=3)
                ).dropna()
                
                # We need at least 3 distinct data points to run a valid Pearson Correlation
                if len(merged_corr) >= 3:
                    try:
                        corr, p_value = pearsonr(merged_corr['Score'], merged_corr['Daily_Return'])
                        if corr > 0.3:
                            corr_text = f"Strong Positive ({corr:.2f})"
                        elif corr < -0.3:
                            corr_text = f"Strong Negative ({corr:.2f})"
                        else:
                            corr_text = f"Weak/None ({corr:.2f})"
                    except Exception:
                        corr_text = "Stat Error"
            
            col3.metric("Sentiment-Price Correlation", corr_text)
            
            avg_sentiment = news_df['Score'].mean() if not news_df.empty else 0
            if avg_sentiment > 0.05:
                sentiment_text = "Overall Bullish 🟢"
            elif avg_sentiment < -0.05:
                sentiment_text = "Overall Bearish 🔴"
            else:
                sentiment_text = "Overall Neutral ⚪"
            col4.metric("AI Sentiment Score", f"{avg_sentiment:.2f}", sentiment_text)

            st.divider()

            # --- INTERACTIVE CHART ---
            st.subheader(f"Price Action vs. NLP News Sentiment")
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Stock Line
            fig.add_trace(go.Scatter(
                x=stock_df.index, y=stock_df['Close'], 
                name="Closing Price", line=dict(color='#1f77b4', width=2)
            ), secondary_y=False)
            
            # 20-Day SMA Line
            if len(stock_df) >= 20:
                stock_df['SMA_20'] = stock_df['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=stock_df.index, y=stock_df['SMA_20'], 
                    name="20-Day SMA", line=dict(color='#ff7f0e', width=1, dash='dash')
                ), secondary_y=False)
                
            # Volume
            fig.add_trace(go.Bar(
                x=stock_df.index, y=stock_df['Volume'],
                name="Volume", marker_color='rgba(158,202,225,0.3)'
            ), secondary_y=True)
            
            # News Overlay
            if not news_df.empty:
                merged = pd.merge(news_df, stock_df['Close'].reset_index(), left_on='Date', right_on='Date', how='inner')
                if not merged.empty:
                    hover_text = merged['Publisher'] + "<br><b>" + merged['Title'] + "</b><br>Sentiment: " + merged['Sentiment']
                    fig.add_trace(go.Scatter(
                        x=merged['Date'], y=merged['Close'], 
                        mode='markers', name='News Events',
                        marker=dict(color=merged['Color'], size=14, line=dict(width=2, color='white'), symbol='circle'),
                        text=hover_text, hoverinfo='text'
                    ), secondary_y=False)
            
            fig.update_layout(
                template="plotly_dark", 
                hovermode="x unified", 
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                barmode='overlay'
            )
            
            fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
            fig.update_yaxes(showgrid=False, showticklabels=False, secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown(f"### 💼 Fundamental Valuation: {company_name} ({ticker_input})")
            st.markdown("Key financial metrics driving institutional sentiment.")
            st.markdown("---")
            
            # Helper formatting
            def format_large_number(num):
                if not num: return "N/A"
                if num >= 1_000_000_000_000:
                    return f"${num/1_000_000_000_000:.2f} Trillion"
                elif num >= 1_000_000_000:
                    return f"${num/1_000_000_000:.2f} Billion"
                elif num >= 1_000_000:
                    return f"${num/1_000_000:.2f} Million"
                return f"${num}"

            # Safely grab metrics
            market_cap = stock_info.get('marketCap', None)
            pe_ratio = stock_info.get('trailingPE', "N/A")
            fwd_pe = stock_info.get('forwardPE', "N/A")
            peg_ratio = stock_info.get('pegRatio', "N/A")
            price_to_book = stock_info.get('priceToBook', "N/A")
            
            profit_margin = stock_info.get('profitMargins', 0) * 100 if stock_info.get('profitMargins') else "N/A"
            operating_margin = stock_info.get('operatingMargins', 0) * 100 if stock_info.get('operatingMargins') else "N/A"
            revenue_growth = stock_info.get('revenueGrowth', 0) * 100 if stock_info.get('revenueGrowth') else "N/A"
            earnings_growth = stock_info.get('earningsGrowth', 0) * 100 if stock_info.get('earningsGrowth') else "N/A"
            
            roe = stock_info.get('returnOnEquity', 0) * 100 if stock_info.get('returnOnEquity') else "N/A"
            roa = stock_info.get('returnOnAssets', 0) * 100 if stock_info.get('returnOnAssets') else "N/A"
            debt_eq = stock_info.get('debtToEquity', "N/A")
            current_ratio = stock_info.get('currentRatio', "N/A")
            
            total_cash = stock_info.get('totalCash', None)
            total_debt = stock_info.get('totalDebt', None)
            free_cash_flow = stock_info.get('freeCashflow', None)
            
            # --- ROW 1: CORE VALUATION ---
            st.markdown("#### 💎 Valuation Metrics")
            val_col1, val_col2, val_col3, val_col4 = st.columns(4)
            with val_col1:
                st.info(f"**Market Cap**\n\n### {format_large_number(market_cap)}")
            with val_col2:
                pe_display = f"{pe_ratio:.2f}" if type(pe_ratio) is float else "N/A"
                st.info(f"**P/E Ratio (TTM)**\n\n### {pe_display}")
            with val_col3:
                peg_display = f"{peg_ratio:.2f}" if type(peg_ratio) is float else "N/A"
                st.info(f"**PEG Ratio**\n\n### {peg_display}")
            with val_col4:
                ptb_display = f"{price_to_book:.2f}" if type(price_to_book) is float else "N/A"
                st.info(f"**Price to Book**\n\n### {ptb_display}")
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- ROW 2: PROFITABILITY & GROWTH ---
            st.markdown("#### 🚀 Profitability & Growth")
            prof_col1, prof_col2, prof_col3, prof_col4 = st.columns(4)
            with prof_col1:
                margin_display = f"{profit_margin:.2f}%" if type(profit_margin) is float else "N/A"
                st.success(f"**Net Profit Margin**\n\n### {margin_display}")
            with prof_col2:
                op_margin_display = f"{operating_margin:.2f}%" if type(operating_margin) is float else "N/A"
                st.success(f"**Operating Margin**\n\n### {op_margin_display}")
            with prof_col3:
                rev_display = f"{revenue_growth:.2f}%" if type(revenue_growth) is float else "N/A"
                st.success(f"**Revenue Growth (YoY)**\n\n### {rev_display}")
            with prof_col4:
                earn_display = f"{earnings_growth:.2f}%" if type(earnings_growth) is float else "N/A"
                st.success(f"**Earnings Growth (YoY)**\n\n### {earn_display}")

            st.markdown("<br>", unsafe_allow_html=True)

            # --- ROW 3: BALANCE SHEET & MANAGEMENT ---
            st.markdown("#### 🏦 Balance Sheet & Management Effectiveness")
            bal_col1, bal_col2, bal_col3, bal_col4 = st.columns(4)
            with bal_col1:
                roe_display = f"{roe:.2f}%" if type(roe) is float else "N/A"
                st.warning(f"**Return on Equity (ROE)**\n\n### {roe_display}")
            with bal_col2:
                roa_display = f"{roa:.2f}%" if type(roa) is float else "N/A"
                st.warning(f"**Return on Assets (ROA)**\n\n### {roa_display}")
            with bal_col3:
                debt_display = f"{debt_eq:.2f}" if type(debt_eq) is float else "N/A"
                st.warning(f"**Debt/Equity Ratio**\n\n### {debt_display}")
            with bal_col4:
                cr_display = f"{current_ratio:.2f}" if type(current_ratio) is float else "N/A"
                st.warning(f"**Current Ratio**\n\n### {cr_display}")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- ROW 4: CASH FLOW ---
            st.markdown("#### 💸 Cash Flow Health")
            cf_col1, cf_col2, cf_col3 = st.columns(3)
            with cf_col1:
                st.info(f"**Total Cash**\n\n### {format_large_number(total_cash)}")
            with cf_col2:
                st.error(f"**Total Debt**\n\n### {format_large_number(total_debt)}")
            with cf_col3:
                st.success(f"**Free Cash Flow**\n\n### {format_large_number(free_cash_flow)}")

            st.markdown("<br>", unsafe_allow_html=True)

            # --- ROW 3: COMPANY PROFILE ---
            with st.expander("📚 Read Deep Company Profile", expanded=True):
                sector = stock_info.get('sector', 'Unknown Sector')
                industry = stock_info.get('industry', 'Unknown Industry')
                employees = stock_info.get('fullTimeEmployees', 'Unknown')
                
                st.markdown(f"**Sector:** {sector} | **Industry:** {industry} | **Employees:** {employees:,}")
                st.markdown("---")
                st.write(stock_info.get('longBusinessSummary', 'No description available in the API for this ticker.'))

        with tab3:
            st.subheader("🧠 Generative AI Analyst Brief")
            st.markdown("Sends the recent news headlines and fundamentals to Google Gemini to write a summary.")
            
            # Display debugging info so the user knows what keys Streamlit actually sees
            try:
                debug_secrets = list(st.secrets.keys())
            except:
                debug_secrets = []
                
            if not ai_client:
                st.warning("⚠️ No Gemini API Key found.")
                with st.expander("Deployment Debugger"):
                    st.write("Streamlit Cloud Secrets Found:", debug_secrets)
                    st.write("Did you click Advanced Settings -> Secrets before deploying and paste `GEMINI_API_KEY=...`?")
            else:
                if st.button("Generate Analyst Brief"):
                    with st.spinner("Consulting AI Quant..."):
                        try:
                            # Build the context
                            headlines = "\n".join(news_df['Title'].head(10).tolist()) if not news_df.empty else "No news available."
                            
                            prompt = f"""
                            You are a senior hedge fund analyst. Write a concise, 2-paragraph market brief for {company_name} ({ticker_input}).
                            
                            Current Price: ${current_price:.2f}
                            P/E Ratio: {pe_ratio}
                            Market Cap: {format_large_number(market_cap)}
                            
                            Recent Headlines:
                            {headlines}
                            
                            Instructions:
                            1. Summarize the macro sentiment driving the stock right now based on the headlines.
                            2. Comment briefly on whether its fundamentals justify the current sentiment.
                            3. Be extremely professional and quantitative in your language. Do not give financial advice.
                            """
                            
                            response = model.generate_content(prompt)
                            st.success("Brief Generated Successfully")
                            st.write(response.text)
                        except Exception as e:
                            st.error(f"Error generating brief. Check Streamlit logs.")
                            # Print the actual error to the server logs so we can see what's failing
                            print("GEMINI API ERROR:", str(e))

        with tab4:
            st.subheader(f"📰 Live News Feed: {company_name}")
            st.markdown("Latest headlines scored in real-time by NLP Sentiment Analysis.")
            st.markdown("---")
            
            if not news_df.empty:
                # Sort by newest first
                sorted_news = news_df.sort_values(by='Date', ascending=False)
                
                for _, row in sorted_news.iterrows():
                    # Color code the sentiment badge
                    if "Bullish" in row['Sentiment']:
                        badge_color = "success"
                        icon = "🟢"
                    elif "Bearish" in row['Sentiment']:
                        badge_color = "error"
                        icon = "🔴"
                    else:
                        badge_color = "normal"
                        icon = "⚪"
                        
                    with st.container():
                        colA, colB = st.columns([4, 1])
                        with colA:
                            st.markdown(f"#### [{row['Title']}]({row['Link']})")
                            st.caption(f"**{row['Publisher']}** • Published: {row['Published']}")
                        with colB:
                            st.metric("Sentiment", f"{icon} {row['Score']:.2f}")
                        st.divider()
            else:
                st.info("No recent news found for this ticker.")

