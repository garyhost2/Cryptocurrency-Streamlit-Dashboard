import streamlit as st
import plotly.graph_objects as go
import altair as alt
from visualisation import Visualization
from Pred import LinearForecast
from Pred import ExponentialSmoothingForecast
from scipy.stats import pearsonr, spearmanr
import pandas as pd 



st.set_page_config(
    page_title="Cryptocurrency Reddit Insights.",
    layout="wide",
    page_icon="🤑",
    initial_sidebar_state="expanded")



st.markdown(
    """
    <style>
        footer {display: none}
        [data-testid="stHeader"] {display: none}
    </style>
    """, unsafe_allow_html = True
)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)
    
title_col, emp_col, btc_col, eth_col, sol_col, dodge_col, xrp_col = st.columns([1,0.2,1,1,1,1,1])

with title_col:
    st.markdown('<p class="dashboard_title">Crypto<br>Dashboard</p>', unsafe_allow_html = True)

with btc_col:
    with st.container():
        btc_price = '7.32 T'
        st.markdown(f'<p class="btc_text">BTC / USDT<br></p><p class="price_details">{btc_price}</p>', unsafe_allow_html = True)
        

with eth_col:
    with st.container():
        eth_price = '3.41 T'
        st.markdown(f'<p class="eth_text">ETH / USDT<br></p><p class="price_details">{eth_price}</p>', unsafe_allow_html = True)

with dodge_col:
    with st.container():
        dodge_price = '0.36 T' 
        st.markdown(f'<p class="xmr_text"> Ð / USDT<br></p><p class="price_details">{dodge_price}</p>', unsafe_allow_html = True)

with sol_col:
    with st.container():
        sol_price = '0.63 T'
        st.markdown(f'<p class="sol_text">SOL / USDT<br></p><p class="price_details">{sol_price}</p>', unsafe_allow_html = True)

with xrp_col:
    with st.container():
        xrp_price = '0.38 T' 
        st.markdown(f'<p class="xrp_text">XRP / USDT<br></p><p class="price_details">{xrp_price}</p>', unsafe_allow_html = True)
        





# Read CSV files for each cryptocurrency
bitcoin_df = pd.read_csv('data/Bitcoin.csv')
ethereum_df = pd.read_csv('data/Ethereum.csv')
solana_df = pd.read_csv('data/Solana.csv')
dogecoin_df = pd.read_csv('data/Dodgecoin.csv')

dataframes = {
    'Bitcoin': bitcoin_df,
    'Ethereum': ethereum_df,
    'Solana': solana_df,
    'Dogecoin': dogecoin_df
}

visualizer = Visualization(dataframes)
models = ['Exponential Smoothing', 'Linear Regression', 'Prophet']
merged_df = pd.read_csv(r'data\btc_reddit.csv')
def correlation_gauge(merged_df):
    
        pearson_corr, _ = pearsonr(merged_df['return'], merged_df['score'])

        # Calculate Spearman correlation coefficient and p-value
        spearman_corr, _ = spearmanr(merged_df['return'], merged_df['score'])

        # Create gauge plot
        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=pearson_corr,
            domain={'x': [0, 1], 'y': [0.6, 1]},
            title={'text': "Pearson Correlation"},
            gauge={'axis': {'range': [-1, 1],'tickcolor': "darkblue"},'bgcolor': 'royalblue','bar': {'color': "darkblue"},}
        ))

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=spearman_corr,
            domain={'x': [0, 1], 'y': [0, 0.4]},
            title={'text': "Spearman Correlation"},
            gauge={'axis': {'range': [-1, 1],'tickcolor': "darkblue"},'bgcolor': 'royalblue','bar': {'color': "darkblue"},}
        ))

        # Update layout
        fig.update_layout(template='plotly_dark')

        # Show plot
        st.plotly_chart(fig, use_container_width=True)

params_col, chart_col, data_col = st.columns([0.5,1.2,0.6])

with params_col:
    
    with st.form(key = 'params_form'):
        
        st.markdown(f'<p class="params_text">Dashboard Parameters', unsafe_allow_html = True)
        
        
        
        
        selected_coin = st.selectbox('Coins', dataframes.keys(), key = 'Models_selectbox')
        
        
        
        
        update_chart = st.form_submit_button('Update chart')
        st.markdown('')
        Model = st.selectbox('Models', models, key = 'Coins_selectbox')
        Predict = st.form_submit_button('Forecast')
        st.markdown('')
        
        if update_chart:
            

            with chart_col:

                with st.container():        
                    visualizer.candlestick(dataframes[selected_coin])
                    st.divider()
                    visualizer.box_plot(dataframes[selected_coin])
                    
            with data_col:
                visualizer.gauge_chart(dataframes[selected_coin])
                st.divider()
                visualizer.gauge_chart_2(dataframes[selected_coin])
                st.divider()
                
        if Predict:
            train_size = int(len(merged_df) * 0.8)
            train_data, test_data = merged_df[:train_size], merged_df[train_size:]
            model = ExponentialSmoothingForecast(train_data, test_data)
            exponential_model = model.fit()
            forecast = model.forecast(exponential_model)
            mse, mae, r2 = model.evaluate(forecast)

            with chart_col:

                with st.container():        
                    model.plot_forecast(exponential_model, forecast)
                    model.plot_metrics_gauge(forecast)


            with data_col:
                 correlation_gauge(merged_df)
                
        
    
footer="""<style>
a:link , a:visited{
font-family: 'Space Grotesk';
color: white;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: #030e3b;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: transparent;
color: #f7931a;
text-align: center;
box-shadow: -6px 8px 20px 1px #ccc6c652;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;'href="https:https://www.linkedin.com/in/yassine-ben-zekri-72aa6b199/" target="_blank">Med Yassine Ben Zekri</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)