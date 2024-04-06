import plotly.graph_objects as go
from scipy.stats import norm, shapiro
import numpy as np
import streamlit as st
import scipy.stats as stats
from scipy.stats import boxcox, yeojohnson
from scipy.stats import shapiro,kstest
import pandas as pd


class Visualization:
    def __init__(self, dataframes):
        self.dataframes = dataframes

    def candlestick(self, df):
        
        fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                                              open=df['open'],
                                              high=df['high'],
                                              low=df['low'],
                                              close=df['close'])])
        fig.update_layout(xaxis_title='Date',
                          yaxis_title='Price',
                          template='plotly_dark',
                          autosize=True,  # Set autosize to True for responsive behavior
                          margin=dict(l=0, r=10, t=40, b=40))  # Optional: Adjust margins
        st.plotly_chart(fig, use_container_width=True)

    def box_plot(self, df):
        

        # Group data by six-month intervals
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['six_month_group'] = pd.PeriodIndex(df['timestamp'], freq='6M')
        grouped_df = df.groupby('six_month_group')

        fig = go.Figure()

        # Plot horizontal box plots for each six-month interval
        for group_name, group_data in grouped_df:
            fig.add_trace(go.Box(y=group_data['return'], name=str(group_name)))

        fig.update_layout(yaxis_title='Return',
                          xaxis_title='Six-Month Interval',
                          template='plotly_dark')
        
        st.plotly_chart(fig, use_container_width=True, width=800, height=600)

    def density_plot(self, df):
        
        daily_return = df['return']
        mean, std_dev = norm.fit(daily_return)
        x = np.linspace(min(daily_return), max(daily_return), 100)
        y = norm.pdf(x, mean, std_dev)
        data_hist = go.Histogram(x=daily_return, histnorm='probability density', name='Data Histogram')
        data_curve = go.Scatter(x=x, y=y, mode='lines', name='Gaussian Fit')
        fig = go.Figure([data_hist, data_curve])
        fig.update_layout(template='plotly_dark',
                          autosize=True,  # Set autosize to True for responsive behavior
                          margin=dict(l=40, r=40, t=40, b=40))  # Optional: Adjust margins
        st.plotly_chart(fig, use_container_width=True)

    def qq_plot(self, df):
        
        returns = df['return']
        
        # Generate Q-Q plot data
        quantiles = np.percentile(returns, np.linspace(0, 100, len(returns)))
        quantiles_theoretical = norm.ppf(np.linspace(0, 1, len(returns)))  # Directly using norm from scipy.stats
        
        # Create Q-Q plot trace
        trace = go.Scatter(x=quantiles_theoretical,
                           y=np.sort(quantiles),
                           mode='markers',
                           marker=dict(color='blue'),
                           name='Q-Q Plot')
        
        layout = go.Layout(xaxis=dict(title='Theoretical Quantiles'),
                           yaxis=dict(title='Sample Quantiles'),
                           showlegend=True,
                           template='plotly_dark',
                           autosize=True,  # Set autosize to True for responsive behavior
                           margin=dict(l=40, r=40, t=40, b=40))  # Optional: Adjust margins
        
        fig = go.Figure(data=[trace], layout=layout)
        st.plotly_chart(fig, use_container_width=True, width=400, height=300)

    def normality(self, df):
        st.subheader('Normality Test')
        returns = df['return']
        # Calculate kurtosis and skewness
        kurtosis = np.mean((returns - np.mean(returns)) ** 4) / np.mean((returns - np.mean(returns)) ** 2) ** 2
        skewness = np.mean((returns - np.mean(returns)) ** 3) / np.mean((returns - np.mean(returns)) ** 2) ** (3/2)
        def shapiro_wilk(self, data):
            return stats.shapiro(data)

        def kolmogorov_smirnov(self, data):
            return stats.kstest(data, 'norm')

        # Shapiro-Wilk test
        shapiro_stat, shapiro_p_value = shapiro(returns)  # Using shapiro directly from scipy.stats

        # Display results
        st.write(f"Kurtosis: {kurtosis:.4f}")
        st.write(f"Skewness: {skewness:.4f}")
        st.write(f"Shapiro-Wilk Test Statistic: {shapiro_stat:.4f}")
        st.write(f"Shapiro-Wilk p-value: {shapiro_p_value:.4f}")
    def shapiro_wilk(self, data):
                return stats.shapiro(data)

    def kolmogorov_smirnov(self, data):
                return stats.kstest(data, 'norm')

    def gauge_chart(self, df):
        output = st.empty()
        with output:
            returns = df['return']
            
            def shapiro_wilk(data):
                return stats.shapiro(data)

            # Calculate statistics
            kurtosis = np.mean((returns - np.mean(returns)) ** 4) / np.mean((returns - np.mean(returns)) ** 2) ** 2
            skewness = np.mean((returns - np.mean(returns)) ** 3) / np.mean((returns - np.mean(returns)) ** 2) ** (3/2)
            shapiro_stat, shapiro_p_value = shapiro_wilk(returns)

            fig = go.Figure()

            fig.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value=kurtosis,
                domain={'x': [0, 1], 'y': [0.6, 1]},
                title = {'text': "kurtosis", 'font': {'size': 24}},
                delta = {'reference': 3, 'increasing': {'color': "RebeccaPurple"}},
                gauge={'axis': {'range': [3, 15],'tickcolor': "darkblue"},'bgcolor': 'royalblue','bar': {'color': "darkblue"},}  # Range for Kurtosis
            ))

            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=skewness,
                domain={'x': [0, 1], 'y': [0, 0.4]},
                title = {'text': "skewness/kurtosis", 'font': {'size': 24}},
                delta = {'reference': 0, 'increasing': {'color': "RebeccaPurple"}},
                gauge={'axis': {'range': [0, 1],'tickcolor': "darkblue"},'bgcolor': 'royalblue','bar': {'color': "darkblue"},}  # Range for Skewness
            ))
            
            
            # Update layout
            fig.update_layout(template='plotly_dark',width=300, height=300, margin=dict(l=0, r=0, t=0, b=0))
            
            st.plotly_chart(fig)
    def gauge_chart_2(self, df):
        output = st.empty()
        with output:
            returns = df['return']
            
            def shapiro_wilk(data):
                return stats.shapiro(data)

            # Calculate statistics
            kurtosis = np.mean((returns - np.mean(returns)) ** 4) / np.mean((returns - np.mean(returns)) ** 2) ** 2
            skewness = np.mean((returns - np.mean(returns)) ** 3) / np.mean((returns - np.mean(returns)) ** 2) ** (3/2)
            shapiro_stat, shapiro_p_value = shapiro_wilk(returns)

            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=shapiro_stat,
                domain={'x': [0, 1], 'y': [0.6, 1]},
                title = {'text': "shapiro", 'font': {'size': 24}},
                
                gauge={'axis': {'range': [0, 1],'tickcolor': "darkblue"},'bgcolor': 'royalblue','bar': {'color': "darkblue"},}  # Range for Skewness
            ))
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=shapiro_p_value,
                domain={'x': [0, 1], 'y': [0, 0.4]},
                title = {'text': "p_value/shapiro", 'font': {'size': 24}},
                
                gauge={'axis': {'range': [0, 1],'tickcolor': "darkblue"},'bgcolor': 'royalblue','bar': {'color': "darkblue"},}  # Range for Skewness
            ))

            # Update layout
            fig.update_layout(template='plotly_dark',width=300, height=300, margin=dict(l=0, r=0, t=0, b=0))
            
            st.plotly_chart(fig)
