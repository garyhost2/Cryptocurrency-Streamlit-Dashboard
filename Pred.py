import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
class LinearForecast:
    def __init__(self, df):
        self.df = df

    def forecast_return(self):
        X = self.df[['score']]
        y = self.df['return']
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Predict return using the score
        self.df['predicted_return'] = model.predict(X)

    def line_forecasted(self, df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['return'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['predicted_return'], mode='lines', name='Forecast'))
        fig.update_layout(title='Cryptocurrency Return Forecast',
                          xaxis_title='Timestamp',
                          yaxis_title='Return',
                          template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    def scatter_plot(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df['score'], y=self.df['return'], mode='markers', name='Score vs Return'))
        fig.update_layout(title='Scatter Plot of Score vs Return',
                          xaxis_title='Score',
                          yaxis_title='Return',
                          template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    def heatmap(self):
        corr = self.df[['score', 'return']].corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values,
                                         x=corr.index.values,
                                         y=corr.columns.values,
                                         colorscale='Viridis'))
        fig.update_layout(title='Correlation Heatmap of Score and Return',
                          template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    def calculate_metrics(self):
        # Calculate metrics for testing and prediction
        y_true = self.df['return']
        y_pred = self.df['predicted_return']

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return mae, mse, r2

    def plot_metrics_gauge(self):
        # Calculate metrics
        mae, mse, r2 = self.calculate_metrics()

        # Create gauge plots for each metric
        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=mae,
            domain={'x': [0, 1], 'y': [0.6, 1]},
            title={'text': "MAE"},
            gauge={'axis': {'range': [0, 1],'tickcolor': "red"},'bgcolor': 'royalblue','bar': {'color': "red"},}  
        ))

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=mse,
            domain={'x': [0, 1], 'y': [0, 0.4]},
            title={'text': "MSE"},
            gauge={'axis': {'range': [0, 1],'tickcolor': "darkblue"},'bgcolor': 'royalblue','bar': {'color': "darkblue"},}
        ))

        

        # Update layout for dark theme
        fig.update_layout(
            template='plotly_dark',
            title='Metrics Gauge Plot',
        )

        # Show plot
        fig.update_layout(template='plotly_dark', width=300, height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig)
class ExponentialSmoothingForecast:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def fit(self):
        model = ExponentialSmoothing(self.train_data['return'], trend='add', seasonal='add', seasonal_periods=24).fit()
        return model

    def forecast(self, model):
        forecast = model.forecast(steps=len(self.test_data))
        return forecast

    def evaluate(self, forecast):
        y_test = self.test_data['return']
        y_pred = forecast

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return mse, mae, r2

    def plot_forecast(self, model, forecast):
        fig = go.Figure()

        # Add actual data
        fig.add_trace(go.Scatter(x=self.train_data['timestamp'], y=self.train_data['return'], mode='lines', name='Actual'))

        # Add forecasted values
        fig.add_trace(go.Scatter(x=self.test_data['timestamp'], y=forecast, mode='lines', name='Forecast'))

        # Update layout with dark theme
        fig.update_layout(title='Cryptocurrency Return Forecast',
                          xaxis_title='Timestamp',
                          yaxis_title='Return',
                          template='plotly_dark')  # Set dark theme

        # Show plot
        st.plotly_chart(fig, use_container_width=True)
    def plot_metrics_gauge(self,forecast):
        # Calculate metrics
        mse, mae, r2 = self.evaluate(forecast)

        # Create gauge plots for each metric
        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=mse,
            domain={'x': [0.6, 1], 'y': [0, 1]},
            title={'text': "MSE"},
            gauge={'axis': {'range': [-1, 1],'tickcolor': "darkblue"},'bgcolor': 'royalblue','bar': {'color': "red"},}  
        ))

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=r2,
            domain={'x': [0, 0.4], 'y': [0, 1]},
            title={'text': "r²"},
            gauge={'axis': {'range': [-1, 1],'tickcolor': "darkblue"},'bgcolor': 'royalblue','bar': {'color': "red"},}
        ))

        

        # Update layout for dark theme
        fig.update_layout(
            template='plotly_dark',
            title='Metrics Gauge Plot',
        )

        # Show plot
        fig.update_layout(template='plotly_dark', width=600, height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig,use_container_width=True)
    