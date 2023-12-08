# Trading Strategy

MBAI 5300G – Programming and Data Processing

ANALYSIS OF CONVENTIONAL TECHNICAL INDICATORS & MACHINE LEARNING METHODS IN ALGORITHMIC TRADING DEVELOPMENT

Prepared By: Huy Nguyen, Chris Gell

## Abstract

Technical Analysis as a concept has been around almost since the inception of the stock market itself. Through the use of statistical methods, price charting, and data collection, all these variables lead themselves along a simple path to a complicated outcome; utilizing past market movements to predict future pricing trends.

However, is there any merit to applying these techniques when it comes to investing? Can the use of technical indicators potentially outperform top mutual funds in Canada?

Over the course of this paper, we examined the use and implementation of classic technical indicators for developing simple algorithmic trading models, while comparing the outcomes against one of Canada’s top performing mutual funds. Then, we looked into more modern methods of asset selection utilizing machine learning techniques such as K-Means to identify clusters of stocks with optimal variance to return measures. Finally, we examined a logic-based approach by ranking stocks in the NASDAQ-100 based on returns, continuously buying the top winners, then rebalancing the portfolio at the end of each month.

Utilizing the Python programming language, we were able to show that implementing a rules-based approach to investing using technical indicators had the potential to outperform the <a href= "https://marketsandresearch.td.com/tdwca/Public/MutualFundsProfile/Summary/ca/TDB908?addRecent=true"> TDB908 mutual fund</a>.

As a result of our analysis, we believe there is ample opportunity for the inclusion of technical based indicators when it comes to stock selection and trading decisions in an investment portfolio. As such, adhering to a rules-based approach has the potential to increase investment returns beyond those seen in a traditional investment fund.

*Keywords: Machine Learning, Technical Analysis, Algorithmic Trading, K-Means, Stocks, Python*
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Introduction

For our research, we examined popular, classic technical indicators such as the Relative Strength Index (RSI), Pocket Pivot, Ichimoku Cloud, Simple Moving Average (SMA), and Bollinger Bands and the impact their trading signals could have on investment performance (Edwards, Magee, & Bassetti, 2007). Furthermore, stock ranking methods and K-Means machine learning were implemented as additional measures for improving investment performance (Radovanovic, 2022).

We hoped to show that by adhering to a rules-based investment approach for making trading decision, the returns from technical indicators could outperform those of a top-ranking mutual fund in Canada.

However, we did not specifically focus on beating the mutual fund for our analysis. It would no doubt be possible to find some combination of back testing data with the above indicators that would beat the mutual fund consistently on every measure, sometimes called “data dredging” when analyzing data (Awati, 2022). To circumvent this, we decided to use the generally accepted standards for the technical indicators looked at. For example, when we analyzed our trading strategy using Bollinger Bands, the SMA used was 20, the general default for this indicator.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Literature Review

There is an abundance of resources available should one wish to learn about incorporating technical indicators into their trading strategies. From simple webpage blogs on Investopedia (Hayes, Technical Analysis: What It Is and How to Use It in Investing, 2022), to the more robust Technical Analysis of Stock Trends written by Edwards and Magee (Edwards, Magee, & Bassetti, 2007), all of these resources contain valuable insights into trading securities with technical indicators.

In particular, Edwards and Magee outline numerous best practices when it comes to trading traditional technical indicators and chart patterns alike. Additionally, the “Journal of Technical Analysis” published by the Chartered Market Technicians Association (Market Technicians Association, 2002) has also recently started to include articles surrounding machine learning models, such as Ross Gordon (2021), where the author examined the use of an Artificial Neural Network for predicting trends in the EUR/USD currency trading pair (Gordon, 2021). As a result, examining how incorporating machine learning and technical indicators into algorithmic models is a logical “next step” in enhance the investment decision making process.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Data Description

Our analysis focused on the top 10 holdings of the mutual fund and encompassed 10 years of daily historical share price data from Sep 30<sup>th</sup>, 2013 to Sep 30<sup>th</sup>, 2023 for consistency amongst the traditional technical indicators. This gave us access to 2,518 observations for conducting our analysis.

For our machine learning and logic-based approaches, we kept to companies within the NASDAQ-100, but due to the machine learning and ranking natures of the methods, held various companies at different times. Where possible, 10 years’ worth of data were used, except for the momentum-based model which only utilized 5 years’ worth of return data.

Each of the daily historical datasets contained columns recording the: Open, High, Low, Close, Volume, Dividends, and Stock Splits of each company, with the Date value being stored as the row index.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Methodology

Data was collected through queries utilizing the yfinance library in Python, which directly retrieves market data from Yahoo! Finance (yfinance Python library version 0.2.32). Some of the strategies downloaded the data, while others directly loaded the query results into a data frame for analysis. The BeautifulSoup library was also utilized in collecting historical ticker data from the NASDAQ-100 index (Beautiful Soup Documentation). This was done since the companies included in the NASDAQ-100 index change over time due to the rebalancing nature of the index for inclusion (CFI Team, n.d.).

Once all of our data was retrieved and checked for consistency, we began implementing our trading strategies with the built-in technical analysis indicators from the pandas\_ta library, or directly writing the functions ourselves for the calculations. 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Exploratory Data Analysis (EDA)

Little, if any exploratory data analysis was performed during this research due to the structured nature of the data being collected. All of the share price information coming from Yahoo! Finance was known to be structured the same across all of the companies selected for analysis. Visualization and analysis would be critical at later stages of our review, but wasn’t required after initially obtaining the data.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Limitations

While the overall data quality we retrieved was complete with no NaN values, there were some issues in implementing the momentum strategy for buying and selling companies on the NASDAQ-100 index. In particular, we struggled finding a reliable historical list of companies that were included in the index throughout the years. As a result, our best resource was to utilize the Wikipedia page, and crawl the ticker list from the webpages that were working correctly (Nasdaq-100, n.d.). Due to this limitation, the returns generated in our momentum strategy are estimated based on the 5 years’ worth of data we managed to collect, extrapolated to approximate 10 years’ worth of returns.

Additionally, we also encountered issues while trying to implement one of our technical analysis strategies. In particular, generating a properly functioning trade signal for the combination strategy involving: RSI, Pocket Pivots, Keltner channels, and Moving Average Convergence Divergence (MACD) indicators. As a result, we ended up removing the Keltner channels and MACD indicator so the model could function.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Prediction Models

All of our analysis and implementation for technical indicators and algorithmic trading was completed utilizing the Python programming language (Python Software Foundation). The following 5 models were chosen and developed to be tested against the selected top performing mutual fund:

- An RSI and Pocket Pivot combination strategy (Fernando, 2023), (Shane, 2022).
- A momentum-based approach, buying the top performers of the NASDAQ-100 index each month, rebalancing at the end of the month.
- K-Means clustering, an unsupervised machine learning method in combination with Ichimoku Clouds (Trading Strategy Guides, 2023).
- Golden Cross, a popular SMA strategy tracking the crosses between the 50 SMA and 200 SMA trend lines (Hayes, Golden Cross Pattern Explained With Examples and Charts, 2023).
- Bollinger Bands, a ranged-bound approach where we would buy or sell based on crosses over the upper or lower bands (Hayes, Bollinger Bands®: What They Are, and What They Tell Investors, 2023).

Furthermore, the price recorded for all of our trades was based on the closing price of the next trading day. This is because our models would only generate a signal based on the current day’s closing price, meaning we would have to wait until the next day to enter into our position due to market closures. 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## RSI & Pocket Pivot

This technical analysis model utilized both RSI and Pocket Pivot to track the momentum and volume of the traded companies for buy or sell signals. Essentially, we would generate a buy or sell signal separately for each of the indicators. Pocket Pivot would signal a buy when the required volume reached a specified threshold relative to the max down volume, or a sell signal when the same was true for the up volume.

For our RSI indicator, we would simply track when the index entered into oversold or overbought territory based on the industry standard measures of below 30 for oversold, and above 70 for overbought. Additionally, no buys or sells would occur if both indicators gave conflicting signals, such as RSI buy and Pocket Pivot sell, instead, we would simply hold our position.

Once these parameters were decided on, we simply implemented the trading logic into Python to create our first algorithmic trading strategy. We would enter into a buy position if either of the indicators signaled a buy opportunity, and sell when either indicated a selling opportunity, assuming any stock was currently being held to sell.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Momentum

This strategy was more unique and cumbersome in its implementation. After crawling the Wikipedia page for the historical list of companies, we would measure the top 10 performers based on monthly returns, then buy them. Then, at the end of the month, sell them to record the gains or losses and buy the next set of top 10 performers. This process would repeat for the 5 years of data we managed to collect through the BeautifulSoup library.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## K-Means With Ichimoku Cloud

For our machine learning algorithmic strategy, we utilized the Sklearn python library (scikit-learn Python library version 1.3) to introduce an unsupervised model for clustering stocks based on their returns and variance. Then, utilizing the elbow method, decided that 6 clusters would be optimal for our model (Radovanovic, 2022). Once the algorithm had done its initial clustering, we manually removed outliers and rechecked our clusters, running the model again.

Next, we selected the clusters which we believed would perform best with the Ichimoku Cloud technical indicators. For the purposes of this paper, we ended up using clusters 4 and 5 which can be seen in Appendix 6 to 9

Once the clusters were identified, all that remained was to implement our buy and sell signals from a function calculating the indicators signals. We would enter into a buy if our conversion line crossed above the baseline (Tenkan-sen crossed above Kijun-sen) and Span B was greater than Span A (Trading Strategy Guides, 2023). Conversely, a sell would be generated if we were currently holding the stock, the conversion line crossed below the baseline, and Span B was less than Span A.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Golden Cross

Arguably one of the simpler strategies we implemented, the Golden Cross is a well defined traditional technical indicator that utilizes two SMAs, one for 50 periods, and the other for 200 periods (Hayes, Golden Cross Pattern Explained With Examples and Charts, 2023). In order to recreate this as a trading strategy, we took the ratio between the 50 SMA and 200 SMA to create an easy method for determining crossovers between the indicators.

When the ratio between the two SMAs was greater than 1, this meant the 50 SMA had crossed above the 200 SMA from below, so we would enter into a buy position based on the next day’s closing price. Conversely, when the ratio was less than 1, this meant the 50 SMA had crossed below the 200 SMA from above, which would then generate a sell signal.

Finally, to prevent repeated buys from the indicator, we would only buy if our current position held 0 stock, and sell if our position held 1 stock. Else no actions would be taken by the algorithm.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Bollinger Bands

Finally, for our last strategy, we implemented a very basic interpretation of the Bollinger Bands. Similar to the Golden Cross strategy, we used two ratios for generating a trade signal for buying or selling. If the lower indicator was less than 1, we would purchase the security, and if the upper indicator was greater than 1, we would sell the security.

The bands were automatically calculated using the pandas\_ta library. However, we did have to change the default period value from 5, to 20, in order to match generally agreed standard for the center SMA line (Hayes, Bollinger Bands®: What They Are, and What They Tell Investors, 2023).
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Experimental Results & Analysis

Overall, the strategies we tested were able to show gross return multiples of: 26.38x for a Relative Strength Index and Pocket Pivot combination, 8.31x for a Golden Cross strategy, 7.45x for Momentum based selection, 3.73x for our machine learning approach utilizing Ichimoku Clouds, and 3.62x for Bollinger Bands, compared to the gross return multiples of 5.2x for the TDB908 mutual fund and 3.51x for the NASDAQ-100 Index.

These results were obtained by testing our trading algorithms buy and sell indicators on the collected historical data for the top 10 holdings of the selected mutual fund. Each strategy would be run through a loop for the individual companies, with results collected into a final table for analysis.

Then, once all 10 of the companies had run through each of the trading strategies and generated a trading returns table, we added 1 to all of these results and calculated the cumulative product of our returns. Finally, we took the average of our cumulative product across all of the companies to arrive at each strategies overall performance for the selected indicators. Summary tables for each strategy’s results can be seen in Appendix 1 to 16.

For an additional comparison measure, we also compared the return of a Naive strategy, which simply bought the stock on the first day, and sold it on the last observation period.

Unsurprisingly, this baseline strategy performed remarkably well, due to the nature of the top 10 holdings themselves of the mutual fund. Each of the companies in the top 10 holdings have seen incredible growth in their corporate value the past 10 years, such as Tesla, Microsoft, and NVIDIA to name a few.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Conclusion

To summarize, our results were able to show that even if some technical indicator strategies or machine learning methods didn’t perform better than the select mutual fund, they still provided valuable insights towards developing a rules-based approach for security selection and investing. Further areas of interest and research would be continuing to delve into incorporating machine learning models for asset selection, trading, or in combination with other technical indicators.

Additionally, other potential areas for future development would be investigating if machine learning models could develop new or unique technical analysis indicators for predicting price action themselves.

Nonetheless, machine learning and technical analysis integration will undoubtedly become a key area of focus in the future as asset valuation and trading strategies continue to evolve to meet increasing investor demands.
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## References

Awati, R. (2022, April 14). *Data Dredging (data fishing)*. Retrieved from Techtarget: https://www.techtarget.com/searchdatamanagement/definition/data-dredging

*Beautiful Soup Documentation*. (n.d.). Retrieved from crummy, version 4.12: Available at https://www.crummy.com/software/BeautifulSoup/bs4/doc/

CFI Team. (n.d.). *NASDAQ-100 Index*. Retrieved from Corporate Finance Institution: https://corporatefinanceinstitute.com/resources/equities/nasdaq-100-index/

Edwards, R. D., Magee, J., & Bassetti, W. (2007). *Technical Analysis of Stock Trends (9th ed.).* New York: AMACOM.

Fernando, J. (2023, March 31). *Relative Strength Index (RSI) Indicator Explained With Formula*. Retrieved from Investopedia: https://www.investopedia.com/terms/r/rsi.asp

Gordon, R. (2021). Technical Analysis & Machine Learning: Foreign Exchange Rate Forecasting with Artificial Nueral Networks. *Journal of Technical Analysis, 71*, 66-95.

Hayes, A. (2022, March 14). *Technical Analysis: What It Is and How to Use It in Investing*. Retrieved from Investopedia: https://www.investopedia.com/terms/t/technicalanalysis.asp#:~:text=Technical%20analysis%20as%20we%20know,helping%20to%20form%20its%20basis

Hayes, A. (2023, September 30). *Bollinger Bands®: What They Are, and What They Tell Investors*. Retrieved from Investopedia: https://www.investopedia.com/terms/b/bollingerbands.asp

Hayes, A. (2023, December 4). *Golden Cross Pattern Explained With Examples and Charts*. Retrieved from Investopedia: https://www.investopedia.com/terms/g/goldencross.asp

Market Technicians Association. (2002). *Journal of technical analysis*.

*Nasdaq-100*. (n.d.). Retrieved from Wikipedia: https://en.wikipedia.org/wiki/Nasdaq-100#Changes\_in\_2023

*Python Software Foundation*. (n.d.). Retrieved from Python Language Reference, version 3.10: Available at https://www.python.org/

Radovanovic, I. (2022, July 16). *Cluster Analysis – Machine Learning for Pairs Trading*. Retrieved from algotrading101: https://algotrading101.com/learn/cluster-analysis-guide/

*scikit-learn Python library version 1.3*. (n.d.). Retrieved from Available at https://scikit-learn.org/stable/

Shane. (2022, August 13). *Pocket Pivots – Trade Like A Stock Market Pro*. Retrieved from Netpicks: https://www.netpicks.com/pocket-pivots/

Trading Strategy Guides. (2023, March 15). *Best Ichimoku Cloud Strategy: Accelerate Profits In 5 Steps*. Retrieved from Trading Strategy Guides: https://tradingstrategyguides.com/best-ichimoku-strategy/#:~:text=The%20Ichimoku%20cloud%20strategy%20attempts,strength%20of%20these%20market%20signals

*yfinance Python library version 0.2.32*. (n.d.). Retrieved from Available at https://pypi.org/project/yfinance/
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Appendix

**Fig 1 – RSI & Pocket Pivot Individual Stock Growth**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.001.jpeg)

**Fig 2 – RSI & Pocket Pivot Returns Table**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.002.png)
**


**Fig 3 – RSI & Pocket Pivot Chart**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.003.jpeg)

**Fig 4 – Momentum Returns Table\***

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.004.png)

**\***The above cumulative product is based on returns from 2010 – 2014



**Fig 5 – Momentum Chart\***

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.005.png)

\*Based on the 5 year returns

**Fig 6 – K-Means Initial Clustering – Outliers**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.006.png)
**


**Fig 7 – K-Means Outliers Removed**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.007.jpeg)

**Fig 8 – K-Means & Ichimoku Cloud Cluster 4 Table**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.008.png)
**


**Fig 9 – K-Means Ichimoku Cloud Chart**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.009.png)

**Fig 10 – Golden Cross Returns Table**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.010.png)
**


**Fig 11 – Golden Cross Buy / Sell Signals**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.011.png)

**Fig 12 – Golden Cross Chart**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.012.png)
**


**Fig 13 – Bollinger Bands Returns Table**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.013.png)

**Fig 14 – Bollinger Bands Buy / Sell Signals**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.014.png)
**


**Fig 15 – Bollinger Bands Chart**

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.015.png)

**Fig 16 – All Strategy Returns\***

![](/img/Aspose.Words.daf8a82a-4601-42a2-8169-638afbf189eb.016.png)

\*Momentum was annualized by taking the 5<sup>th</sup> root of 2.73, then raising the result to the power of 10 to estimate 10 years’ worth of compounded returns.
<p align="right">(<a href="#readme-top">back to top</a>)</p>
