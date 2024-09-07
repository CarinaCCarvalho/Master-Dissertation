# Master-Dissertation
Topological Data Analysis in Financial Asset Management


In this dissertation, we present an application of Topological Data Analysis (TDA) in a financial context, specifically addressing the problem of asset allocation in portfolio construction. We explore an optimization method by applying topological data analysis concepts to Enhanced Indexing. In this document, we present fundamental concepts and properties of topological data analysis. To develop the enhanced indexing method with topological data analysis, we begin by using the Takens' embedding method to reconstruct the time series of returns of a financial index in a high-dimensional space. We generate point cloud data sets and associate them with a topological space. Then we apply persistent homology to discover the topological patterns that appear in the multidimensional time series. The TDA norm allows a new approach to studying time series volatility. A two-step TDA optimization model for enhanced indexing is presented. In the first step, the assets that make up the market index are filtered and divided into three groups. This strategy has proven beneficial for investors. In the second step, we apply the optimization model to build a portfolio from the filtered asset groups for enhanced indexing. We applied statistical measures to the results to evaluate their performance on four financial indices. The results showed that the TDA model offers superior performance on several measures, including average excess returns over the benchmark index, compared to some enhanced indexing models.

Important: Optimizing the code was not a priority at this stage. My goal was primarily to ensure that the solution worked as expected.
During the year I did my thesis, I learned python, and the subject of topological data analysis!!

About the files:


- 'tdabins'
  Topological Data Analysis Model: I divided the assets of an index into 3 categories, 
and applied the TDA model to each of them, and observed their statistical analysis.


- 'vr'
  This code generates and visualizes the Vietoris-Rips complex for different values of ε, connecting points by edges and filling triangles in a two-dimensional space, based on the distance between the points.

- 'markowitz'
 implementing the Markowitz model.

- 'Naive'
  implementing the naive model.

IMPORTANT: My dissertation has more important codes. However, I am currently developing and improving the TDA model by adding costs and choosing the best parameters. 
My thesis can be found on Universidade de Coimbra scientific repository.

