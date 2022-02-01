STAT4005 hw 4
================

# Heteroskedasticity

Heteroskedasticity means that the variance of the time series is
non-constant. It is a property commonly exhibited by stock data. For
heteroskedastic series, GARCH model should be used to fit the data
instead of ARIMA model to capture the non-constant variance. In this
notebook, we will compare the fit of two different model on HSBC stock
price on HKEX.

## Preprocessing

Import required libraries.

``` r
#install.packages("tseries")
#install.packages("forecast")
library(tseries)
```

    ## Registered S3 method overwritten by 'quantmod':
    ##   method            from
    ##   as.zoo.data.frame zoo

``` r
library(forecast)
```

Import and clean the stock data.

``` r
hsbc = get.hist.quote(instrument="0005.hk", start="2016-01-01", end="2021-12-31", quote=c("AdjClose"))
```

    ## 'getSymbols' currently uses auto.assign=TRUE by default, but will
    ## use auto.assign=FALSE in 0.5-0. You will still be able to use
    ## 'loadSymbols' to automatically load data. getOption("getSymbols.env")
    ## and getOption("getSymbols.auto.assign") will still be checked for
    ## alternate defaults.
    ## 
    ## This message is shown once per session and may be disabled by setting 
    ## options("getSymbols.warning4.0"=FALSE). See ?getSymbols for details.

    ## time series starts 2016-01-04
    ## time series ends   2021-12-30

``` r
hsbc[is.na(hsbc)==T] = mean(hsbc, na.rm=T)
y = as.numeric(log(hsbc))
par(mfrow=c(1,2))
ts.plot(y)
acf(y)
```

![](HSBC_files/figure-gfm/part%202a-1.png)<!-- --> ## ARIMA fitting The
ACF plot shows that there is a trend within the time series. To
establish stationarity for further analysis, we can perform differencing
on the series.

``` r
diffy = as.numeric(diff(log(hsbc)))
par(mfrow=c(1,2))
ts.plot(diffy)
acf(diffy)
```

![](HSBC_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

From the ACF plot, we can see that for k \> 0, all ACF(k) is within the
confidence interval. This suggests ARIMA(0, 0, 0) should best fit the
differenced data. The hypothesis can be confirmed using auto ARIMA
fitting, showing that ARIMA(0, 1, 0) best fits the original data.

``` r
model.arima = auto.arima(y, ic="aicc")
model.arima
```

    ## Series: y 
    ## ARIMA(0,1,0) 
    ## 
    ## sigma^2 estimated as 0.0002056:  log likelihood=4179.46
    ## AIC=-8356.92   AICc=-8356.92   BIC=-8351.62

However, residual-squared of ARIMA(0, 0, 0) is not white noise. This
suggests that the residual is also not white noise. As the residual
still contains information, some other model should be used to capture
those information.

``` r
res.arima = model.arima$res
par(mfrow=c(1,2))
acf(res.arima)
acf(res.arima^2)
```

![](HSBC_files/figure-gfm/part%202c-1.png)<!-- -->

Before fitting a GARCH model, we should test for GARCH effect using
Lagrange Multiplier test. Since p-value = 0 \< 0.05, we can reject null
hypothesis and conclude that that GARCH effect is significant.

``` r
n = length(res.arima)
res.sq.arima = res.arima^2
df.garch = data.frame(
rsq = res.sq.arima,
rsq1 = c(rep(0, 1), res.sq.arima[1:(n-1)]),
rsq2 = c(rep(0, 2), res.sq.arima[1:(n-2)]),
rsq3 = c(rep(0, 3), res.sq.arima[1:(n-3)]),
rsq4 = c(rep(0, 4), res.sq.arima[1:(n-4)])
)
model.garch.effect = lm(formula = rsq ~ ., data = df.garch)
summary(model.garch.effect)
```

    ## 
    ## Call:
    ## lm(formula = rsq ~ ., data = df.garch)
    ## 
    ## Residuals:
    ##        Min         1Q     Median         3Q        Max 
    ## -0.0008522 -0.0001651 -0.0001372 -0.0000267  0.0097987 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 1.501e-04  1.733e-05   8.660  < 2e-16 ***
    ## rsq1        7.378e-02  2.597e-02   2.841  0.00457 ** 
    ## rsq2        6.205e-02  2.599e-02   2.387  0.01710 *  
    ## rsq3        6.401e-02  2.599e-02   2.463  0.01391 *  
    ## rsq4        6.976e-02  2.597e-02   2.686  0.00731 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.0005592 on 1475 degrees of freedom
    ## Multiple R-squared:  0.0229, Adjusted R-squared:  0.02025 
    ## F-statistic: 8.642 on 4 and 1475 DF,  p-value: 6.807e-07

We can find the best GARCH(p, q) model by trying out combinations of p
and q and selecting the one which minimises AIC.

``` r
df.order <- data.frame(p = c(0, 1, 0, 1, 2, 1, 2), q = c(1, 0, 2, 1, 0, 2, 1))
vec.aic.garch = apply(df.order, 1,
function(order) {
  p <- order[1]; q <- order[2]
  AIC(garch(res.arima, order = c(p, q)))
} )
```

    ## 
    ##  ***** ESTIMATION WITH ANALYTICAL GRADIENT ***** 
    ## 
    ## 
    ##      I     INITIAL X(I)        D(I)
    ## 
    ##      1     1.953214e-04     1.000e+00
    ##      2     5.000000e-02     1.000e+00
    ## 
    ##     IT   NF      F         RELDF    PRELDF    RELDX   STPPAR   D*STEP   NPRELDF
    ##      0    1 -5.552e+03
    ##      1    8 -5.552e+03  3.07e-05  5.91e-05  4.3e-05  1.8e+10  4.3e-06  5.35e+05
    ##      2   16 -5.557e+03  9.75e-04  1.62e-03  3.8e-01  2.0e+00  6.2e-02  1.89e+00
    ##      3   17 -5.558e+03  2.00e-04  1.38e-04  1.2e-01  0.0e+00  3.0e-02  1.38e-04
    ##      4   18 -5.559e+03  7.82e-05  6.34e-05  9.4e-02  0.0e+00  2.9e-02  6.34e-05
    ##      5   19 -5.559e+03  8.17e-06  7.12e-06  3.0e-02  0.0e+00  1.1e-02  7.12e-06
    ##      6   20 -5.559e+03  2.83e-07  2.75e-07  6.8e-03  0.0e+00  2.5e-03  2.75e-07
    ##      7   21 -5.559e+03  1.65e-09  1.79e-09  3.0e-04  0.0e+00  1.1e-04  1.79e-09
    ##      8   22 -5.559e+03  1.38e-11  1.38e-11  1.3e-07  0.0e+00  4.8e-08  1.38e-11
    ## 
    ##  ***** RELATIVE FUNCTION CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.558738e+03   RELDX        1.294e-07
    ##  FUNC. EVALS      22         GRAD. EVALS       9
    ##  PRELDF       1.385e-11      NPRELDF      1.385e-11
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    1.722582e-04     1.000e+00    -1.667e-01
    ##      2    1.840331e-01     1.000e+00     3.111e-05
    ## 
    ## 
    ##  ***** ESTIMATION WITH ANALYTICAL GRADIENT ***** 
    ## 
    ## 
    ##      I     INITIAL X(I)        D(I)
    ## 
    ##      1     1.953214e-04     1.000e+00
    ##      2     5.000000e-02     1.000e+00
    ## 
    ##     IT   NF      F         RELDF    PRELDF    RELDX   STPPAR   D*STEP   NPRELDF
    ##      0    1 -5.539e+03
    ##      1   11 -5.539e+03  2.93e-11  1.32e-10  4.5e-08  3.6e+10  4.5e-09  2.36e+00
    ##      2   23 -5.539e+03  1.48e-15  5.88e-15  4.4e-12  8.7e+03  4.4e-13  2.63e-08
    ##      3   29 -5.539e+03 -1.81e-15  7.39e-18  5.5e-15  6.9e+06  5.5e-16  2.63e-08
    ## 
    ##  ***** FALSE CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.538570e+03   RELDX        5.528e-15
    ##  FUNC. EVALS      29         GRAD. EVALS       3
    ##  PRELDF       7.390e-18      NPRELDF      2.630e-08
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    1.953169e-04     1.000e+00     7.404e+01
    ##      2    5.000000e-02     1.000e+00    -1.845e-03

    ## Warning in garch(res.arima, order = c(p, q)): singular information

    ## 
    ##  ***** ESTIMATION WITH ANALYTICAL GRADIENT ***** 
    ## 
    ## 
    ##      I     INITIAL X(I)        D(I)
    ## 
    ##      1     1.850413e-04     1.000e+00
    ##      2     5.000000e-02     1.000e+00
    ##      3     5.000000e-02     1.000e+00
    ## 
    ##     IT   NF      F         RELDF    PRELDF    RELDX   STPPAR   D*STEP   NPRELDF
    ##      0    1 -5.561e+03
    ##      1    7 -5.561e+03  1.29e-04  2.85e-04  1.0e-04  1.6e+10  1.0e-05  2.26e+06
    ##      2    8 -5.561e+03  2.47e-06  2.69e-06  7.7e-05  2.0e+00  1.0e-05  4.97e+00
    ##      3   16 -5.578e+03  2.89e-03  4.99e-03  4.8e-01  2.0e+00  1.2e-01  4.96e+00
    ##      4   18 -5.578e+03  1.53e-04  2.03e-04  4.2e-02  2.0e+00  1.7e-02  4.45e-03
    ##      5   19 -5.578e+03  2.47e-06  2.49e-05  4.3e-02  2.0e+00  1.7e-02  3.91e-03
    ##      6   24 -5.578e+03  2.98e-06  5.58e-06  4.1e-06  4.3e+01  1.2e-06  7.82e-04
    ##      7   34 -5.582e+03  6.54e-04  4.53e-04  1.4e-01  0.0e+00  6.4e-02  4.53e-04
    ##      8   35 -5.583e+03  2.51e-04  2.05e-04  1.1e-01  0.0e+00  5.8e-02  2.05e-04
    ##      9   36 -5.584e+03  2.78e-05  2.41e-05  3.4e-02  0.0e+00  2.1e-02  2.41e-05
    ##     10   37 -5.584e+03  1.11e-06  1.07e-06  8.3e-03  0.0e+00  5.5e-03  1.07e-06
    ##     11   38 -5.584e+03  1.01e-08  1.10e-08  3.7e-04  0.0e+00  2.3e-04  1.10e-08
    ##     12   39 -5.584e+03  1.84e-10  1.59e-10  5.0e-05  0.0e+00  2.8e-05  1.59e-10
    ##     13   40 -5.584e+03  1.51e-11  1.29e-11  2.4e-05  0.0e+00  1.6e-05  1.29e-11
    ## 
    ##  ***** RELATIVE FUNCTION CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.583650e+03   RELDX        2.382e-05
    ##  FUNC. EVALS      40         GRAD. EVALS      14
    ##  PRELDF       1.291e-11      NPRELDF      1.291e-11
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    1.273226e-04     1.000e+00    -6.789e+00
    ##      2    2.130176e-01     1.000e+00    -1.825e-03
    ##      3    2.606316e-01     1.000e+00     7.531e-04
    ## 
    ## 
    ##  ***** ESTIMATION WITH ANALYTICAL GRADIENT ***** 
    ## 
    ## 
    ##      I     INITIAL X(I)        D(I)
    ## 
    ##      1     1.850413e-04     1.000e+00
    ##      2     5.000000e-02     1.000e+00
    ##      3     5.000000e-02     1.000e+00
    ## 
    ##     IT   NF      F         RELDF    PRELDF    RELDX   STPPAR   D*STEP   NPRELDF
    ##      0    1 -5.553e+03
    ##      1    8 -5.553e+03  3.90e-05  7.52e-05  4.6e-05  2.0e+10  4.6e-06  7.52e+05
    ##      2   16 -5.560e+03  1.34e-03  2.07e-03  4.1e-01  2.0e+00  6.9e-02  2.53e+00
    ##      3   19 -5.584e+03  4.24e-03  3.08e-03  6.9e-01  1.9e+00  2.8e-01  4.00e-01
    ##      4   21 -5.591e+03  1.31e-03  1.19e-03  7.6e-02  2.0e+00  5.5e-02  1.04e+02
    ##      5   23 -5.610e+03  3.36e-03  2.84e-03  1.2e-01  2.0e+00  1.1e-01  1.08e+04
    ##      6   25 -5.631e+03  3.64e-03  3.45e-03  8.9e-02  2.0e+00  9.8e-02  2.01e+06
    ##      7   27 -5.635e+03  7.32e-04  7.23e-04  1.6e-02  2.0e+00  2.0e-02  1.18e+04
    ##      8   37 -5.636e+03  2.68e-04  5.58e-04  3.5e-06  3.9e+00  4.2e-06  1.37e+03
    ##      9   38 -5.636e+03  6.32e-06  5.28e-06  3.4e-06  2.0e+00  4.2e-06  7.51e+02
    ##     10   47 -5.644e+03  1.44e-03  1.54e-03  3.1e-02  2.0e+00  3.9e-02  7.62e+02
    ##     11   54 -5.644e+03  4.29e-06  8.88e-06  3.3e-07  2.2e+01  4.2e-07  5.13e-02
    ##     12   64 -5.653e+03  1.45e-03  2.13e-03  4.8e-02  1.9e+00  6.5e-02  3.95e-02
    ##     13   66 -5.666e+03  2.28e-03  1.35e-03  6.3e-02  0.0e+00  1.2e-01  1.35e-03
    ##     14   68 -5.669e+03  6.56e-04  9.61e-04  1.8e-02  1.8e+00  3.8e-02  8.47e-03
    ##     15   71 -5.687e+03  3.12e-03  3.11e-03  6.6e-02  3.7e-01  1.5e-01  9.41e-03
    ##     16   82 -5.688e+03  2.15e-04  7.40e-04  5.3e-07  3.3e+00  9.8e-07  1.28e-03
    ##     17   83 -5.689e+03  7.09e-05  5.89e-05  4.7e-07  2.0e+00  9.8e-07  1.01e-03
    ##     18   84 -5.689e+03  2.09e-06  2.10e-06  4.7e-07  2.0e+00  9.8e-07  1.32e-03
    ##     19   85 -5.689e+03  2.62e-08  3.99e-08  4.8e-07  2.0e+00  9.8e-07  1.29e-03
    ##     20   94 -5.691e+03  3.45e-04  3.65e-04  7.7e-03  4.9e-01  1.6e-02  1.29e-03
    ##     21   95 -5.691e+03  1.14e-04  1.78e-04  6.7e-03  1.6e+00  1.6e-02  1.12e-03
    ##     22   97 -5.692e+03  3.39e-05  4.80e-05  1.4e-03  9.9e-01  2.7e-03  1.08e-04
    ##     23   98 -5.692e+03  8.83e-07  1.25e-06  1.4e-04  0.0e+00  2.7e-04  1.25e-06
    ##     24   99 -5.692e+03  1.36e-07  3.29e-07  5.0e-05  0.0e+00  1.0e-04  3.29e-07
    ##     25  100 -5.692e+03  1.09e-07  1.85e-07  3.8e-05  0.0e+00  8.4e-05  1.85e-07
    ##     26  101 -5.692e+03  3.23e-09  7.92e-10  3.4e-06  0.0e+00  7.1e-06  7.92e-10
    ##     27  102 -5.692e+03 -1.44e-10  1.15e-11  4.4e-07  0.0e+00  8.5e-07  1.15e-11
    ## 
    ##  ***** RELATIVE FUNCTION CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.691511e+03   RELDX        4.446e-07
    ##  FUNC. EVALS     102         GRAD. EVALS      27
    ##  PRELDF       1.153e-11      NPRELDF      1.153e-11
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    1.813480e-06     1.000e+00     1.227e+03
    ##      2    3.876590e-02     1.000e+00     1.850e-01
    ##      3    9.513883e-01     1.000e+00     2.073e-01
    ## 
    ## 
    ##  ***** ESTIMATION WITH ANALYTICAL GRADIENT ***** 
    ## 
    ## 
    ##      I     INITIAL X(I)        D(I)
    ## 
    ##      1     1.850413e-04     1.000e+00
    ##      2     5.000000e-02     1.000e+00
    ##      3     5.000000e-02     1.000e+00
    ## 
    ##     IT   NF      F         RELDF    PRELDF    RELDX   STPPAR   D*STEP   NPRELDF
    ##      0    1 -5.534e+03
    ##      1   10 -5.534e+03  4.97e-09  1.46e-08  5.2e-07  3.0e+10  5.2e-08  2.22e+02
    ##      2   31 -5.534e+03  8.22e-16  2.61e-16  3.2e-14  6.5e+06  3.2e-15  1.78e-06
    ##      3   33 -5.534e+03 -9.86e-16  7.42e-17  9.1e-15  2.3e+07  9.1e-16  1.78e-06
    ## 
    ##  ***** FALSE CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.534493e+03   RELDX        9.099e-15
    ##  FUNC. EVALS      33         GRAD. EVALS       3
    ##  PRELDF       7.425e-17      NPRELDF      1.779e-06
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    1.850930e-04     1.000e+00    -4.516e+02
    ##      2    5.000000e-02     1.000e+00     5.296e-03
    ##      3    5.000000e-02     1.000e+00     7.177e-03

    ## Warning in garch(res.arima, order = c(p, q)): singular information

    ## 
    ##  ***** ESTIMATION WITH ANALYTICAL GRADIENT ***** 
    ## 
    ## 
    ##      I     INITIAL X(I)        D(I)
    ## 
    ##      1     1.747612e-04     1.000e+00
    ##      2     5.000000e-02     1.000e+00
    ##      3     5.000000e-02     1.000e+00
    ##      4     5.000000e-02     1.000e+00
    ## 
    ##     IT   NF      F         RELDF    PRELDF    RELDX   STPPAR   D*STEP   NPRELDF
    ##      0    1 -5.562e+03
    ##      1    7 -5.563e+03  1.45e-04  3.18e-04  1.0e-04  1.8e+10  1.0e-05  2.81e+06
    ##      2    8 -5.563e+03  2.49e-06  2.73e-06  7.7e-05  2.0e+00  1.0e-05  5.73e+00
    ##      3   16 -5.582e+03  3.34e-03  5.76e-03  5.0e-01  2.0e+00  1.3e-01  5.72e+00
    ##      4   18 -5.592e+03  1.91e-03  1.67e-03  3.9e-01  2.0e+00  1.3e-01  3.54e-01
    ##      5   20 -5.614e+03  3.88e-03  3.81e-03  4.0e-01  2.0e+00  2.5e-01  8.47e+01
    ##      6   22 -5.621e+03  1.16e-03  9.18e-04  5.1e-02  2.0e+00  5.1e-02  6.45e+00
    ##      7   24 -5.621e+03  1.53e-04  2.38e-03  4.2e-02  2.0e+00  4.5e-02  1.19e+01
    ##      8   25 -5.631e+03  1.62e-03  3.16e-03  2.0e-02  2.0e+00  2.3e-02  5.43e+00
    ##      9   27 -5.636e+03  8.90e-04  9.46e-04  5.1e-02  2.0e+00  6.3e-02  5.03e+00
    ##     10   33 -5.636e+03  5.26e-05  8.30e-05  1.1e-06  9.3e+00  1.3e-06  2.01e-01
    ##     11   34 -5.636e+03  1.18e-06  1.01e-06  9.8e-07  2.0e+00  1.3e-06  1.66e-02
    ##     12   50 -5.636e+03 -4.20e-15  8.86e-15  9.9e-15  1.7e+00  1.2e-14 -2.06e-03
    ## 
    ##  ***** FALSE CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.635842e+03   RELDX        9.935e-15
    ##  FUNC. EVALS      50         GRAD. EVALS      12
    ##  PRELDF       8.855e-15      NPRELDF     -2.064e-03
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    2.512664e-05     1.000e+00    -4.095e+03
    ##      2    8.894633e-02     1.000e+00     6.175e+01
    ##      3    2.672710e-01     1.000e+00     1.333e+02
    ##      4    6.125554e-01     1.000e+00    -4.208e+01
    ## 
    ## 
    ##  ***** ESTIMATION WITH ANALYTICAL GRADIENT ***** 
    ## 
    ## 
    ##      I     INITIAL X(I)        D(I)
    ## 
    ##      1     1.747612e-04     1.000e+00
    ##      2     5.000000e-02     1.000e+00
    ##      3     5.000000e-02     1.000e+00
    ##      4     5.000000e-02     1.000e+00
    ## 
    ##     IT   NF      F         RELDF    PRELDF    RELDX   STPPAR   D*STEP   NPRELDF
    ##      0    1 -5.550e+03
    ##      1    8 -5.550e+03  4.92e-05  9.43e-05  4.9e-05  2.2e+10  4.9e-06  1.04e+06
    ##      2   16 -5.561e+03  1.89e-03  2.78e-03  4.4e-01  2.0e+00  8.0e-02  3.38e+00
    ##      3   19 -5.622e+03  1.09e-02  6.09e-03  5.6e-01  2.0e+00  3.2e-01  1.45e+00
    ##      4   21 -5.637e+03  2.71e-03  1.77e-03  4.9e-02  2.0e+00  3.2e-02  1.27e+00
    ##      5   22 -5.656e+03  3.25e-03  3.67e-03  8.6e-02  2.0e+00  6.4e-02  8.90e+00
    ##      6   24 -5.656e+03  9.11e-05  3.05e-04  9.2e-03  2.0e+00  7.4e-03  5.18e+00
    ##      7   25 -5.657e+03  2.38e-04  3.09e-04  9.1e-03  2.0e+00  7.4e-03  1.48e+00
    ##      8   26 -5.658e+03  9.56e-05  1.20e-04  9.0e-03  2.0e+00  7.4e-03  5.38e-01
    ##      9   27 -5.658e+03  2.04e-05  5.85e-05  9.2e-03  2.0e+00  7.4e-03  3.39e-02
    ##     10   28 -5.658e+03  3.56e-05  4.18e-05  7.1e-03  2.0e+00  7.4e-03  7.03e-03
    ##     11   31 -5.659e+03  1.45e-04  1.95e-04  9.0e-02  1.9e+00  1.1e-01  5.54e-03
    ##     12   33 -5.659e+03  4.35e-05  9.86e-05  8.3e-03  1.9e+00  1.1e-02  1.75e-03
    ##     13   37 -5.678e+03  3.23e-03  1.76e-03  1.3e-01  0.0e+00  2.0e-01  1.76e-03
    ##     14   40 -5.678e+03  1.27e-04  7.30e-04  2.3e-02  1.9e+00  3.8e-02  1.46e-02
    ##     15   41 -5.683e+03  7.31e-04  9.75e-04  2.2e-02  1.8e+00  3.8e-02  5.24e-03
    ##     16   43 -5.684e+03  3.15e-04  3.76e-04  2.3e-02  9.7e-01  3.8e-02  1.86e-03
    ##     17   45 -5.688e+03  5.60e-04  6.33e-04  2.4e-02  9.6e-01  3.8e-02  4.91e-03
    ##     18   47 -5.689e+03  1.93e-04  2.57e-04  1.0e-02  9.9e-01  1.7e-02  5.62e-04
    ##     19   48 -5.689e+03  2.48e-06  1.23e-05  5.1e-03  0.0e+00  9.6e-03  1.23e-05
    ##     20   49 -5.689e+03  3.81e-06  8.14e-06  4.6e-04  0.0e+00  8.7e-04  8.14e-06
    ##     21   50 -5.689e+03  1.92e-06  2.03e-06  4.3e-04  1.5e+00  8.7e-04  2.42e-06
    ##     22   51 -5.689e+03  1.02e-07  5.10e-08  3.7e-04  0.0e+00  7.0e-04  5.10e-08
    ##     23   53 -5.689e+03  9.08e-10  1.32e-08  1.5e-04  7.7e-01  2.8e-04  2.06e-08
    ##     24   55 -5.689e+03  7.11e-10  1.21e-09  2.5e-05  1.6e+00  4.8e-05  7.84e-09
    ##     25   58 -5.689e+03  1.42e-10  2.63e-10  5.7e-06  1.9e+00  1.1e-05  6.91e-09
    ##     26   62 -5.689e+03  1.15e-11  2.09e-11  4.6e-07  2.0e+00  8.8e-07  6.64e-09
    ##     27   67 -5.689e+03  3.16e-13  5.29e-13  1.2e-08  2.0e+00  2.2e-08  6.62e-09
    ##     28   75 -5.689e+03 -3.53e-14  1.63e-18  5.7e-15  2.0e+01  9.8e-15  6.62e-09
    ## 
    ##  ***** FALSE CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.688658e+03   RELDX        5.666e-15
    ##  FUNC. EVALS      75         GRAD. EVALS      28
    ##  PRELDF       1.634e-18      NPRELDF      6.625e-09
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    3.005348e-06     1.000e+00    -8.534e-01
    ##      2    6.651099e-02     1.000e+00    -3.927e-01
    ##      3    1.591856e-01     1.000e+00     1.601e-01
    ##      4    7.580893e-01     1.000e+00     2.000e-01

    ## Warning in garch(res.arima, order = c(p, q)): singular information

From the following plot, we can see that the 4th combination (1, 1)
minimises AIC.

``` r
par(mfrow = c(1, 1))
plot(1:nrow(df.order), vec.aic.garch)
```

![](HSBC_files/figure-gfm/unnamed-chunk-4-1.png)<!-- --> Therefore, we
fit GARCH(1, 1) to the residual of the ARIMA fitting.

``` r
model.garch = garch(res.arima, order = c(1, 1))
```

    ## 
    ##  ***** ESTIMATION WITH ANALYTICAL GRADIENT ***** 
    ## 
    ## 
    ##      I     INITIAL X(I)        D(I)
    ## 
    ##      1     1.850413e-04     1.000e+00
    ##      2     5.000000e-02     1.000e+00
    ##      3     5.000000e-02     1.000e+00
    ## 
    ##     IT   NF      F         RELDF    PRELDF    RELDX   STPPAR   D*STEP   NPRELDF
    ##      0    1 -5.553e+03
    ##      1    8 -5.553e+03  3.90e-05  7.52e-05  4.6e-05  2.0e+10  4.6e-06  7.52e+05
    ##      2   16 -5.560e+03  1.34e-03  2.07e-03  4.1e-01  2.0e+00  6.9e-02  2.53e+00
    ##      3   19 -5.584e+03  4.24e-03  3.08e-03  6.9e-01  1.9e+00  2.8e-01  4.00e-01
    ##      4   21 -5.591e+03  1.31e-03  1.19e-03  7.6e-02  2.0e+00  5.5e-02  1.04e+02
    ##      5   23 -5.610e+03  3.36e-03  2.84e-03  1.2e-01  2.0e+00  1.1e-01  1.08e+04
    ##      6   25 -5.631e+03  3.64e-03  3.45e-03  8.9e-02  2.0e+00  9.8e-02  2.01e+06
    ##      7   27 -5.635e+03  7.32e-04  7.23e-04  1.6e-02  2.0e+00  2.0e-02  1.18e+04
    ##      8   37 -5.636e+03  2.68e-04  5.58e-04  3.5e-06  3.9e+00  4.2e-06  1.37e+03
    ##      9   38 -5.636e+03  6.32e-06  5.28e-06  3.4e-06  2.0e+00  4.2e-06  7.51e+02
    ##     10   47 -5.644e+03  1.44e-03  1.54e-03  3.1e-02  2.0e+00  3.9e-02  7.62e+02
    ##     11   54 -5.644e+03  4.29e-06  8.88e-06  3.3e-07  2.2e+01  4.2e-07  5.13e-02
    ##     12   64 -5.653e+03  1.45e-03  2.13e-03  4.8e-02  1.9e+00  6.5e-02  3.95e-02
    ##     13   66 -5.666e+03  2.28e-03  1.35e-03  6.3e-02  0.0e+00  1.2e-01  1.35e-03
    ##     14   68 -5.669e+03  6.56e-04  9.61e-04  1.8e-02  1.8e+00  3.8e-02  8.47e-03
    ##     15   71 -5.687e+03  3.12e-03  3.11e-03  6.6e-02  3.7e-01  1.5e-01  9.41e-03
    ##     16   82 -5.688e+03  2.15e-04  7.40e-04  5.3e-07  3.3e+00  9.8e-07  1.28e-03
    ##     17   83 -5.689e+03  7.09e-05  5.89e-05  4.7e-07  2.0e+00  9.8e-07  1.01e-03
    ##     18   84 -5.689e+03  2.09e-06  2.10e-06  4.7e-07  2.0e+00  9.8e-07  1.32e-03
    ##     19   85 -5.689e+03  2.62e-08  3.99e-08  4.8e-07  2.0e+00  9.8e-07  1.29e-03
    ##     20   94 -5.691e+03  3.45e-04  3.65e-04  7.7e-03  4.9e-01  1.6e-02  1.29e-03
    ##     21   95 -5.691e+03  1.14e-04  1.78e-04  6.7e-03  1.6e+00  1.6e-02  1.12e-03
    ##     22   97 -5.692e+03  3.39e-05  4.80e-05  1.4e-03  9.9e-01  2.7e-03  1.08e-04
    ##     23   98 -5.692e+03  8.83e-07  1.25e-06  1.4e-04  0.0e+00  2.7e-04  1.25e-06
    ##     24   99 -5.692e+03  1.36e-07  3.29e-07  5.0e-05  0.0e+00  1.0e-04  3.29e-07
    ##     25  100 -5.692e+03  1.09e-07  1.85e-07  3.8e-05  0.0e+00  8.4e-05  1.85e-07
    ##     26  101 -5.692e+03  3.23e-09  7.92e-10  3.4e-06  0.0e+00  7.1e-06  7.92e-10
    ##     27  102 -5.692e+03 -1.44e-10  1.15e-11  4.4e-07  0.0e+00  8.5e-07  1.15e-11
    ## 
    ##  ***** RELATIVE FUNCTION CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.691511e+03   RELDX        4.446e-07
    ##  FUNC. EVALS     102         GRAD. EVALS      27
    ##  PRELDF       1.153e-11      NPRELDF      1.153e-11
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    1.813480e-06     1.000e+00     1.227e+03
    ##      2    3.876590e-02     1.000e+00     1.850e-01
    ##      3    9.513883e-01     1.000e+00     2.073e-01

``` r
summary(model.garch)
```

    ## 
    ## Call:
    ## garch(x = res.arima, order = c(1, 1))
    ## 
    ## Model:
    ## GARCH(1,1)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -5.1492 -0.5448  0.0000  0.4979  7.2931 
    ## 
    ## Coefficient(s):
    ##     Estimate  Std. Error  t value Pr(>|t|)    
    ## a0 1.814e-06   3.277e-07    5.533 3.14e-08 ***
    ## a1 3.877e-02   4.067e-03    9.533  < 2e-16 ***
    ## b1 9.514e-01   5.148e-03  184.792  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Diagnostic Tests:
    ##  Jarque Bera Test
    ## 
    ## data:  Residuals
    ## X-squared = 1112.5, df = 2, p-value < 2.2e-16
    ## 
    ## 
    ##  Box-Ljung test
    ## 
    ## data:  Squared.Residuals
    ## X-squared = 0.017004, df = 1, p-value = 0.8962

The time series plot and ACF plots show that the residual of GARCH(1, 1)
is indeed white noise. Moreover, the residual is normally distributed
because normal quantile-quantile plot shows approximately a straight
line.

``` r
res.garch = residuals(model.garch)
res.garch = na.omit(res.garch)
par(mfrow = c(2, 2))
ts.plot(res.garch)
acf(res.garch)
acf(res.garch^2)
qqnorm(res.garch)
```

![](HSBC_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->
