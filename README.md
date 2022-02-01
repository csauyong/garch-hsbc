Heteroskedasticity
================

# Introduction

Heteroskedasticity means that the variance of the time series is
non-constant. It is a property commonly exhibited by stock data. For
heteroskedastic series, GARCH model should be used to fit the data
instead of ARIMA model to capture the non-constant variance. In this
notebook, we will compare the fit of two different model on HSBC stock
price on HKEX.

# Preprocessing

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

![](HSBC_files/figure-gfm/part%202a-1.png)<!-- --> # ARIMA fitting The
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

![](HSBC_files/figure-gfm/part%202c-1.png)<!-- --> # GARCH fitting
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
    ## F-statistic: 8.642 on 4 and 1475 DF,  p-value: 6.806e-07

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
    ##  FUNCTION    -5.558737e+03   RELDX        1.295e-07
    ##  FUNC. EVALS      22         GRAD. EVALS       9
    ##  PRELDF       1.385e-11      NPRELDF      1.385e-11
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    1.722583e-04     1.000e+00    -1.667e-01
    ##      2    1.840324e-01     1.000e+00     3.111e-05
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
    ##      2   23 -5.539e+03  1.97e-15  5.62e-15  4.2e-12  9.1e+03  4.2e-13  2.63e-08
    ##      3   28 -5.539e+03 -1.81e-15  2.39e-17  1.8e-14  2.1e+06  1.8e-15  2.63e-08
    ## 
    ##  ***** FALSE CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.538570e+03   RELDX        1.786e-14
    ##  FUNC. EVALS      28         GRAD. EVALS       3
    ##  PRELDF       2.388e-17      NPRELDF      2.630e-08
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
    ##      5   19 -5.578e+03  2.47e-06  2.49e-05  4.3e-02  2.0e+00  1.7e-02  3.90e-03
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
    ##  FUNCTION    -5.583650e+03   RELDX        2.381e-05
    ##  FUNC. EVALS      40         GRAD. EVALS      14
    ##  PRELDF       1.291e-11      NPRELDF      1.291e-11
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    1.273226e-04     1.000e+00    -6.787e+00
    ##      2    2.130165e-01     1.000e+00    -1.825e-03
    ##      3    2.606324e-01     1.000e+00     7.530e-04
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
    ##      6   25 -5.631e+03  3.64e-03  3.44e-03  8.9e-02  2.0e+00  9.8e-02  2.01e+06
    ##      7   27 -5.635e+03  7.31e-04  7.22e-04  1.6e-02  2.0e+00  2.0e-02  1.21e+04
    ##      8   37 -5.636e+03  2.56e-04  6.02e-04  3.9e-06  3.6e+00  4.7e-06  1.38e+03
    ##      9   38 -5.636e+03  1.88e-05  1.59e-05  3.7e-06  2.0e+00  4.7e-06  7.52e+02
    ##     10   39 -5.636e+03  6.61e-07  5.98e-07  3.8e-06  2.0e+00  4.7e-06  7.73e+02
    ##     11   48 -5.650e+03  2.40e-03  3.00e-03  5.8e-02  2.0e+00  7.7e-02  7.71e+02
    ##     12   49 -5.658e+03  1.43e-03  1.59e-03  5.1e-02  7.9e-01  7.7e-02  2.47e-03
    ##     13   51 -5.669e+03  1.96e-03  2.27e-03  8.5e-02  0.0e+00  1.9e-01  2.27e-03
    ##     14   63 -5.676e+03  1.25e-03  1.99e-03  1.5e-06  2.6e+00  2.7e-06  1.11e-02
    ##     15   64 -5.677e+03  7.76e-05  1.52e-04  1.2e-06  2.0e+00  2.7e-06  8.58e-03
    ##     16   65 -5.677e+03  2.05e-05  1.80e-05  1.2e-06  2.0e+00  2.7e-06  1.13e-02
    ##     17   66 -5.677e+03  3.35e-07  3.04e-07  1.2e-06  2.0e+00  2.7e-06  1.17e-02
    ##     18   75 -5.684e+03  1.27e-03  1.14e-03  1.9e-02  1.0e+00  4.5e-02  1.17e-02
    ##     19   77 -5.685e+03  2.62e-04  2.56e-04  3.7e-03  2.0e+00  8.9e-03  5.09e-01
    ##     20   79 -5.688e+03  5.05e-04  5.10e-04  7.3e-03  2.0e+00  1.8e-02  2.02e+00
    ##     21   80 -5.691e+03  5.12e-04  7.67e-04  1.4e-02  1.8e+00  3.6e-02  9.14e-03
    ##     22   82 -5.691e+03  4.68e-05  9.28e-05  3.3e-03  1.0e+00  7.0e-03  1.30e-04
    ##     23   84 -5.691e+03  1.20e-06  1.68e-05  5.1e-04  1.5e+00  1.1e-03  1.96e-05
    ##     24   85 -5.692e+03  6.74e-06  6.45e-06  2.2e-04  1.3e+00  5.7e-04  6.47e-06
    ##     25   86 -5.692e+03  1.23e-07  2.07e-07  1.1e-04  0.0e+00  2.2e-04  2.07e-07
    ##     26   88 -5.692e+03  1.25e-07  1.06e-07  1.4e-04  0.0e+00  3.7e-04  1.06e-07
    ##     27  102 -5.692e+03 -6.65e-14  6.22e-14  1.8e-14  1.7e+04  3.3e-14  1.14e-09
    ## 
    ##  ***** FALSE CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.691511e+03   RELDX        1.756e-14
    ##  FUNC. EVALS     102         GRAD. EVALS      27
    ##  PRELDF       6.222e-14      NPRELDF      1.138e-09
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    1.811773e-06     1.000e+00    -1.060e+04
    ##      2    3.874633e-02     1.000e+00    -1.052e+00
    ##      3    9.514124e-01     1.000e+00    -1.101e+00
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
    ##      2   26 -5.534e+03  3.29e-16  2.33e-14  2.9e-12  7.3e+04  2.9e-13  1.78e-06
    ##      3   27 -5.534e+03  8.22e-16  1.17e-14  1.4e-12  1.5e+05  1.4e-13  1.78e-06
    ##      4   31 -5.534e+03 -3.29e-15  3.80e-17  4.7e-15  4.5e+07  4.7e-16  1.78e-06
    ## 
    ##  ***** FALSE CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.534493e+03   RELDX        4.655e-15
    ##  FUNC. EVALS      31         GRAD. EVALS       4
    ##  PRELDF       3.798e-17      NPRELDF      1.779e-06
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    1.850930e-04     1.000e+00    -4.516e+02
    ##      2    5.000000e-02     1.000e+00     5.299e-03
    ##      3    5.000000e-02     1.000e+00     7.180e-03

    ## Warning in garch(res.arima, order = c(p, q)): singular information

    ## 
    ##  ***** ESTIMATION WITH ANALYTICAL GRADIENT ***** 
    ## 
    ## 
    ##      I     INITIAL X(I)        D(I)
    ## 
    ##      1     1.747613e-04     1.000e+00
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
    ##      7   24 -5.621e+03  1.55e-04  2.38e-03  4.2e-02  2.0e+00  4.5e-02  1.19e+01
    ##      8   25 -5.631e+03  1.62e-03  3.15e-03  2.0e-02  2.0e+00  2.3e-02  5.44e+00
    ##      9   27 -5.636e+03  8.90e-04  9.45e-04  5.1e-02  2.0e+00  6.3e-02  5.03e+00
    ##     10   33 -5.636e+03  5.25e-05  8.28e-05  1.1e-06  9.3e+00  1.3e-06  2.00e-01
    ##     11   34 -5.636e+03  1.18e-06  1.01e-06  9.8e-07  2.0e+00  1.3e-06  1.65e-02
    ##     12   50 -5.636e+03 -4.03e-15  8.41e-15  9.4e-15  1.7e+00  1.2e-14 -2.06e-03
    ## 
    ##  ***** FALSE CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.635842e+03   RELDX        9.436e-15
    ##  FUNC. EVALS      50         GRAD. EVALS      12
    ##  PRELDF       8.410e-15      NPRELDF     -2.059e-03
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    2.512356e-05     1.000e+00    -4.094e+03
    ##      2    8.893964e-02     1.000e+00     6.177e+01
    ##      3    2.672806e-01     1.000e+00     1.333e+02
    ##      4    6.125768e-01     1.000e+00    -4.205e+01
    ## 
    ## 
    ##  ***** ESTIMATION WITH ANALYTICAL GRADIENT ***** 
    ## 
    ## 
    ##      I     INITIAL X(I)        D(I)
    ## 
    ##      1     1.747613e-04     1.000e+00
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
    ##      5   22 -5.656e+03  3.25e-03  3.67e-03  8.6e-02  2.0e+00  6.4e-02  8.91e+00
    ##      6   24 -5.656e+03  9.09e-05  3.04e-04  9.2e-03  2.0e+00  7.4e-03  5.16e+00
    ##      7   25 -5.657e+03  2.38e-04  3.08e-04  9.1e-03  2.0e+00  7.4e-03  1.48e+00
    ##      8   26 -5.658e+03  9.56e-05  1.20e-04  9.0e-03  2.0e+00  7.4e-03  5.37e-01
    ##      9   27 -5.658e+03  2.05e-05  5.85e-05  9.1e-03  2.0e+00  7.4e-03  3.39e-02
    ##     10   28 -5.658e+03  3.55e-05  4.17e-05  7.1e-03  2.0e+00  7.4e-03  7.04e-03
    ##     11   31 -5.659e+03  1.45e-04  1.95e-04  9.0e-02  1.9e+00  1.1e-01  5.54e-03
    ##     12   33 -5.659e+03  4.35e-05  9.88e-05  8.1e-03  1.9e+00  1.1e-02  1.74e-03
    ##     13   37 -5.678e+03  3.26e-03  1.77e-03  1.3e-01  0.0e+00  2.1e-01  1.77e-03
    ##     14   40 -5.678e+03  9.31e-05  7.06e-04  2.2e-02  1.9e+00  3.7e-02  1.42e-02
    ##     15   41 -5.683e+03  7.36e-04  9.72e-04  2.1e-02  1.8e+00  3.7e-02  5.23e-03
    ##     16   43 -5.684e+03  2.99e-04  3.58e-04  2.2e-02  9.9e-01  3.7e-02  1.92e-03
    ##     17   45 -5.687e+03  5.42e-04  6.02e-04  2.3e-02  9.2e-01  3.7e-02  4.44e-03
    ##     18   46 -5.688e+03  1.35e-04  4.78e-04  2.2e-02  7.2e-01  3.7e-02  6.79e-04
    ##     19   47 -5.688e+03  7.51e-05  1.77e-04  1.8e-02  0.0e+00  3.1e-02  1.77e-04
    ##     20   48 -5.689e+03  2.47e-05  2.16e-05  5.9e-03  0.0e+00  1.1e-02  2.16e-05
    ##     21   49 -5.689e+03  2.47e-06  5.23e-06  1.9e-03  0.0e+00  3.3e-03  5.23e-06
    ##     22   50 -5.689e+03  1.05e-06  1.58e-06  8.1e-04  0.0e+00  1.5e-03  1.58e-06
    ##     23   52 -5.689e+03  2.35e-07  5.15e-07  1.7e-04  1.1e+00  2.9e-04  6.97e-07
    ##     24   53 -5.689e+03  2.62e-08  4.24e-08  1.9e-04  0.0e+00  3.5e-04  4.24e-08
    ##     25   55 -5.689e+03  2.69e-09  1.15e-09  2.1e-05  1.6e+00  4.2e-05  1.67e-09
    ##     26   57 -5.689e+03  4.68e-10  7.40e-11  4.2e-06  1.8e+00  8.4e-06  5.68e-10
    ##     27   59 -5.689e+03  6.51e-11  1.01e-11  8.3e-07  2.0e+00  1.7e-06  4.97e-10
    ##     28   62 -5.689e+03  3.61e-12  2.07e-13  1.8e-08  2.0e+00  3.7e-08  4.87e-10
    ##     29   70 -5.689e+03 -1.12e-15  6.94e-19  2.9e-15  6.0e+03  7.3e-15  4.87e-10
    ## 
    ##  ***** FALSE CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.688658e+03   RELDX        2.857e-15
    ##  FUNC. EVALS      70         GRAD. EVALS      29
    ##  PRELDF       6.939e-19      NPRELDF      4.865e-10
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    3.008396e-06     1.000e+00    -2.285e-02
    ##      2    6.659901e-02     1.000e+00     3.129e-01
    ##      3    1.594464e-01     1.000e+00     3.038e-01
    ##      4    7.577371e-01     1.000e+00     3.169e-01

    ## Warning in garch(res.arima, order = c(p, q)): singular information

From the following plot, we can see that the 4th combination (1, 1)
minimises AIC.

``` r
par(mfrow = c(1, 1))
plot(1:nrow(df.order), vec.aic.garch)
```

![](HSBC_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

Therefore, we fit GARCH(1, 1) to the residual of the ARIMA fitting.

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
    ##      6   25 -5.631e+03  3.64e-03  3.44e-03  8.9e-02  2.0e+00  9.8e-02  2.01e+06
    ##      7   27 -5.635e+03  7.31e-04  7.22e-04  1.6e-02  2.0e+00  2.0e-02  1.21e+04
    ##      8   37 -5.636e+03  2.56e-04  6.02e-04  3.9e-06  3.6e+00  4.7e-06  1.38e+03
    ##      9   38 -5.636e+03  1.88e-05  1.59e-05  3.7e-06  2.0e+00  4.7e-06  7.52e+02
    ##     10   39 -5.636e+03  6.61e-07  5.98e-07  3.8e-06  2.0e+00  4.7e-06  7.73e+02
    ##     11   48 -5.650e+03  2.40e-03  3.00e-03  5.8e-02  2.0e+00  7.7e-02  7.71e+02
    ##     12   49 -5.658e+03  1.43e-03  1.59e-03  5.1e-02  7.9e-01  7.7e-02  2.47e-03
    ##     13   51 -5.669e+03  1.96e-03  2.27e-03  8.5e-02  0.0e+00  1.9e-01  2.27e-03
    ##     14   63 -5.676e+03  1.25e-03  1.99e-03  1.5e-06  2.6e+00  2.7e-06  1.11e-02
    ##     15   64 -5.677e+03  7.76e-05  1.52e-04  1.2e-06  2.0e+00  2.7e-06  8.58e-03
    ##     16   65 -5.677e+03  2.05e-05  1.80e-05  1.2e-06  2.0e+00  2.7e-06  1.13e-02
    ##     17   66 -5.677e+03  3.35e-07  3.04e-07  1.2e-06  2.0e+00  2.7e-06  1.17e-02
    ##     18   75 -5.684e+03  1.27e-03  1.14e-03  1.9e-02  1.0e+00  4.5e-02  1.17e-02
    ##     19   77 -5.685e+03  2.62e-04  2.56e-04  3.7e-03  2.0e+00  8.9e-03  5.09e-01
    ##     20   79 -5.688e+03  5.05e-04  5.10e-04  7.3e-03  2.0e+00  1.8e-02  2.02e+00
    ##     21   80 -5.691e+03  5.12e-04  7.67e-04  1.4e-02  1.8e+00  3.6e-02  9.14e-03
    ##     22   82 -5.691e+03  4.68e-05  9.28e-05  3.3e-03  1.0e+00  7.0e-03  1.30e-04
    ##     23   84 -5.691e+03  1.20e-06  1.68e-05  5.1e-04  1.5e+00  1.1e-03  1.96e-05
    ##     24   85 -5.692e+03  6.74e-06  6.45e-06  2.2e-04  1.3e+00  5.7e-04  6.47e-06
    ##     25   86 -5.692e+03  1.23e-07  2.07e-07  1.1e-04  0.0e+00  2.2e-04  2.07e-07
    ##     26   88 -5.692e+03  1.25e-07  1.06e-07  1.4e-04  0.0e+00  3.7e-04  1.06e-07
    ##     27  102 -5.692e+03 -6.65e-14  6.22e-14  1.8e-14  1.7e+04  3.3e-14  1.14e-09
    ## 
    ##  ***** FALSE CONVERGENCE *****
    ## 
    ##  FUNCTION    -5.691511e+03   RELDX        1.756e-14
    ##  FUNC. EVALS     102         GRAD. EVALS      27
    ##  PRELDF       6.222e-14      NPRELDF      1.138e-09
    ## 
    ##      I      FINAL X(I)        D(I)          G(I)
    ## 
    ##      1    1.811773e-06     1.000e+00    -1.060e+04
    ##      2    3.874633e-02     1.000e+00    -1.052e+00
    ##      3    9.514124e-01     1.000e+00    -1.101e+00

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
    ## -5.1494 -0.5448  0.0000  0.4979  7.2936 
    ## 
    ## Coefficient(s):
    ##     Estimate  Std. Error  t value Pr(>|t|)    
    ## a0 1.812e-06   3.275e-07    5.533 3.15e-08 ***
    ## a1 3.875e-02   4.063e-03    9.536  < 2e-16 ***
    ## b1 9.514e-01   5.145e-03  184.937  < 2e-16 ***
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
    ## X-squared = 0.017096, df = 1, p-value = 0.896

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
