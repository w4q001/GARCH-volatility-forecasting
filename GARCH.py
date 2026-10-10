import numpy as np
import pandas as pd
import scipy.stats as stats
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST_SIZE = 500
NAIVE_WINDOW = 20
REFIT_EVERY = 20


def qlike(realized, forecast):
    return np.mean(np.log(forecast) + realized / forecast)



print("STEP 1-2: Loading data and fitting GARCH(1,1) (Normal errors)")
print("=" * 70)

df = pd.read_csv("spy_prices.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M:%S")
df = df.sort_values("Date").reset_index(drop=True)
df = df[["Date", "Close"]].dropna()

df["log_ret"] = 100 * np.log(df["Close"] / df["Close"].shift(1))
df = df.dropna().reset_index(drop=True)
print(f"Loaded {len(df)} daily log returns, {df['Date'].min().date()} to {df['Date'].max().date()}")

model_normal = arch_model(df["log_ret"], mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
res_normal = model_normal.fit(disp="off")
print(res_normal.summary())

omega_n, alpha_n, beta_n = res_normal.params["omega"], res_normal.params["alpha[1]"], res_normal.params["beta[1]"]
persistence_n = alpha_n + beta_n
print(f"\nPersistence (alpha+beta) = {persistence_n:.4f} -- "
      f"{'stationary' if persistence_n < 1 else 'WARNING: non-stationary'}")
print(f"Long-run annualized vol = {np.sqrt((omega_n / (1 - persistence_n)) * 252):.2f}%")
print(f"Half-life = {np.log(0.5) / np.log(persistence_n):.1f} trading days")


print("\n\nSTEP 3: Diagnostics")
print("=" * 70)

std_resid = res_normal.std_resid.dropna()
lb = acorr_ljungbox(std_resid ** 2, lags=[10, 20], return_df=True)
print("Ljung-Box test on squared standardized residuals:")
print(lb)

skew, kurt = stats.skew(std_resid), stats.kurtosis(std_resid)
jb_stat, jb_p = stats.jarque_bera(std_resid)
print(f"\nSkewness: {skew:.3f}, Excess kurtosis: {kurt:.3f}")
print(f"Jarque-Bera: stat={jb_stat:.2f}, p={jb_p:.2e} (low p rejects Normality)")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
stats.probplot(std_resid, dist="norm", plot=axes[0])
axes[0].set_title("QQ-plot of standardized residuals vs Normal")
axes[0].grid(alpha=0.3)
axes[1].hist(std_resid, bins=60, density=True, alpha=0.6, label="Standardized residuals")
x = np.linspace(std_resid.min(), std_resid.max(), 300)
axes[1].plot(x, stats.norm.pdf(x), 'r--', label="Normal pdf")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("diagnostics_plot.png", dpi=150)
plt.close()
print("Saved diagnostics_plot.png")

model_t = arch_model(df["log_ret"], mean="Constant", vol="GARCH", p=1, q=1, dist="t")
res_t = model_t.fit(disp="off")
print(f"\nNormal AIC = {res_normal.aic:.2f}  |  Student-t AIC = {res_t.aic:.2f}")
print(f"Student-t nu (degrees of freedom) = {res_t.params['nu']:.2f}")


print("\n\nSTEP 4: Out-of-sample forecast evaluation")
print("=" * 70)

returns = df["log_ret"].values
n = len(returns)
train_end = n - TEST_SIZE

garch_forecast_var = np.full(TEST_SIZE, np.nan)
naive_forecast_var = np.full(TEST_SIZE, np.nan)
realized_var = np.full(TEST_SIZE, np.nan)
omega = alpha = beta = mu = sigma2_prev = eps_prev = None

for i in range(TEST_SIZE):
    t = train_end + i
    if i % REFIT_EVERY == 0:
        train_returns = returns[:t]
        m = arch_model(train_returns, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        r = m.fit(disp="off")
        omega, alpha, beta, mu = r.params["omega"], r.params["alpha[1]"], r.params["beta[1]"], r.params["mu"]
        sigma2_prev = r.conditional_volatility[-1] ** 2
        eps_prev = train_returns[-1] - mu

    sigma2_t = omega + alpha * eps_prev ** 2 + beta * sigma2_prev
    garch_forecast_var[i] = sigma2_t
    naive_forecast_var[i] = np.var(returns[t - NAIVE_WINDOW:t], ddof=1)
    realized_var[i] = (returns[t] - returns[:t].mean()) ** 2
    eps_prev, sigma2_prev = returns[t] - mu, sigma2_t

rmse_garch = np.sqrt(np.mean((realized_var - garch_forecast_var) ** 2))
rmse_naive = np.sqrt(np.mean((realized_var - naive_forecast_var) ** 2))
qlike_garch, qlike_naive = qlike(realized_var, garch_forecast_var), qlike(realized_var, naive_forecast_var)

print(f"GARCH(1,1)-t : RMSE={rmse_garch:.4f}, QLIKE={qlike_garch:.4f}")
print(f"Naive (20d)  : RMSE={rmse_naive:.4f}, QLIKE={qlike_naive:.4f}")
print(f"RMSE improvement:  {100 * (rmse_naive - rmse_garch) / rmse_naive:+.1f}%")
print(f"QLIKE improvement: {100 * (qlike_naive - qlike_garch) / abs(qlike_naive):+.1f}%")

out_dates = df["Date"].iloc[train_end:train_end + TEST_SIZE].values
ann = lambda v: np.sqrt(v * 252)
plt.figure(figsize=(12, 5))
plt.plot(out_dates, ann(realized_var), color="gray", alpha=0.5, label="Realized vol (actual)")
plt.plot(out_dates, ann(garch_forecast_var), color="crimson", label="GARCH(1,1)-t forecast")
plt.plot(out_dates, ann(naive_forecast_var), color="steelblue", linestyle="--", label="Naive 20d rolling forecast")
plt.title("Out-of-sample 1-day-ahead volatility forecast: GARCH vs naive benchmark")
plt.ylabel("Annualized volatility (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_comparison_plot.png", dpi=150)
plt.close()
print("Saved forecast_comparison_plot.png")


print("\n\nSTEP 5: Sensitivity analysis")
print("=" * 70)


def run_one(refit_every, naive_window):
    g_var = np.full(TEST_SIZE, np.nan)
    n_var = np.full(TEST_SIZE, np.nan)
    r_var = np.full(TEST_SIZE, np.nan)
    om = al = be = m = s2p = ep = None
    for i in range(TEST_SIZE):
        t = train_end + i
        if i % refit_every == 0:
            tr = returns[:t]
            mod = arch_model(tr, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
            r = mod.fit(disp="off")
            om, al, be, m = r.params["omega"], r.params["alpha[1]"], r.params["beta[1]"], r.params["mu"]
            s2p = r.conditional_volatility[-1] ** 2
            ep = tr[-1] - m
        s2t = om + al * ep ** 2 + be * s2p
        g_var[i] = s2t
        n_var[i] = np.var(returns[t - naive_window:t], ddof=1)
        r_var[i] = (returns[t] - returns[:t].mean()) ** 2
        ep, s2p = returns[t] - m, s2t
    rg = np.sqrt(np.mean((r_var - g_var) ** 2))
    rn = np.sqrt(np.mean((r_var - n_var) ** 2))
    qg, qn = qlike(r_var, g_var), qlike(r_var, n_var)
    return rg, rn, qg, qn


rows = []
for refit_every in [10, 20, 40]:
    for naive_window in [10, 20, 60]:
        rg, rn, qg, qn = run_one(refit_every, naive_window)
        rmse_imp = 100 * (rn - rg) / rn
        qlike_imp = 100 * (qn - qg) / abs(qn)
        rows.append([refit_every, naive_window, rmse_imp, qlike_imp])
        print(f"refit={refit_every:>3}d, naive_window={naive_window:>3}d -> "
              f"RMSE {rmse_imp:+.1f}%, QLIKE {qlike_imp:+.1f}%")

sens = pd.DataFrame(rows, columns=["refit_every", "naive_window", "rmse_improve_%", "qlike_improve_%"])
n_pos = (sens["rmse_improve_%"] > 0).sum()
print(f"\nGARCH beat naive in {n_pos}/{len(sens)} configurations on RMSE")


print("\n\nSTEP 6: Final plots")
print("=" * 70)

cond_vol_annual = res_t.conditional_volatility * np.sqrt(252)
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
axes[0].plot(df["Date"], df["log_ret"], color="steelblue", lw=0.6)
axes[0].set_ylabel("Daily log return (%)")
axes[0].set_title("SPY daily returns -- note how big moves cluster together in time")
axes[0].grid(alpha=0.3)
axes[1].plot(df["Date"], cond_vol_annual, color="crimson", lw=0.9)
axes[1].set_ylabel("Annualized volatility (%)")
axes[1].set_title("GARCH(1,1)-t fitted conditional volatility")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("volatility_clustering_plot.png", dpi=150)
plt.close()
print("Saved volatility_clustering_plot.png")

omega_t, alpha_t, beta_t = res_t.params["omega"], res_t.params["alpha[1]"], res_t.params["beta[1]"]
persistence_t = alpha_t + beta_t
long_run_var_t = omega_t / (1 - persistence_t)
sigma2_1 = res_t.conditional_volatility.iloc[-1] ** 2
horizons = np.arange(1, 121)
forecast_var = long_run_var_t + (persistence_t ** (horizons - 1)) * (sigma2_1 - long_run_var_t)
long_run_vol = np.sqrt(long_run_var_t * 252)
half_life_t = np.log(0.5) / np.log(persistence_t)

plt.figure(figsize=(10, 5))
plt.plot(horizons, np.sqrt(forecast_var * 252), color="darkorange", lw=2, label="h-day-ahead GARCH forecast")
plt.axhline(long_run_vol, color="gray", linestyle="--", label=f"Long-run volatility ({long_run_vol:.1f}%)")
plt.axvline(half_life_t, color="gray", linestyle=":", alpha=0.7)
plt.xlabel("Forecast horizon (trading days ahead)")
plt.ylabel("Forecasted annualized volatility (%)")
plt.title(f"Volatility forecast decays toward long-run average (persistence = {persistence_t:.3f})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("forecast_decay_plot.png", dpi=150)
plt.close()
print("Saved forecast_decay_plot.png")

print("\n\nAll done -- 4 plots saved in the current folder.")


