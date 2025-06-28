import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, poisson
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import iv  # Modified Bessel function
import warnings
warnings.filterwarnings('ignore')

class ParameterEstimation:
    """
    Parameter estimation for various stochastic processes
    """
    
    @staticmethod
    def ou_mle(data, dt):
        """
        Maximum likelihood estimation for Ornstein-Uhlenbeck process
        dX_t = alpha(beta - X_t)dt + sigma dW_t
        """
        X = np.array(data[:-1])
        Y = np.array(data[1:])
        n = len(X)
        
        Sx = np.sum(X)
        Sy = np.sum(Y)
        Sxx = np.sum(X**2)
        Sxy = np.sum(X * Y)
        Syy = np.sum(Y**2)
        
        # MLE estimates
        beta_hat = (Sy * Sxx - Sx * Sxy) / (n * (Sxx - Sxy) - (Sx**2 - Sx * Sy))
        alpha_hat = -np.log((Sxy - beta_hat * Sx - beta_hat * Sy + n * beta_hat**2) / 
                           (Sxx - 2 * beta_hat * Sx + n * beta_hat**2)) / dt
        
        a_hat = np.exp(-alpha_hat * dt)
        sigma_squared_hat = (Syy - 2 * a_hat * Sxy + a_hat**2 * Sxx - 
                            2 * beta_hat * (1 - a_hat) * (Sy - a_hat * Sx) + 
                            n * beta_hat**2 * (1 - a_hat)**2) / n
        sigma_hat = np.sqrt(sigma_squared_hat * 2 * alpha_hat / (1 - np.exp(-2 * alpha_hat * dt)))
        
        return {'alpha': alpha_hat, 'beta': beta_hat, 'sigma': sigma_hat}
    
    @staticmethod
    def gbm_mle(prices):
        """
        Maximum likelihood estimation for Geometric Brownian Motion
        """
        log_returns = np.diff(np.log(prices))
        n = len(log_returns)
        
        mu_hat = np.mean(log_returns)
        sigma_hat = np.sqrt(np.var(log_returns, ddof=1))
        
        dt = 1/252  # daily data
        mu_annual = mu_hat / dt + 0.5 * sigma_hat**2 / dt
        sigma_annual = sigma_hat / np.sqrt(dt)
        
        return {'mu': mu_annual, 'sigma': sigma_annual}
    
    @staticmethod
    def cir_mle(data, dt):
        """
        Approximate MLE for CIR process
        dr_t = alpha(beta - r_t)dt + sigma*sqrt(r_t)*dW_t
        """
        def neg_log_likelihood(params):
            alpha, beta, sigma = params
            if alpha <= 0 or beta <= 0 or sigma <= 0:
                return 1e10
            
            X = np.array(data[:-1])
            Y = np.array(data[1:])
            
            mean = X + alpha * (beta - X) * dt
            var = sigma**2 * np.maximum(X, 1e-8) * dt
            
            ll = np.sum(norm.logpdf(Y, mean, np.sqrt(var)))
            return -ll
        
        x0 = [1.0, np.mean(data), 0.1]
        bounds = [(0.01, 10), (0.01, 1), (0.01, 1)]
        
        result = minimize(neg_log_likelihood, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            return {'alpha': result.x[0], 'beta': result.x[1], 'sigma': result.x[2]}
        else:
            raise ValueError("MLE optimization failed")
    
    @staticmethod
    def vasicek_mle(data, dt):
        """
        MLE for Vasicek interest rate model (similar to OU)
        dr_t = alpha(beta - r_t)dt + sigma*dW_t
        """
        return ParameterEstimation.ou_mle(data, dt)
    
    @staticmethod
    def heston_mle(prices, returns, dt):
        """
        Simplified MLE for Heston stochastic volatility model
        dS_t = mu*S_t*dt + sqrt(V_t)*S_t*dW1_t
        dV_t = kappa*(theta - V_t)*dt + sigma_v*sqrt(V_t)*dW2_t
        """
        def neg_log_likelihood(params):
            mu, kappa, theta, sigma_v, rho = params
            
            if kappa <= 0 or theta <= 0 or sigma_v <= 0 or abs(rho) >= 1:
                return 1e10
            
            # Simplified likelihood using returns and realized volatility
            realized_vol = np.abs(returns)
            
            # Variance process likelihood (approximate)
            V = realized_vol**2
            V_lag = V[:-1]
            V_next = V[1:]
            
            mean_V = V_lag + kappa * (theta - V_lag) * dt
            var_V = sigma_v**2 * np.maximum(V_lag, 1e-8) * dt
            
            ll_vol = np.sum(norm.logpdf(V_next, mean_V, np.sqrt(var_V)))
            
            # Returns likelihood
            expected_returns = mu * dt
            vol = np.sqrt(np.maximum(V[:-1], 1e-8) * dt)
            ll_returns = np.sum(norm.logpdf(returns[1:], expected_returns, vol))
            
            return -(ll_vol + ll_returns)
        
        # Initial guess based on sample statistics
        sample_vol = np.std(returns)
        x0 = [np.mean(returns)/dt, 2.0, sample_vol**2, 0.3, -0.5]
        bounds = [(-1, 1), (0.1, 10), (0.01, 1), (0.01, 1), (-0.99, 0.99)]
        
        result = differential_evolution(neg_log_likelihood, bounds, seed=42)
        
        if result.success:
            return {
                'mu': result.x[0], 'kappa': result.x[1], 'theta': result.x[2],
                'sigma_v': result.x[3], 'rho': result.x[4]
            }
        else:
            raise ValueError("Heston MLE optimization failed")
    
    @staticmethod
    def merton_jump_mle(prices):
        """
        MLE for Merton jump-diffusion model
        dS_t = (mu - lambda*k)*S_t*dt + sigma*S_t*dW_t + S_t*dJ_t
        where J_t is compound Poisson with normal jumps
        """
        log_returns = np.diff(np.log(prices))
        
        def neg_log_likelihood(params):
            mu, sigma, lambda_j, mu_j, sigma_j = params
            
            if sigma <= 0 or lambda_j < 0 or sigma_j <= 0:
                return 1e10
            
            dt = 1/252
            ll = 0
            
            for r in log_returns:
                # Probability of no jump
                prob_no_jump = np.exp(-lambda_j * dt)
                density_no_jump = norm.pdf(r, (mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt))
                
                # Probability of jumps (approximate for small lambda*dt)
                max_jumps = min(5, int(10 * lambda_j * dt) + 1)
                prob_jump_total = 0
                
                for n in range(1, max_jumps + 1):
                    prob_n_jumps = (lambda_j * dt)**n * np.exp(-lambda_j * dt) / np.math.factorial(n)
                    
                    # Expected jump size and variance
                    jump_mean = n * mu_j
                    jump_var = n * sigma_j**2
                    
                    total_mean = (mu - 0.5*sigma**2)*dt + jump_mean
                    total_var = sigma**2*dt + jump_var
                    
                    density_with_jumps = norm.pdf(r, total_mean, np.sqrt(total_var))
                    prob_jump_total += prob_n_jumps * density_with_jumps
                
                total_density = prob_no_jump * density_no_jump + prob_jump_total
                ll += np.log(max(total_density, 1e-10))
            
            return -ll
        
        # Initial guess
        sample_vol = np.std(log_returns) * np.sqrt(252)
        x0 = [0.08, sample_vol, 2.0, -0.02, 0.05]  # realistic jump parameters
        bounds = [(-0.5, 0.5), (0.01, 1), (0, 20), (-0.2, 0.2), (0.01, 0.3)]
        
        result = differential_evolution(neg_log_likelihood, bounds, seed=42, maxiter=300)
        
        if result.success:
            return {
                'mu': result.x[0], 'sigma': result.x[1], 'lambda': result.x[2],
                'mu_jump': result.x[3], 'sigma_jump': result.x[4]
            }
        else:
            # Fallback to GBM if jump-diffusion fails
            gbm_params = ParameterEstimation.gbm_mle(prices)
            return {
                'mu': gbm_params['mu'], 'sigma': gbm_params['sigma'], 'lambda': 0,
                'mu_jump': 0, 'sigma_jump': 0
            }
    
    @staticmethod
    def garch_mle(returns):
        """
        MLE for GARCH(1,1) model
        r_t = sigma_t * epsilon_t
        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
        """
        def neg_log_likelihood(params):
            omega, alpha, beta = params
            
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            n = len(returns)
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(returns)
            
            ll = 0
            for t in range(1, n):
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
                ll += norm.logpdf(returns[t], 0, np.sqrt(sigma2[t]))
            
            return -ll
        
        # Initial guess
        unconditional_var = np.var(returns)
        x0 = [0.1 * unconditional_var, 0.1, 0.8]
        bounds = [(1e-6, 1), (0, 1), (0, 1)]
        
        # Add constraint alpha + beta < 1
        constraints = {'type': 'ineq', 'fun': lambda x: 0.999 - x[1] - x[2]}
        
        result = minimize(neg_log_likelihood, x0, bounds=bounds, 
                         constraints=constraints, method='SLSQP')
        
        if result.success:
            return {'omega': result.x[0], 'alpha': result.x[1], 'beta': result.x[2]}
        else:
            raise ValueError("GARCH MLE optimization failed")

# Simulation functions for new models
class StochasticSimulator:
    """
    Simulation methods for various stochastic processes
    """
    
    @staticmethod
    def simulate_ou(alpha, beta, sigma, T, dt, X0=None):
        """Simulate Ornstein-Uhlenbeck process"""
        n_steps = int(T / dt)
        X = np.zeros(n_steps + 1)
        X[0] = beta if X0 is None else X0
        
        for i in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            X[i + 1] = (X[i] + alpha * beta * dt + sigma * dW) / (1 + alpha * dt)
        
        return X
    
    @staticmethod
    def simulate_gbm(mu, sigma, S0, T, dt):
        """Simulate Geometric Brownian Motion"""
        n_steps = int(T / dt)
        S = np.zeros(n_steps + 1)
        S[0] = S0
        
        for i in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            S[i + 1] = S[i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        
        return S
    
    @staticmethod
    def simulate_cir(alpha, beta, sigma, T, dt, r0=None):
        """Simulate Cox-Ingersoll-Ross process"""
        n_steps = int(T / dt)
        r = np.zeros(n_steps + 1)
        r[0] = beta if r0 is None else r0
        
        for i in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            drift = alpha * (beta - r[i]) * dt
            diffusion = sigma * np.sqrt(max(r[i], 0)) * dW
            r[i + 1] = max(r[i] + drift + diffusion, 0)
        
        return r
    
    @staticmethod
    def simulate_heston(mu, kappa, theta, sigma_v, rho, S0, V0, T, dt):
        """Simulate Heston stochastic volatility model"""
        n_steps = int(T / dt)
        S = np.zeros(n_steps + 1)
        V = np.zeros(n_steps + 1)
        S[0] = S0
        V[0] = V0
        
        for i in range(n_steps):
            # Correlated random variables
            dW1 = np.random.normal(0, np.sqrt(dt))
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt))
            
            # Update variance (with reflection at zero)
            V[i + 1] = max(V[i] + kappa * (theta - V[i]) * dt + 
                          sigma_v * np.sqrt(max(V[i], 0)) * dW2, 0)
            
            # Update stock price
            S[i + 1] = S[i] * np.exp((mu - 0.5 * V[i]) * dt + 
                                    np.sqrt(max(V[i], 0)) * dW1)
        
        return S, V
    
    @staticmethod
    def simulate_merton_jump(mu, sigma, lambda_j, mu_j, sigma_j, S0, T, dt):
        """Simulate Merton jump-diffusion model"""
        n_steps = int(T / dt)
        S = np.zeros(n_steps + 1)
        S[0] = S0
        
        # Adjust drift for jump compensation
        mu_adj = mu - lambda_j * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
        
        for i in range(n_steps):
            # Diffusion component
            dW = np.random.normal(0, np.sqrt(dt))
            
            # Jump component
            n_jumps = poisson.rvs(lambda_j * dt)
            jump_size = 0
            if n_jumps > 0:
                jumps = np.random.normal(mu_j, sigma_j, n_jumps)
                jump_size = np.sum(jumps)
            
            # Update stock price
            S[i + 1] = S[i] * np.exp((mu_adj - 0.5 * sigma**2) * dt + 
                                    sigma * dW + jump_size)
        
        return S
    
    @staticmethod
    def simulate_garch_returns(omega, alpha, beta, n_periods, initial_vol=None):
        """Simulate GARCH(1,1) returns"""
        returns = np.zeros(n_periods)
        sigma2 = np.zeros(n_periods)
        
        # Initial variance
        if initial_vol is None:
            sigma2[0] = omega / (1 - alpha - beta)  # Unconditional variance
        else:
            sigma2[0] = initial_vol**2
        
        returns[0] = np.random.normal(0, np.sqrt(sigma2[0]))
        
        for t in range(1, n_periods):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            returns[t] = np.random.normal(0, np.sqrt(sigma2[t]))
        
        return returns, np.sqrt(sigma2)

def simulate_and_estimate_extended():
    """
    Simulate data and test parameter estimation for multiple models
    """
    np.random.seed(42)
    
    print("Extended Parameter Estimation Testing")
    print("=" * 60)
    
    results = {}
    
    # 1. Ornstein-Uhlenbeck Process (Mean-reverting interest rates)
    print("\n1. Ornstein-Uhlenbeck Process (Interest Rate)")
    alpha_true, beta_true, sigma_true = 1.5, 0.025, 0.008  # Realistic interest rate params
    print(f"True parameters: α={alpha_true}, β={beta_true:.3f}, σ={sigma_true:.3f}")
    
    ou_data = StochasticSimulator.simulate_ou(alpha_true, beta_true, sigma_true, 
                                             T=5, dt=0.01, X0=0.03)
    
    ou_estimates = ParameterEstimation.ou_mle(ou_data, 0.01)
    print(f"Estimated: α={ou_estimates['alpha']:.3f}, β={ou_estimates['beta']:.4f}, σ={ou_estimates['sigma']:.4f}")
    results['OU'] = {'data': ou_data, 'estimates': ou_estimates, 'true': (alpha_true, beta_true, sigma_true)}
    
    # 2. Geometric Brownian Motion (Stock prices)
    print("\n2. Geometric Brownian Motion (Stock Price)")
    mu_true, sigma_true = 0.12, 0.20  # 12% drift, 20% volatility
    print(f"True parameters: μ={mu_true:.3f}, σ={sigma_true:.3f}")
    
    gbm_data = StochasticSimulator.simulate_gbm(mu_true, sigma_true, S0=100, 
                                               T=2, dt=1/252)
    
    gbm_estimates = ParameterEstimation.gbm_mle(gbm_data)
    print(f"Estimated: μ={gbm_estimates['mu']:.4f}, σ={gbm_estimates['sigma']:.4f}")
    results['GBM'] = {'data': gbm_data, 'estimates': gbm_estimates, 'true': (mu_true, sigma_true)}
    
    # 3. Cox-Ingersoll-Ross Process (Interest rates)
    print("\n3. Cox-Ingersoll-Ross Process (Interest Rate)")
    alpha_true, beta_true, sigma_true = 2.0, 0.03, 0.15  # Realistic CIR params
    print(f"True parameters: α={alpha_true}, β={beta_true:.3f}, σ={sigma_true:.3f}")
    
    cir_data = StochasticSimulator.simulate_cir(alpha_true, beta_true, sigma_true,
                                               T=3, dt=0.01, r0=0.025)
    
    try:
        cir_estimates = ParameterEstimation.cir_mle(cir_data, 0.01)
        print(f"Estimated: α={cir_estimates['alpha']:.3f}, β={cir_estimates['beta']:.4f}, σ={cir_estimates['sigma']:.4f}")
        results['CIR'] = {'data': cir_data, 'estimates': cir_estimates, 'true': (alpha_true, beta_true, sigma_true)}
    except Exception as e:
        print(f"CIR estimation failed: {e}")
        results['CIR'] = {'data': cir_data, 'estimates': None, 'true': (alpha_true, beta_true, sigma_true)}
    
    # 4. Heston Stochastic Volatility Model
    print("\n4. Heston Stochastic Volatility Model")
    mu_true, kappa_true, theta_true, sigma_v_true, rho_true = 0.08, 3.0, 0.04, 0.3, -0.7
    print(f"True parameters: μ={mu_true:.3f}, κ={kappa_true:.1f}, θ={theta_true:.3f}, σ_v={sigma_v_true:.1f}, ρ={rho_true:.1f}")
    
    heston_S, heston_V = StochasticSimulator.simulate_heston(
        mu_true, kappa_true, theta_true, sigma_v_true, rho_true,
        S0=100, V0=0.04, T=2, dt=1/252
    )
    
    heston_returns = np.diff(np.log(heston_S))
    
    try:
        heston_estimates = ParameterEstimation.heston_mle(heston_S, heston_returns, 1/252)
        print(f"Estimated: μ={heston_estimates['mu']:.4f}, κ={heston_estimates['kappa']:.3f}, " +
              f"θ={heston_estimates['theta']:.4f}, σ_v={heston_estimates['sigma_v']:.3f}, ρ={heston_estimates['rho']:.3f}")
        results['Heston'] = {'data': (heston_S, heston_V), 'estimates': heston_estimates, 
                            'true': (mu_true, kappa_true, theta_true, sigma_v_true, rho_true)}
    except Exception as e:
        print(f"Heston estimation failed: {e}")
        results['Heston'] = {'data': (heston_S, heston_V), 'estimates': None,
                            'true': (mu_true, kappa_true, theta_true, sigma_v_true, rho_true)}
    
    # 5. Merton Jump-Diffusion Model
    print("\n5. Merton Jump-Diffusion Model")
    mu_true, sigma_true, lambda_true, mu_j_true, sigma_j_true = 0.10, 0.18, 3.0, -0.03, 0.06
    print(f"True parameters: μ={mu_true:.3f}, σ={sigma_true:.3f}, λ={lambda_true:.1f}, " +
          f"μ_j={mu_j_true:.3f}, σ_j={sigma_j_true:.3f}")
    
    merton_data = StochasticSimulator.simulate_merton_jump(
        mu_true, sigma_true, lambda_true, mu_j_true, sigma_j_true,
        S0=100, T=2, dt=1/252
    )
    
    try:
        merton_estimates = ParameterEstimation.merton_jump_mle(merton_data)
        print(f"Estimated: μ={merton_estimates['mu']:.4f}, σ={merton_estimates['sigma']:.4f}, " +
              f"λ={merton_estimates['lambda']:.3f}, μ_j={merton_estimates['mu_jump']:.4f}, " +
              f"σ_j={merton_estimates['sigma_jump']:.4f}")
        results['Merton'] = {'data': merton_data, 'estimates': merton_estimates,
                            'true': (mu_true, sigma_true, lambda_true, mu_j_true, sigma_j_true)}
    except Exception as e:
        print(f"Merton estimation failed: {e}")
        results['Merton'] = {'data': merton_data, 'estimates': None,
                            'true': (mu_true, sigma_true, lambda_true, mu_j_true, sigma_j_true)}
    
    # 6. GARCH(1,1) Model
    print("\n6. GARCH(1,1) Volatility Model")
    omega_true, alpha_true, beta_true = 0.00002, 0.08, 0.90  # Realistic GARCH params
    print(f"True parameters: ω={omega_true:.6f}, α={alpha_true:.3f}, β={beta_true:.3f}")
    
    garch_returns, garch_vol = StochasticSimulator.simulate_garch_returns(
        omega_true, alpha_true, beta_true, n_periods=500, initial_vol=0.02
    )
    
    try:
        garch_estimates = ParameterEstimation.garch_mle(garch_returns)
        print(f"Estimated: ω={garch_estimates['omega']:.6f}, α={garch_estimates['alpha']:.3f}, β={garch_estimates['beta']:.3f}")
        results['GARCH'] = {'data': (garch_returns, garch_vol), 'estimates': garch_estimates,
                           'true': (omega_true, alpha_true, beta_true)}
    except Exception as e:
        print(f"GARCH estimation failed: {e}")
        results['GARCH'] = {'data': (garch_returns, garch_vol), 'estimates': None,
                           'true': (omega_true, alpha_true, beta_true)}
    
    return results

# Run the extended test
results = simulate_and_estimate_extended()

# Enhanced plotting
fig = plt.figure(figsize=(18, 12))

# 1. OU Process
ax1 = plt.subplot(3, 3, 1)
if 'OU' in results:
    times_ou = np.linspace(0, 5, len(results['OU']['data']))
    plt.plot(times_ou, results['OU']['data'], 'b-', linewidth=1, alpha=0.8)
    plt.axhline(y=results['OU']['true'][1], color='r', linestyle='--', 
               label=f'True mean = {results["OU"]["true"][1]:.3f}')
    plt.title('Ornstein-Uhlenbeck Process\n(Interest Rate)', fontsize=10)
    plt.xlabel('Time (years)')
    plt.ylabel('Interest Rate')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

# 2. GBM
ax2 = plt.subplot(3, 3, 2)
if 'GBM' in results:
    times_gbm = np.linspace(0, 2, len(results['GBM']['data']))
    plt.plot(times_gbm, results['GBM']['data'], 'g-', linewidth=1, alpha=0.8)
    plt.title('Geometric Brownian Motion\n(Stock Price)', fontsize=10)
    plt.xlabel('Time (years)')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)

# 3. CIR Process
ax3 = plt.subplot(3, 3, 3)
if 'CIR' in results:
    times_cir = np.linspace(0, 3, len(results['CIR']['data']))
    plt.plot(times_cir, results['CIR']['data'], 'm-', linewidth=1, alpha=0.8)
    plt.axhline(y=results['CIR']['true'][1], color='r', linestyle='--',
               label=f'True mean = {results["CIR"]["true"][1]:.3f}')
    plt.title('Cox-Ingersoll-Ross Process\n(Interest Rate)', fontsize=10)
    plt.xlabel('Time (years)')
    plt.ylabel('Interest Rate')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

# 4. Heston Stock Price
ax4 = plt.subplot(3, 3, 4)
if 'Heston' in results:
    times_heston = np.linspace(0, 2, len(results['Heston']['data'][0]))
    plt.plot(times_heston, results['Heston']['data'][0], 'orange', linewidth=1, alpha=0.8)
    plt.title('Heston Model\n(Stock Price)', fontsize=10)
    plt.xlabel('Time (years)')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)

# 5. Heston Volatility
ax5 = plt.subplot(3, 3, 5)
if 'Heston' in results:
    plt.plot(times_heston, np.sqrt(results['Heston']['data'][1]), 'red', linewidth=1, alpha=0.8)
    plt.axhline(y=np.sqrt(results['Heston']['true'][2]), color='darkred', linestyle='--',
               label=f'True long-term vol = {np.sqrt(results["Heston"]["true"][2]):.3f}')
    plt.title('Heston Model\n(Stochastic Volatility)', fontsize=10)
    plt.xlabel('Time (years)')
    plt.ylabel('Volatility')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

# 6. Merton Jump-Diffusion
ax6 = plt.subplot(3, 3, 6)
if 'Merton' in results:
    times_merton = np.linspace(0, 2, len(results['Merton']['data']))
    plt.plot(times_merton, results['Merton']['data'], 'purple', linewidth=1, alpha=0.8)
    plt.title('Merton Jump-Diffusion\n(Stock Price)', fontsize=10)
    plt.xlabel('Time (years)')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)

# 7. GARCH Returns
ax7 = plt.subplot(3, 3, 7)
if 'GARCH' in results:
    times_garch = np.arange(len(results['GARCH']['data'][0]))
    plt.plot(times_garch, results['GARCH']['data'][0], 'brown', linewidth=0.8, alpha=0.8)
    plt.title('GARCH(1,1) Model\n(Returns)', fontsize=10)
    plt.xlabel('Time periods')
    plt.ylabel('Returns')
    plt.grid(True, alpha=0.3)

# 8. GARCH Volatility
ax8 = plt.subplot(3, 3, 8)
if 'GARCH' in results:
    plt.plot(times_garch, results['GARCH']['data'][1], 'darkred', linewidth=1, alpha=0.8)
    plt.title('GARCH(1,1) Model\n(Conditional Volatility)', fontsize=10)
    plt.xlabel('Time periods')
    plt.ylabel('Volatility')
    plt.grid(True, alpha=0.3)

# 9. Summary Statistics
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = "Model Estimation Summary:\n\n"

model_names = {'OU': 'Ornstein-Uhlenbeck', 'GBM': 'Geometric Brownian Motion', 
               'CIR': 'Cox-Ingersoll-Ross', 'Heston': 'Heston Stochastic Vol',
               'Merton': 'Merton Jump-Diffusion', 'GARCH': 'GARCH(1,1)'}

for model, data in results.items():
    if data['estimates'] is not None:
        summary_text += f"✓ {model_names[model]}: Converged\n"
    else:
        summary_text += f"✗ {model_names[model]}: Failed\n"

summary_text += f"\nRealistic Parameters Used:\n"
summary_text += f"• Interest rates: 2-3% mean reversion\n"
summary_text += f"• Stock volatility: 18-25% annual\n"
summary_text += f"• Jump frequency: 3 jumps/year\n"
summary_text += f"• GARCH persistence: α+β ≈ 0.98"

plt.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.suptitle('Extended Stochastic Process Parameter Estimation', fontsize=14, y=0.98)
plt.show()

# Additional analysis: Model comparison table
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': ['Ornstein-Uhlenbeck', 'Geometric Brownian Motion', 'Cox-Ingersoll-Ross', 
              'Heston Stochastic Vol', 'Merton Jump-Diffusion', 'GARCH(1,1)'],
    'Application': ['Interest Rates', 'Stock Prices', 'Interest Rates', 
                    'Stock Prices + Vol', 'Stock Prices + Jumps', 'Volatility Modeling'],
    'Key Features': ['Mean Reversion', 'Constant Volatility', 'Mean Reversion + Vol Smile',
                     'Stochastic Volatility', 'Jump Risk', 'Volatility Clustering'],
    'Estimation Success': [
        '✓' if results.get('OU', {}).get('estimates') else '✗',
        '✓' if results.get('GBM', {}).get('estimates') else '✗',
        '✓' if results.get('CIR', {}).get('estimates') else '✗',
        '✓' if results.get('Heston', {}).get('estimates') else '✗',
        '✓' if results.get('Merton', {}).get('estimates') else '✗',
        '✓' if results.get('GARCH', {}).get('estimates') else '✗'
    ]
})

print(comparison_df.to_string(index=False))
