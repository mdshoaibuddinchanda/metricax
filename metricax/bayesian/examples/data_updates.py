#!/usr/bin/env python3
"""
Bayesian Data Updates using MetricaX

This example demonstrates how to update beliefs as new data arrives,
showing the power of Bayesian inference for online learning scenarios.
"""

import random
import math

# Import from parent bayesian module
from .. import (
    update_beta_binomial, update_normal_known_variance, update_poisson_gamma,
    beta_mean, beta_var
)


def manufacturing_quality_control():
    """
    Manufacturing quality control scenario:
    Monitor defect rate and update beliefs as new batches are inspected.
    """
    print("🏭 Manufacturing Quality Control with Bayesian Updates")
    print("=" * 55)
    
    # Initial belief: defect rate around 2% (Beta(2, 98))
    alpha, beta = 2, 98
    
    print(f"📊 Initial belief: Beta({alpha}, {beta})")
    print(f"   Expected defect rate: {beta_mean(alpha, beta):.3f}")
    
    # Calculate 95% credible interval approximation
    mean_val = beta_mean(alpha, beta)
    var_val = beta_var(alpha, beta)
    std_val = math.sqrt(var_val)
    print(f"   95% credible interval: [{mean_val - 1.96*std_val:.3f}, {mean_val + 1.96*std_val:.3f}]")
    print()
    
    # Simulate incoming batches
    batches = [
        (100, 3),   # 100 items, 3 defects
        (150, 2),   # 150 items, 2 defects  
        (200, 5),   # 200 items, 5 defects
        (120, 1),   # 120 items, 1 defect
        (180, 4),   # 180 items, 4 defects
    ]
    
    print("📦 Processing batches sequentially:")
    print("-" * 40)
    
    results = []
    for i, (items, defects) in enumerate(batches, 1):
        # Update beliefs with new data
        alpha, beta = update_beta_binomial(alpha, beta, defects, items - defects)
        
        current_rate = beta_mean(alpha, beta)
        current_var = beta_var(alpha, beta)
        
        print(f"Batch {i}: {defects}/{items} defects")
        print(f"   Updated: Beta({alpha}, {beta})")
        print(f"   Defect rate: {current_rate:.4f} ± {math.sqrt(current_var):.4f}")
        
        # Quality alert
        if current_rate > 0.03:  # 3% threshold
            print("   ⚠️  QUALITY ALERT: Defect rate above 3%!")
        else:
            print("   ✅ Quality within acceptable range")
        print()
        
        results.append({
            'batch': i,
            'items': items,
            'defects': defects,
            'alpha': alpha,
            'beta': beta,
            'rate': current_rate
        })
    
    return results


def sensor_calibration():
    """
    Sensor calibration scenario:
    Update beliefs about sensor accuracy as calibration data arrives.
    """
    print("🔬 Sensor Calibration with Normal Updates")
    print("=" * 40)
    
    # Prior belief: sensor reads true value ± 0.5 units
    mu_prior, sigma_prior = 0.0, 0.5
    true_value = 10.0  # Known calibration standard
    
    print(f"📡 Prior belief: N({mu_prior}, {sigma_prior}²)")
    print(f"   True calibration value: {true_value}")
    print()
    
    # Simulate sensor readings (with some bias and noise)
    sensor_bias = 0.2  # Sensor has slight positive bias
    sensor_noise = 0.3  # Sensor noise standard deviation
    
    readings = []
    results = []
    print("📊 Calibration readings:")
    print("-" * 25)
    
    for i in range(5):
        # Simulate a reading
        reading = true_value + sensor_bias + random.gauss(0, sensor_noise)
        readings.append(reading)
        
        # Update beliefs about sensor bias
        mu_new, sigma_new = update_normal_known_variance(
            mu_prior, sigma_prior, 
            [r - true_value for r in readings],  # Bias = reading - true_value
            sigma_likelihood=sensor_noise
        )
        
        print(f"Reading {i+1}: {reading:.2f}")
        print(f"   Estimated bias: {mu_new:.3f} ± {sigma_new:.3f}")
        
        # Update for next iteration
        mu_prior, sigma_prior = mu_new, sigma_new
        
        if abs(mu_new) > 0.1:  # Significant bias threshold
            print("   ⚠️  Sensor needs recalibration!")
        else:
            print("   ✅ Sensor within tolerance")
        print()
        
        results.append({
            'reading': i+1,
            'value': reading,
            'bias_estimate': mu_new,
            'uncertainty': sigma_new
        })
    
    return results


def poisson_process_monitoring():
    """
    Monitor a Poisson process (e.g., website errors per hour)
    and update rate parameter as data arrives.
    """
    print("📈 Poisson Process Monitoring")
    print("=" * 30)
    
    # Prior belief: ~2 errors per hour (Gamma(2, 1))
    alpha, beta = 2.0, 1.0
    
    print(f"⏰ Prior belief: Gamma({alpha}, {beta})")
    print(f"   Expected rate: {alpha/beta:.2f} errors/hour")
    print()
    
    # Simulate hourly error counts
    hours_data = [
        (1, 3),   # 1 hour, 3 errors
        (2, 5),   # 2 hours, 5 errors total
        (1, 2),   # 1 hour, 2 errors
        (3, 8),   # 3 hours, 8 errors total
        (2, 4),   # 2 hours, 4 errors total
    ]
    
    print("🕐 Hourly monitoring:")
    print("-" * 20)
    
    results = []
    for i, (hours, total_errors) in enumerate(hours_data, 1):
        # Update beliefs
        alpha, beta = update_poisson_gamma(alpha, beta, total_errors, hours)
        
        current_rate = alpha / beta
        current_var = alpha / (beta ** 2)
        
        print(f"Period {i}: {total_errors} errors in {hours} hour(s)")
        print(f"   Updated: Gamma({alpha}, {beta})")
        print(f"   Error rate: {current_rate:.2f} ± {math.sqrt(current_var):.2f} per hour")
        
        if current_rate > 3.0:  # Alert threshold
            print("   🚨 HIGH ERROR RATE - Investigation needed!")
        else:
            print("   ✅ Error rate normal")
        print()
        
        results.append({
            'period': i,
            'hours': hours,
            'errors': total_errors,
            'alpha': alpha,
            'beta': beta,
            'rate': current_rate
        })
    
    return results


def demonstrate_conjugate_priors():
    """
    Show the mathematical elegance of conjugate priors.
    """
    print("🎯 Conjugate Priors: Mathematical Elegance")
    print("=" * 45)
    
    print("1. Beta-Binomial Conjugacy:")
    print("   Prior: Beta(α, β)")
    print("   Likelihood: Binomial(n, p)")
    print("   Posterior: Beta(α + successes, β + failures)")
    
    alpha, beta = update_beta_binomial(1, 1, 7, 3)
    print(f"   Example: Beta(1,1) + 7 successes, 3 failures → Beta({alpha}, {beta})")
    print()
    
    print("2. Normal-Normal Conjugacy:")
    print("   Prior: N(μ₀, σ₀²)")
    print("   Likelihood: N(μ, σ²) with known σ²")
    print("   Posterior: N(μₙ, σₙ²) - analytically computed")
    
    mu, sigma = update_normal_known_variance(0, 1, [1, 2, 3], 1)
    print(f"   Example: N(0,1) + data [1,2,3] → N({mu:.2f}, {sigma:.2f}²)")
    print()
    
    print("3. Gamma-Poisson Conjugacy:")
    print("   Prior: Gamma(α, β)")
    print("   Likelihood: Poisson(λ)")
    print("   Posterior: Gamma(α + Σx, β + n)")
    
    alpha, beta = update_poisson_gamma(2, 1, 15, 5)
    print(f"   Example: Gamma(2,1) + 15 events in 5 periods → Gamma({alpha}, {beta})")


def run_example():
    """Main function to run all data update examples"""
    results = {}
    
    results['quality_control'] = manufacturing_quality_control()
    print("\n" + "="*60)
    
    results['sensor_calibration'] = sensor_calibration()
    print("\n" + "="*60)
    
    results['poisson_monitoring'] = poisson_process_monitoring()
    print("\n" + "="*60)
    
    demonstrate_conjugate_priors()
    
    return results


if __name__ == "__main__":
    run_example()
