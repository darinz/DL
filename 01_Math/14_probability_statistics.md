# Probability & Statistics for Deep Learning

> **Essential probability and statistics concepts that form the foundation of machine learning algorithms, uncertainty quantification, and data analysis.**

---

## Table of Contents

1. [Probability Fundamentals](#probability-fundamentals)
2. [Random Variables and Distributions](#random-variables-and-distributions)
3. [Statistical Inference](#statistical-inference)
4. [Hypothesis Testing](#hypothesis-testing)
5. [Bayesian Statistics](#bayesian-statistics)
6. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Probability Fundamentals

### What is Probability?

Probability is a measure of the likelihood that an event will occur. It provides a mathematical framework for dealing with uncertainty and randomness. In deep learning, probability is used to model data, uncertainty, and predictions.

### Basic Concepts

#### Sample Space
The sample space $\Omega$ is the set of all possible outcomes of an experiment.

#### Event
An event $A$ is a subset of the sample space.

#### Probability Axioms
For any event $A$:
1. $P(A) \geq 0$ (non-negativity)
2. $P(\Omega) = 1$ (normalization)
3. For mutually exclusive events $A_1, A_2, \ldots$: $P(\cup_i A_i) = \sum_i P(A_i)$ (additivity)

**Step-by-step:**
- List all possible outcomes (sample space).
- Define events as subsets of outcomes.
- Assign probabilities according to the axioms above.

### Conditional Probability

The conditional probability of event $A$ given event $B$ is:

```math
P(A|B) = \frac{P(A \cap B)}{P(B)}
```

This represents the probability of $A$ occurring given that $B$ has already occurred. Conditional probability is crucial for understanding dependencies in data and for Bayesian inference.

**Step-by-step:**
- Find the probability that both $A$ and $B$ occur.
- Divide by the probability that $B$ occurs.

### Bayes' Theorem

Bayes' theorem relates conditional probabilities:

```math
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
```

Where $P(B) = \sum_i P(B|A_i)P(A_i)$ (law of total probability).

- **Intuition:** Bayes' theorem allows us to update our beliefs about $A$ after observing $B$.
- **Deep learning connection:** Used in probabilistic models, Bayesian neural networks, and uncertainty estimation.

**Step-by-step:**
- Compute the likelihood $P(B|A)$ and the prior $P(A)$.
- Compute the evidence $P(B)$ (sum over all possible $A_i$).
- Divide to get the posterior $P(A|B)$.

### Independence

Events $A$ and $B$ are independent if:

```math
P(A \cap B) = P(A)P(B)
```

- **Intuition:** The occurrence of $A$ does not affect the probability of $B$.

**Step-by-step:**
- Check if $P(A \cap B) = P(A)P(B)$. If so, $A$ and $B$ are independent.

### Python Implementation: Basic Probability

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Simulate coin flips
def simulate_coin_flips(n_flips):
    """Simulate n coin flips"""
    return np.random.choice(['H', 'T'], size=n_flips)

# Simulate dice rolls
def simulate_dice_rolls(n_rolls):
    """Simulate n dice rolls"""
    return np.random.randint(1, 7, size=n_rolls)

# Example: Coin flip simulation
np.random.seed(42)
n_flips = 1000
flips = simulate_coin_flips(n_flips)

# Count outcomes
flip_counts = Counter(flips)
heads_prob = flip_counts['H'] / n_flips
tails_prob = flip_counts['T'] / n_flips

print(f"Coin flip results (n={n_flips}):")
print(f"Heads: {flip_counts['H']} (P(H) = {heads_prob:.3f})")
print(f"Tails: {flip_counts['T']} (P(T) = {tails_prob:.3f})")

# Example: Dice roll simulation
n_rolls = 1000
rolls = simulate_dice_rolls(n_rolls)

# Count outcomes
roll_counts = Counter(rolls)
roll_probs = {face: count/n_rolls for face, count in roll_counts.items()}

print(f"\nDice roll results (n={n_rolls}):")
for face in sorted(roll_probs.keys()):
    print(f"Face {face}: {roll_counts[face]} (P({face}) = {roll_probs[face]:.3f})")

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Coin flip visualization
faces = list(flip_counts.keys())
counts = list(flip_counts.values())
ax1.bar(faces, counts, color=['gold', 'silver'])
ax1.set_title('Coin Flip Results')
ax1.set_ylabel('Count')
ax1.set_ylim(0, max(counts) * 1.1)

# Dice roll visualization
dice_faces = list(roll_counts.keys())
dice_counts = list(roll_counts.values())
ax2.bar(dice_faces, dice_counts, color='skyblue')
ax2.set_title('Dice Roll Results')
ax2.set_xlabel('Face')
ax2.set_ylabel('Count')
ax2.set_ylim(0, max(dice_counts) * 1.1)

plt.tight_layout()
plt.show()

# Conditional probability example
def simulate_conditional_probability():
    """Simulate conditional probability: P(A|B) where A=sum>10, B=first die=6"""
    n_simulations = 10000
    count_b = 0  # First die is 6
    count_a_and_b = 0  # Sum > 10 AND first die is 6
    
    for _ in range(n_simulations):
        die1 = np.random.randint(1, 7)
        die2 = np.random.randint(1, 7)
        total = die1 + die2
        
        if die1 == 6:  # Event B
            count_b += 1
            if total > 10:  # Event A
                count_a_and_b += 1
    
    p_b = count_b / n_simulations
    p_a_given_b = count_a_and_b / count_b if count_b > 0 else 0
    
    return p_b, p_a_given_b

p_b, p_a_given_b = simulate_conditional_probability()
print(f"\nConditional Probability Example:")
print(f"P(B) = P(first die = 6) = {p_b:.3f}")
print(f"P(A|B) = P(sum > 10 | first die = 6) = {p_a_given_b:.3f}")
```

**Code Annotations:**
- Simulates coin flips and dice rolls to illustrate probability concepts.
- Visualizes empirical probabilities and compares to theoretical values.
- Demonstrates conditional probability estimation by simulation.

> **Tip:** Try increasing the number of simulations to see empirical probabilities approach theoretical values.

---

## Random Variables and Distributions

### Random Variables

A random variable $X$ is a function that assigns a real number to each outcome in the sample space.
- **Discrete random variables:** Take on a countable number of values (e.g., coin flips, dice rolls).
- **Continuous random variables:** Take on uncountably many values (e.g., height, weight).

### Probability Mass Function (PMF)

For a discrete random variable $X$:

```math
p_X(x) = P(X = x)
```

### Probability Density Function (PDF)

For a continuous random variable $X$:

```math
P(a \leq X \leq b) = \int_a^b f_X(x) dx
```

### Cumulative Distribution Function (CDF)

```math
F_X(x) = P(X \leq x)
```

### Expected Value

For discrete $X$:
```math
E[X] = \sum_x x \cdot p_X(x)
```

For continuous $X$:
```math
E[X] = \int_{-\infty}^{\infty} x \cdot f_X(x) dx
```

- **Intuition:** The expected value is the long-run average value of $X$.
- **Step-by-step:**
  - For discrete: Multiply each possible value by its probability, then sum.
  - For continuous: Integrate $x$ times the PDF over all $x$.

### Variance

```math
\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
```

- **Intuition:** Variance measures the spread or dispersion of a random variable.
- **Step-by-step:**
  - Compute the expected value (mean).
  - Subtract the mean from each value, square the result.
  - Take the expected value of these squared differences.

### Common Distributions
- **Bernoulli:** Models binary outcomes (success/failure).
- **Binomial:** Number of successes in $n$ independent Bernoulli trials.
- **Normal (Gaussian):** Bell-shaped, models many natural phenomena.
- **Exponential:** Time between events in a Poisson process.
- **Uniform:** All outcomes equally likely.

> **Tip:** The normal distribution is fundamental in deep learning due to the central limit theorem and its use in weight initialization.

### Python Implementation: Common Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Bernoulli Distribution
def bernoulli_example():
    """Example of Bernoulli distribution (coin flip)"""
    p = 0.6  # probability of success
    n_samples = 1000
    
    # Generate samples (0 or 1)
    samples = np.random.binomial(1, p, n_samples)
    
    # Theoretical PMF
    x = [0, 1]
    pmf = [1-p, p]
    
    # Plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(samples, bins=[-0.5, 0.5, 1.5], alpha=0.7, density=True, label='Samples')
    plt.plot(x, pmf, 'ro-', linewidth=2, label='Theoretical PMF')
    plt.title('Bernoulli Distribution (p=0.6)')
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    # Binomial Distribution
    plt.subplot(1, 2, 2)
    n_trials = 10
    binomial_samples = np.random.binomial(n_trials, p, n_samples)
    
    plt.hist(binomial_samples, bins=range(n_trials+2), alpha=0.7, density=True, label='Samples')
    
    # Theoretical PMF
    x_binom = np.arange(0, n_trials+1)
    pmf_binom = stats.binom.pmf(x_binom, n_trials, p)
    plt.plot(x_binom, pmf_binom, 'ro-', linewidth=2, label='Theoretical PMF')
    
    plt.title(f'Binomial Distribution (n={n_trials}, p={p})')
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Normal Distribution
def normal_distribution_example():
    """Example of normal distribution"""
    mu = 0  # mean
    sigma = 1  # standard deviation
    n_samples = 10000
    
    # Generate samples
    samples = np.random.normal(mu, sigma, n_samples)
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    # Histogram and PDF
    plt.subplot(1, 3, 1)
    plt.hist(samples, bins=50, alpha=0.7, density=True, label='Samples')
    
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    pdf = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, 'r-', linewidth=2, label='Theoretical PDF')
    
    plt.title('Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    # CDF
    plt.subplot(1, 3, 2)
    sorted_samples = np.sort(samples)
    empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    plt.plot(sorted_samples, empirical_cdf, 'b-', alpha=0.7, label='Empirical CDF')
    
    theoretical_cdf = stats.norm.cdf(x, mu, sigma)
    plt.plot(x, theoretical_cdf, 'r-', linewidth=2, label='Theoretical CDF')
    
    plt.title('Cumulative Distribution Function')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    
    # Q-Q plot
    plt.subplot(1, 3, 3)
    stats.probplot(samples, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normal)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Sample mean: {np.mean(samples):.3f} (theoretical: {mu})")
    print(f"Sample std: {np.std(samples):.3f} (theoretical: {sigma})")
    print(f"Sample variance: {np.var(samples):.3f} (theoretical: {sigma**2})")

# Run examples
bernoulli_example()
normal_distribution_example()

# Multiple distributions comparison
def compare_distributions():
    """Compare different probability distributions"""
    n_samples = 10000
    
    # Generate samples from different distributions
    normal_samples = np.random.normal(0, 1, n_samples)
    exponential_samples = np.random.exponential(1, n_samples)
    uniform_samples = np.random.uniform(-2, 2, n_samples)
    
    plt.figure(figsize=(15, 5))
    
    # Histograms
    plt.subplot(1, 3, 1)
    plt.hist(normal_samples, bins=50, alpha=0.7, density=True, label='Samples')
    x = np.linspace(-4, 4, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='Theoretical PDF')
    plt.title('Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.hist(exponential_samples, bins=50, alpha=0.7, density=True, label='Samples')
    x = np.linspace(0, 5, 100)
    plt.plot(x, stats.expon.pdf(x, 0, 1), 'r-', linewidth=2, label='Theoretical PDF')
    plt.title('Exponential Distribution')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.hist(uniform_samples, bins=50, alpha=0.7, density=True, label='Samples')
    x = np.linspace(-2, 2, 100)
    plt.plot(x, stats.uniform.pdf(x, -2, 4), 'r-', linewidth=2, label='Theoretical PDF')
    plt.title('Uniform Distribution')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

compare_distributions()

```

**Code Annotations:**
- Demonstrates Bernoulli, Binomial, Normal, Exponential, and Uniform distributions.
- Shows how to generate samples, plot histograms, and compare to theoretical PMFs/PDFs.
- Visualizes empirical vs. theoretical CDFs and Q-Q plots for normality.
- Compares multiple distributions side by side.

> **Tip:** Try changing the parameters (mean, variance, probability) to see how the distributions change!

---

## Statistical Inference

Statistical inference is the process of drawing conclusions about populations from data. In deep learning, this is crucial for understanding model performance, generalization, and uncertainty.

### Descriptive Statistics

- **Mean:** Average value; gives a sense of the "center" of the data.
- **Variance:** Spread of data; how much the data varies from the mean.
- **Standard deviation:** Square root of variance; also measures spread, but in the same units as the data.
- **Covariance:** How two variables vary together; positive means they increase together, negative means one increases as the other decreases.
- **Correlation:** Strength and direction of linear relationship; ranges from -1 (perfect negative) to +1 (perfect positive).

#### Formulas

Mean:
```math
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
```
Variance:
```math
s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2
```
Standard deviation:
```math
s = \sqrt{s^2}
```
Covariance:
```math
\text{Cov}(X, Y) = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})
```
Correlation:
```math
\rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
```

> **Intuitive Note:**
> - Variance and standard deviation tell you how "spread out" your data is. If all values are close to the mean, variance is low.
> - Covariance and correlation help you understand relationships between variables, which is key in feature engineering for deep learning.

### Python Implementation: Descriptive Statistics

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Generate sample data
np.random.seed(42)
n_samples = 1000

# Generate correlated data
x = np.random.normal(0, 1, n_samples)
y = 0.7 * x + np.random.normal(0, 0.5, n_samples)  # y is correlated with x

# Calculate descriptive statistics
mean_x, mean_y = np.mean(x), np.mean(y)
var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
std_x, std_y = np.std(x, ddof=1), np.std(y, ddof=1)
cov_xy = np.cov(x, y)[0, 1]
corr_xy = np.corrcoef(x, y)[0, 1]

print("Descriptive Statistics:")
print(f"X - Mean: {mean_x:.3f}, Variance: {var_x:.3f}, Std: {std_x:.3f}")
print(f"Y - Mean: {mean_y:.3f}, Variance: {var_y:.3f}, Std: {std_y:.3f}")
print(f"Covariance(X,Y): {cov_xy:.3f}")
print(f"Correlation(X,Y): {corr_xy:.3f}")

# Visualize the data
plt.figure(figsize=(15, 5))

# Scatter plot
plt.subplot(1, 3, 1)
plt.scatter(x, y, alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Scatter Plot (ρ = {corr_xy:.3f})')
plt.grid(True)

# Histograms
plt.subplot(1, 3, 2)
plt.hist(x, bins=30, alpha=0.7, label=f'X (μ={mean_x:.2f}, σ={std_x:.2f})')
plt.hist(y, bins=30, alpha=0.7, label=f'Y (μ={mean_y:.2f}, σ={std_y:.2f})')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histograms')
plt.legend()
plt.grid(True)

# Box plots
plt.subplot(1, 3, 3)
plt.boxplot([x, y], labels=['X', 'Y'])
plt.ylabel('Value')
plt.title('Box Plots')
plt.grid(True)

plt.tight_layout()
plt.show()

# Central Limit Theorem demonstration
def central_limit_theorem():
    """Demonstrate the Central Limit Theorem (CLT):
    As sample size increases, the distribution of the sample mean approaches a normal distribution, regardless of the original distribution."""
    n_experiments = 1000
    sample_sizes = [1, 5, 10, 30]
    
    plt.figure(figsize=(15, 10))
    
    for i, n in enumerate(sample_sizes):
        # Generate means of n samples from exponential distribution (not normal!)
        means = []
        for _ in range(n_experiments):
            samples = np.random.exponential(1, n)
            means.append(np.mean(samples))
        
        plt.subplot(2, 2, i+1)
        plt.hist(means, bins=30, alpha=0.7, density=True, label=f'Sample means (n={n})')
        
        # Theoretical normal distribution
        mu_theoretical = 1  # mean of exponential(1)
        sigma_theoretical = 1 / np.sqrt(n)  # std of sample mean
        x = np.linspace(min(means), max(means), 100)
        plt.plot(x, stats.norm.pdf(x, mu_theoretical, sigma_theoretical), 
                'r-', linewidth=2, label='Theoretical Normal')
        
        plt.title(f'CLT: Sample Size = {n}')
        plt.xlabel('Sample Mean')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

central_limit_theorem()
```

**Code Annotations:**
- Computes and visualizes mean, variance, covariance, and correlation.
- Demonstrates the Central Limit Theorem (CLT) with sample means.
- Shows how non-normal data (exponential) leads to normal sample means as n increases.

> **Deep Learning Note:**
> - Understanding variance and correlation is key for feature selection and understanding model input relationships.
> - The CLT explains why neural network weight initializations often use normal distributions.

---

## Hypothesis Testing

Hypothesis testing is a framework for making decisions about populations based on sample data. In deep learning, this is used for model comparison, ablation studies, and validating improvements.

### Null and Alternative Hypotheses

- **Null Hypothesis ($H_0$):** Default assumption (e.g., no effect, no difference)
- **Alternative Hypothesis ($H_1$):** Research hypothesis (e.g., there is an effect)

> **Analogy:**
> - Think of $H_0$ as the "status quo" and $H_1$ as the "challenger." You need strong evidence to reject the status quo.

### P-value

The p-value is the probability of observing data as extreme as or more extreme than the observed data, assuming the null hypothesis is true. A small p-value suggests the observed data is unlikely under $H_0$.

### Significance Level

The significance level $\alpha$ is the threshold for rejecting the null hypothesis (typically 0.05). If $p < \alpha$, reject $H_0$.

### Type I and Type II Errors

- **Type I Error:** Rejecting $H_0$ when it's true (false positive)
- **Type II Error:** Failing to reject $H_0$ when it's false (false negative)

> **Pitfall:**
> - Lowering $\alpha$ reduces Type I errors but increases Type II errors. There's a trade-off!

### Common Tests
- **Z-test:** For population mean with known standard deviation.
- **T-test:** For population mean with unknown standard deviation.
- **Chi-square test:** For independence or goodness of fit.

### Python Implementation: Hypothesis Testing

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# One-sample t-test example
def one_sample_ttest_example():
    """Example of one-sample t-test: tests if the sample mean differs from a hypothesized value."""
    # Generate sample data
    np.random.seed(42)
    sample = np.random.normal(5.2, 1.5, 30)  # True mean = 5.2
    
    # Test H0: μ = 5.0 vs H1: μ ≠ 5.0
    null_mean = 5.0
    t_stat, p_value = stats.ttest_1samp(sample, null_mean)
    
    print("One-Sample T-Test:")
    print(f"Sample mean: {np.mean(sample):.3f}")
    print(f"Sample std: {np.std(sample, ddof=1):.3f}")
    print(f"Null hypothesis: μ = {null_mean}")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(f"Reject H0 at α=0.05: {p_value < 0.05}")
    
    # Visualize
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(sample, bins=15, alpha=0.7, density=True, label='Sample')
    x = np.linspace(min(sample), max(sample), 100)
    plt.plot(x, stats.norm.pdf(x, np.mean(sample), np.std(sample, ddof=1)), 
            'r-', linewidth=2, label='Sample distribution')
    plt.axvline(null_mean, color='g', linestyle='--', linewidth=2, label=f'H0: μ = {null_mean}')
    plt.title('Sample Distribution vs Null Hypothesis')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    # T-distribution
    plt.subplot(1, 2, 2)
    x = np.linspace(-4, 4, 100)
    plt.plot(x, stats.t.pdf(x, df=len(sample)-1), 'b-', linewidth=2, label='t-distribution')
    plt.axvline(t_stat, color='r', linestyle='--', linewidth=2, label=f't-statistic = {t_stat:.3f}')
    plt.axvline(-t_stat, color='r', linestyle='--', linewidth=2)
    plt.fill_between(x, stats.t.pdf(x, df=len(sample)-1), 
                    where=(x <= -abs(t_stat)) | (x >= abs(t_stat)), 
                    alpha=0.3, color='red', label=f'p-value = {p_value:.3f}')
    plt.title('T-Distribution with Critical Region')
    plt.xlabel('t')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Two-sample t-test example
def two_sample_ttest_example():
    """Example of two-sample t-test: tests if two sample means are different."""
    # Generate two samples
    np.random.seed(42)
    sample1 = np.random.normal(5.0, 1.0, 25)
    sample2 = np.random.normal(5.5, 1.0, 25)  # Different mean
    
    # Test H0: μ1 = μ2 vs H1: μ1 ≠ μ2
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    
    print("\nTwo-Sample T-Test:")
    print(f"Sample 1 - Mean: {np.mean(sample1):.3f}, Std: {np.std(sample1, ddof=1):.3f}")
    print(f"Sample 2 - Mean: {np.mean(sample2):.3f}, Std: {np.std(sample2, ddof=1):.3f}")
    print(f"Null hypothesis: μ1 = μ2")
    print(f"T-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(f"Reject H0 at α=0.05: {p_value < 0.05}")
    
    # Visualize
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(sample1, bins=15, alpha=0.7, density=True, label='Sample 1')
    plt.hist(sample2, bins=15, alpha=0.7, density=True, label='Sample 2')
    plt.title('Sample Distributions')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([sample1, sample2], labels=['Sample 1', 'Sample 2'])
    plt.title('Box Plots')
    plt.ylabel('Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Chi-square test example
def chi_square_test_example():
    """Example of chi-square goodness of fit test: tests if observed categorical data matches expected frequencies."""
    # Generate categorical data
    np.random.seed(42)
    categories = ['A', 'B', 'C', 'D']
    observed = np.random.multinomial(100, [0.25, 0.25, 0.25, 0.25])
    expected = [25, 25, 25, 25]  # Expected under null hypothesis
    
    # Chi-square test
    chi2_stat, p_value = stats.chisquare(observed, expected)
    
    print("\nChi-Square Goodness of Fit Test:")
    print("Observed counts:", dict(zip(categories, observed)))
    print("Expected counts:", dict(zip(categories, expected)))
    print(f"Chi-square statistic: {chi2_stat:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(f"Reject H0 at α=0.05: {p_value < 0.05}")
    
    # Visualize
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    x = np.arange(len(categories))
    width = 0.35
    plt.bar(x - width/2, observed, width, label='Observed', alpha=0.7)
    plt.bar(x + width/2, expected, width, label='Expected', alpha=0.7)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Observed vs Expected Counts')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    residuals = observed - expected
    plt.bar(categories, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Category')
    plt.ylabel('Residual (Observed - Expected)')
    plt.title('Residuals')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run hypothesis testing examples
one_sample_ttest_example()
two_sample_ttest_example()
chi_square_test_example()
```

**Code Annotations:**
- Demonstrates one-sample and two-sample t-tests, and chi-square test.
- Visualizes distributions, critical regions, and test statistics.
- Shows how to interpret p-values and test results.

> **Deep Learning Note:**
> - Use hypothesis testing to compare model variants or check if improvements are statistically significant.

---

## Bayesian Statistics

Bayesian statistics provides a framework for updating beliefs in light of new data. In deep learning, Bayesian methods are used for uncertainty estimation and regularization.

### Bayes' Theorem Revisited

```math
P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}
```

Where:
- $P(\theta|D)$ is the posterior probability (updated belief after seeing data)
- $P(D|\theta)$ is the likelihood (how likely data is given parameters)
- $P(\theta)$ is the prior probability (belief before seeing data)
- $P(D)$ is the evidence (normalizing constant)

### Prior, Likelihood, and Posterior

- **Prior:** Initial belief about parameters before seeing data
- **Likelihood:** How well the data supports different parameter values
- **Posterior:** Updated belief after seeing the data

> **Analogy:**
> - Prior is your initial guess, likelihood is how well your guess explains the data, posterior is your updated guess.

### Python Implementation: Bayesian Inference

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Bayesian inference for coin flip
def bayesian_coin_flip():
    """Bayesian inference for coin flip probability: updates belief about probability of heads after observing data."""
    # Prior: Beta distribution (conjugate prior for binomial)
    alpha_prior, beta_prior = 2, 2  # Beta(2,2) - slightly favors 0.5
    
    # Data: observed flips
    n_flips = 20
    n_heads = 14
    
    # Likelihood: binomial distribution
    # Posterior: Beta(alpha_prior + n_heads, beta_prior + n_tails)
    alpha_posterior = alpha_prior + n_heads
    beta_posterior = beta_prior + (n_flips - n_heads)
    
    # Generate points for plotting
    p_values = np.linspace(0, 1, 1000)
    
    # Prior, likelihood, and posterior
    prior = stats.beta.pdf(p_values, alpha_prior, beta_prior)
    likelihood = stats.binom.pmf(n_heads, n_flips, p_values)
    posterior = stats.beta.pdf(p_values, alpha_posterior, beta_posterior)
    
    # Normalize likelihood for plotting
    likelihood = likelihood / np.max(likelihood) * np.max(posterior)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(p_values, prior, 'b-', linewidth=2, label='Prior: Beta(2,2)')
    plt.title('Prior Distribution')
    plt.xlabel('p (probability of heads)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(p_values, likelihood, 'g-', linewidth=2, label='Likelihood')
    plt.title('Likelihood Function')
    plt.xlabel('p (probability of heads)')
    plt.ylabel('Likelihood (normalized)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(p_values, posterior, 'r-', linewidth=2, label='Posterior: Beta(16,8)')
    plt.title('Posterior Distribution')
    plt.xlabel('p (probability of heads)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Credible interval
    credible_interval = stats.beta.interval(0.95, alpha_posterior, beta_posterior)
    print(f"Data: {n_heads} heads out of {n_flips} flips")
    print(f"Posterior mean: {alpha_posterior / (alpha_posterior + beta_posterior):.3f}")
    print(f"95% credible interval: [{credible_interval[0]:.3f}, {credible_interval[1]:.3f}]")

# Bayesian linear regression
def bayesian_linear_regression():
    """Simple Bayesian linear regression: infers a distribution over regression parameters."""
    # Generate synthetic data
    np.random.seed(42)
    n_points = 20
    x = np.linspace(0, 10, n_points)
    true_slope = 2.0
    true_intercept = 1.0
    noise_std = 1.0
    
    y = true_slope * x + true_intercept + np.random.normal(0, noise_std, n_points)
    
    # Prior: normal distributions for slope and intercept
    slope_prior_mean, slope_prior_std = 0, 10
    intercept_prior_mean, intercept_prior_std = 0, 10
    
    # Posterior (assuming known noise variance)
    # For simplicity, we'll use the analytical solution
    X = np.column_stack([np.ones_like(x), x])
    posterior_cov = np.linalg.inv(X.T @ X / noise_std**2 + 
                                 np.diag([1/intercept_prior_std**2, 1/slope_prior_std**2]))
    posterior_mean = posterior_cov @ (X.T @ y / noise_std**2 + 
                                    np.array([intercept_prior_mean/intercept_prior_std**2,
                                             slope_prior_mean/slope_prior_std**2]))
    
    # Sample from posterior
    n_samples = 1000
    posterior_samples = np.random.multivariate_normal(posterior_mean, posterior_cov, n_samples)
    
    plt.figure(figsize=(15, 5))
    
    # Data and regression lines
    plt.subplot(1, 3, 1)
    plt.scatter(x, y, alpha=0.7, label='Data')
    
    # Plot sample regression lines
    for i in range(0, n_samples, 100):
        slope, intercept = posterior_samples[i]
        plt.plot(x, slope * x + intercept, 'r-', alpha=0.1)
    
    # True line
    plt.plot(x, true_slope * x + true_intercept, 'g-', linewidth=2, label='True')
    # Posterior mean line
    plt.plot(x, posterior_mean[1] * x + posterior_mean[0], 'b-', linewidth=2, label='Posterior Mean')
    
    plt.title('Bayesian Linear Regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # Posterior distributions
    plt.subplot(1, 3, 2)
    plt.hist(posterior_samples[:, 0], bins=30, alpha=0.7, density=True, label='Intercept')
    plt.axvline(true_intercept, color='g', linestyle='--', linewidth=2, label='True')
    plt.axvline(posterior_mean[0], color='b', linestyle='--', linewidth=2, label='Posterior Mean')
    plt.title('Posterior Distribution: Intercept')
    plt.xlabel('Intercept')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.hist(posterior_samples[:, 1], bins=30, alpha=0.7, density=True, label='Slope')
    plt.axvline(true_slope, color='g', linestyle='--', linewidth=2, label='True')
    plt.axvline(posterior_mean[1], color='b', linestyle='--', linewidth=2, label='Posterior Mean')
    plt.title('Posterior Distribution: Slope')
    plt.xlabel('Slope')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"True parameters: intercept={true_intercept}, slope={true_slope}")
    print(f"Posterior means: intercept={posterior_mean[0]:.3f}, slope={posterior_mean[1]:.3f}")

# Run Bayesian examples
bayesian_coin_flip()
bayesian_linear_regression()
```

**Code Annotations:**
- Demonstrates Bayesian inference for coin flips and linear regression.
- Visualizes prior, likelihood, and posterior distributions.
- Shows credible intervals and posterior samples.
- Explains how Bayesian methods quantify uncertainty in predictions and parameters.

> **Deep Learning Note:**
> - Bayesian neural networks use similar principles to estimate uncertainty in weights and predictions.

---

## Applications in Deep Learning

Probability and statistics are deeply integrated into deep learning:

### Loss Functions

- **Mean Squared Error (MSE):** Assumes Gaussian noise in regression.
- **Cross-Entropy Loss:** Assumes Bernoulli or categorical likelihood in classification.

### Regularization

- **L1 Regularization (Lasso):** Encourages sparsity in weights.
- **L2 Regularization (Ridge):** Penalizes large weights, encourages smoothness.
- **Dropout:** Randomly drops units during training to prevent overfitting; can be interpreted as approximate Bayesian inference.

### Uncertainty Quantification

- **Bayesian neural networks:** Place distributions over weights.
- **Dropout as Bayesian approximation:** Use dropout at test time to estimate prediction uncertainty.

### Python Implementation: Deep Learning Applications

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Generate classification data
def generate_classification_data():
    """Generate synthetic classification data"""
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                             n_redundant=5, n_clusters_per_class=1, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Compare different regularization methods
def compare_regularization():
    """Compare L1 and L2 regularization"""
    X_train, X_test, y_train, y_test = generate_classification_data()
    
    # Models with different regularization
    models = {
        'No regularization': LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000),
        'L1 regularization': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000),
        'L2 regularization': LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'coefficients': model.coef_[0],
            'intercept': model.intercept_[0]
        }
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Accuracy comparison
    plt.subplot(1, 3, 1)
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    plt.bar(names, accuracies, alpha=0.7)
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Coefficient comparison
    plt.subplot(1, 3, 2)
    for name in names:
        coefs = results[name]['coefficients']
        plt.plot(range(len(coefs)), coefs, 'o-', label=name, alpha=0.7)
    plt.title('Coefficient Comparison')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.legend()
    plt.grid(True)
    
    # Coefficient magnitude distribution
    plt.subplot(1, 3, 3)
    for name in names:
        coefs = np.abs(results[name]['coefficients'])
        plt.hist(coefs, bins=20, alpha=0.7, label=name)
    plt.title('Coefficient Magnitude Distribution')
    plt.xlabel('|Coefficient|')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Accuracy: {result['accuracy']:.3f}")
        print(f"  Non-zero coefficients: {np.sum(result['coefficients'] != 0)}")
        print(f"  Mean |coefficient|: {np.mean(np.abs(result['coefficients'])):.3f}")

# Uncertainty quantification with dropout
def dropout_uncertainty():
    """Demonstrate uncertainty quantification with dropout"""
    # Simple neural network with dropout
    class DropoutNetwork:
        def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
            self.W1 = np.random.randn(input_size, hidden_size) * 0.01
            self.b1 = np.zeros(hidden_size)
            self.W2 = np.random.randn(hidden_size, output_size) * 0.01
            self.b2 = np.zeros(output_size)
            self.dropout_rate = dropout_rate
        
        def relu(self, x):
            return np.maximum(0, x)
        
        def forward(self, x, training=True):
            # First layer
            z1 = x @ self.W1 + self.b1
            a1 = self.relu(z1)
            
            # Dropout
            if training:
                mask1 = np.random.binomial(1, 1-self.dropout_rate, size=a1.shape) / (1-self.dropout_rate)
                a1 = a1 * mask1
            
            # Output layer
            z2 = a1 @ self.W2 + self.b2
            return z2
        
        def predict_with_uncertainty(self, x, n_samples=100):
            """Make predictions with uncertainty estimation"""
            predictions = []
            for _ in range(n_samples):
                pred = self.forward(x, training=True)  # Keep dropout on for uncertainty
                predictions.append(pred)
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            return mean_pred, std_pred
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 100)
    
    # Train network (simplified training)
    network = DropoutNetwork(input_size=5, hidden_size=10, output_size=1, dropout_rate=0.3)
    
    # Make predictions with uncertainty
    mean_pred, std_pred = network.predict_with_uncertainty(X, n_samples=100)
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    # Predictions vs true values
    plt.subplot(1, 3, 1)
    plt.scatter(y, mean_pred, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs True Values')
    plt.grid(True)
    
    # Uncertainty vs prediction error
    plt.subplot(1, 3, 2)
    errors = np.abs(y - mean_pred.flatten())
    plt.scatter(std_pred.flatten(), errors, alpha=0.7)
    plt.xlabel('Prediction Uncertainty (std)')
    plt.ylabel('Absolute Error')
    plt.title('Uncertainty vs Error')
    plt.grid(True)
    
    # Uncertainty distribution
    plt.subplot(1, 3, 3)
    plt.hist(std_pred.flatten(), bins=20, alpha=0.7)
    plt.xlabel('Prediction Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Mean prediction error: {np.mean(errors):.3f}")
    print(f"Mean uncertainty: {np.mean(std_pred):.3f}")
    print(f"Correlation between uncertainty and error: {np.corrcoef(std_pred.flatten(), errors)[0,1]:.3f}")

# Run deep learning applications
compare_regularization()
dropout_uncertainty()
```

**Code Annotations:**
- Compares L1, L2, and no regularization in logistic regression.
- Demonstrates uncertainty quantification with dropout in a neural network.
- Visualizes accuracy, coefficient distributions, and uncertainty.
- Shows how regularization affects model weights and performance.
- Illustrates the relationship between prediction uncertainty and error.

> **Deep Learning Note:**
> - Regularization and uncertainty quantification are essential for robust, generalizable models.

---

## Summary

Probability and statistics are fundamental to deep learning because:

1. **Uncertainty quantification** helps understand model confidence
2. **Loss functions** are based on statistical principles
3. **Regularization** techniques use statistical concepts
4. **Model evaluation** relies on statistical measures
5. **Bayesian methods** provide principled uncertainty estimation

Key concepts include:
- **Probability distributions** for modeling data and parameters
- **Statistical inference** for drawing conclusions from data
- **Hypothesis testing** for model validation
- **Bayesian inference** for uncertainty quantification
- **Regularization** for preventing overfitting

Understanding these concepts enables:
- Better model design and evaluation
- Proper interpretation of results
- Robust uncertainty quantification
- Informed decision-making in model selection

---

## Further Reading

- **"Probability and Statistics"** by Morris H. DeGroot and Mark J. Schervish
- **"Statistical Inference"** by George Casella and Roger L. Berger
- **"Bayesian Data Analysis"** by Andrew Gelman et al.
- **"Pattern Recognition and Machine Learning"** by Christopher Bishop 