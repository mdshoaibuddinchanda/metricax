#!/usr/bin/env python3
"""
Bayesian Spam Filter using MetricaX

This example demonstrates how to build a simple Bayesian spam filter
using discrete Bayesian updates to classify emails as spam or ham.
"""

from collections import defaultdict

# Import from parent bayesian module
from .. import (
    bayes_update_discrete, bayes_posterior, bayes_odds
)


class BayesianSpamFilter:
    """
    A simple Bayesian spam filter using word frequencies.
    """
    
    def __init__(self):
        self.word_counts_spam = defaultdict(int)
        self.word_counts_ham = defaultdict(int)
        self.total_spam = 0
        self.total_ham = 0
        self.vocabulary = set()
    
    def train(self, emails, labels):
        """
        Train the spam filter on labeled email data.
        
        Args:
            emails: List of email texts (strings)
            labels: List of labels ('spam' or 'ham')
        """
        for email, label in zip(emails, labels):
            words = self._tokenize(email)
            
            if label == 'spam':
                self.total_spam += 1
                for word in words:
                    self.word_counts_spam[word] += 1
                    self.vocabulary.add(word)
            else:  # ham
                self.total_ham += 1
                for word in words:
                    self.word_counts_ham[word] += 1
                    self.vocabulary.add(word)
    
    def classify(self, email):
        """
        Classify an email as spam or ham using Bayes' theorem.
        
        Args:
            email: Email text to classify
            
        Returns:
            tuple: (prediction, probability_spam, probability_ham)
        """
        words = self._tokenize(email)
        
        # Prior probabilities
        prior_spam = self.total_spam / (self.total_spam + self.total_ham)
        prior_ham = self.total_ham / (self.total_spam + self.total_ham)
        
        # Calculate likelihoods for each word
        likelihood_spam = 1.0
        likelihood_ham = 1.0
        
        for word in words:
            if word in self.vocabulary:
                # Laplace smoothing to avoid zero probabilities
                prob_word_given_spam = (self.word_counts_spam[word] + 1) / (self.total_spam + len(self.vocabulary))
                prob_word_given_ham = (self.word_counts_ham[word] + 1) / (self.total_ham + len(self.vocabulary))
                
                likelihood_spam *= prob_word_given_spam
                likelihood_ham *= prob_word_given_ham
        
        # Apply Bayes' theorem using MetricaX
        priors = [prior_spam, prior_ham]
        likelihoods = [likelihood_spam, likelihood_ham]
        
        posteriors = bayes_update_discrete(priors, likelihoods)
        
        prob_spam, prob_ham = posteriors
        prediction = 'spam' if prob_spam > prob_ham else 'ham'
        
        return prediction, prob_spam, prob_ham
    
    def _tokenize(self, text):
        """Simple tokenization - split by spaces and convert to lowercase."""
        return text.lower().split()


def spam_filter_demo():
    """
    Demonstrate the Bayesian spam filter with example data.
    """
    print("📧 Bayesian Spam Filter Demo")
    print("=" * 40)
    
    # Training data (simplified examples)
    training_emails = [
        "buy now cheap viagra pills online",
        "win money fast easy cash prize",
        "free lottery winner congratulations claim",
        "meeting tomorrow at 3pm conference room",
        "project deadline reminder please review",
        "lunch plans this weekend restaurant",
        "discount offer limited time only buy",
        "family vacation photos attached enjoy",
        "work schedule update team meeting",
        "urgent money transfer help needed"
    ]
    
    training_labels = [
        'spam', 'spam', 'spam',  # spam emails
        'ham', 'ham', 'ham',     # legitimate emails  
        'spam',                   # more spam
        'ham', 'ham',            # more legitimate
        'spam'                   # more spam
    ]
    
    # Create and train the filter
    spam_filter = BayesianSpamFilter()
    spam_filter.train(training_emails, training_labels)
    
    print(f"📚 Training completed:")
    print(f"   Spam emails: {spam_filter.total_spam}")
    print(f"   Ham emails: {spam_filter.total_ham}")
    print(f"   Vocabulary size: {len(spam_filter.vocabulary)}")
    print()
    
    # Test emails
    test_emails = [
        "buy cheap pills online now",           # Should be spam
        "meeting scheduled for tomorrow",       # Should be ham
        "congratulations you won money prize",  # Should be spam
        "project update and deadline info",     # Should be ham
        "free offer limited time discount"      # Should be spam
    ]
    
    print("🧪 Classification Results:")
    print("-" * 40)
    
    results = []
    for i, email in enumerate(test_emails, 1):
        prediction, prob_spam, prob_ham = spam_filter.classify(email)
        
        print(f"Email {i}: '{email}'")
        print(f"   Prediction: {prediction.upper()}")
        print(f"   P(Spam): {prob_spam:.3f}")
        print(f"   P(Ham):  {prob_ham:.3f}")
        print(f"   Confidence: {max(prob_spam, prob_ham):.3f}")
        print()
        
        results.append({
            'email': email,
            'prediction': prediction,
            'prob_spam': prob_spam,
            'prob_ham': prob_ham
        })
    
    return results


def demonstrate_bayes_theorem():
    """
    Show the mathematical foundation using MetricaX functions.
    """
    print("🧮 Bayes' Theorem in Action")
    print("=" * 30)
    
    # Example: P(Spam|word="free")
    # Prior: P(Spam) = 0.4 (40% of emails are spam)
    # Likelihood: P("free"|Spam) = 0.8 (80% of spam contains "free")
    # Marginal: P("free") = 0.5 (50% of all emails contain "free")
    
    prior_spam = 0.4
    likelihood_free_given_spam = 0.8
    marginal_free = 0.5
    
    posterior_spam = bayes_posterior(prior_spam, likelihood_free_given_spam, marginal_free)
    
    print(f"Prior P(Spam) = {prior_spam}")
    print(f"Likelihood P('free'|Spam) = {likelihood_free_given_spam}")
    print(f"Marginal P('free') = {marginal_free}")
    print(f"Posterior P(Spam|'free') = {posterior_spam:.3f}")
    print()
    
    # Using odds form
    prior_odds = prior_spam / (1 - prior_spam)  # P(Spam) / P(Ham)
    likelihood_ratio = likelihood_free_given_spam / 0.2  # P(free|Spam) / P(free|Ham)
    
    posterior_odds = bayes_odds(prior_odds, likelihood_ratio)
    posterior_prob = posterior_odds / (1 + posterior_odds)
    
    print(f"Using odds form:")
    print(f"Prior odds = {prior_odds:.3f}")
    print(f"Likelihood ratio = {likelihood_ratio:.3f}")
    print(f"Posterior odds = {posterior_odds:.3f}")
    print(f"Posterior probability = {posterior_prob:.3f}")


def run_example():
    """Main function to run the spam filter example"""
    results = spam_filter_demo()
    print("\n" + "="*50)
    demonstrate_bayes_theorem()
    return results


if __name__ == "__main__":
    run_example()
