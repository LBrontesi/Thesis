# Credit Risk Modeling with Partial Information and Default Contagion

This repository contains the code and materials supporting a thesis on advanced credit risk modeling. The work focuses on pricing defaultable securities—such as corporate bonds and credit default swaps—under varying informational environments.

The thesis develops a framework that accounts for:  
- **Partial vs. complete information**, capturing the effect of unobserved economic states on pricing.  
- **Default contagion**, using self-exciting intensities modeled with Hawkes processes to represent the influence of prior defaults.  
- **Economic regime dynamics**, modeled via Markov chains and Hidden Markov Models.

The repository includes implementations for:  
- Simulating default times under stochastic intensities.  
- Applying filtering techniques to estimate hidden states.  
- Pricing defaultable instruments under different information assumptions.

Overall, the project highlights how limited information and contagion effects impact the valuation of credit-sensitive assets.
