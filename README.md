# LLM Lab: Building a Transformer from First Principles

Welcome to the **LLM Lab** on YLabs! In this lab, you will build a complete, generative Transformer engine from first principles using pure **Python and NumPy**.

## 🚀 Lab Overview
This lab takes you from raw text tokenization to continuous latent embeddings, scaled contextual attention matrices, feature standardization with LayerNorm, and deep residual layers, culminating in building and training your own **MicroGPT** model with verified emergent transformer behaviors.

### Sprints & Milestones
- **Sprint 1 · Token Representations & Vector Embeddings** (Vector Metrics, Subword Lexer, Latent Lookup)
- **Sprint 2 · Output Projections & Probability Distributions** (Log-Sum-Exp Normalizer, Entropy Scaling, Loss)
- **Sprint 3 · Multi-Head Attention & Causal Sequence Modeling** (Similarity Core, Temporal Mask, Multi-Head Aggregator)
- **Sprint 4 · Deep Residual Stacks & Normalization Layers** (Standardization, Feature Expansion MLP, Decoder Assembly)
- **Sprint 5 · Stochastic Decoding & Autoregressive Inference** (Tail Truncation, Sequential Rollout Pipeline)
- **Sprint 6 · Model Synthesis & Emergent Behavior Verification** (Batch Optimization, Behavioral Evaluation)

## 🛠️ Testing Your Code
You can run automated tests for any ticket directly in your terminal:
```bash
# Example: Run test for Sprint 3 Ticket 1
python yuvro_scoring_rules/sprint_3/ticket_1/main_test.py
```
Or click the **Run Tests** button in the YLabs IDE header.