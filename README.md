# Robust Variational Inference

Conducted two experiments on Variational Inference under Columbia's Prof Cynthia Rush.

1. Derive and implement Coordinate Ascent Variational Inference (CAVI) algorithm for approximating posterior distribution of a toy Gaussian mixture model. Based off of the review paper [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf). The goal was to become familiar with variational inference on a simple model.

2. Derived and implemented modified CAVI algorithm for probabilistic linear regression to emperically evaluate robustness to model misspecification. The modification consits of using a tempered posterior, downweighting the likelihood by raising it to an exponent. Insipired by theoretical results in [On the Robustness to Misspecification of α-Posteriors and Their Variational Approximations](https://arxiv.org/pdf/2104.08324.pdf). Wrote up derivation and results in a 30 page LaTex document in this repo.


