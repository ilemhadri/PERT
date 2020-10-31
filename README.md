This notebook implements several approaches to importance sampling for the PERT Problem.  
We are given a graph that represents dependencies between tasks. The goal is to assess the probability that the completion time of the final task exceeds a given threshold. The framework is known as Program Evaluation and Review Technique or [PERT](https://en.wikipedia.org/wiki/Program_evaluation_and_review_technique).

The reference here is Art's notes on importance sampling. https://statweb.stanford.edu/~owen/mc/Ch-var-is.pdf
I recommend thoroughly reading these notes before attempting the notebook.  
