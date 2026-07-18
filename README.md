# DE-2LS-UCOPs-IEEE-CEC-2026-Competition

DE-2LS is a differential evolution method with late-stage local search for **unconstrained single-objective numerical optimization (UCOPs)**. The method is built on RDEx and introduces two conservative late-stage refinements: a smoothed exploitation-biased branch-rate update and a guarded coordinate-pattern local search.

## Competition Result

- **Competition:** IEEE CEC 2026 Unconstrained Single-Objective Numerical Optimization
- **Rank:** **2nd place among 11 entries**

## Paper

- **Title:** *DE-2LS: Differential Evolution with Late-Stage local-search for Unconstrained Single-Objective Numerical Optimization*
- **arXiv:** https://arxiv.org/abs/2606.27762

## Method Summary

DE-2LS preserves the original RDEx evolutionary search engine and adds two late-stage mechanisms:

1. **Smoothed exploitation-biased branch-rate control**
2. **Guarded coordinate-pattern local search**

The design is intentionally conservative: the main RDEx search behavior is preserved, while local search is used as a budget-aware late-stage refinement mechanism.

## Main Findings Reported in the Paper

- DE-2LS improves the original RDEx in a direct head-to-head comparison.
- DE-2LS increases the U-score from **33602.0** to **37448.0**, corresponding to an improvement of **11.45%**.
- In the broader comparison reported in the manuscript, DE-2LS achieves the **best overall U-score** among the compared algorithms.

## Repository Purpose

This repository contains the implementation and experimental material for the DE-2LS method developed for the IEEE CEC 2026 competition on UCOPs.

Typical repository contents may include:

- source code for DE-2LS
- benchmark execution scripts
- result files
- manuscript file

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{chauhan2026de2ls,
  title   = {DE-2LS: Differential Evolution with Late-Stage local-search for Unconstrained Single-Objective Numerical Optimization},
  author  = {Chauhan, Dikshit},
  journal = {arXiv preprint arXiv:2606.27762},
  year    = {2026}
}
```

## Contact

**Dikshit Chauhan**  
Department of Electrical and Computer Engineering  
National University of Singapore  
Email: dikshitchauhan608@gmail.com

## Acknowledgment

This work acknowledges the IEEE CEC numerical optimization competition organizers for providing the U-score evaluation procedure and related competition resources, and the RDEx authors for making their source code available.
