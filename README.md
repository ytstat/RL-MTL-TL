### Implementation details for the experiments in paper "Learning from Similar Linear Representations: Adaptivity, Minimaxity, and Robustness" by Tian et al. (2023)
#### Implementation of different MTL methods
- ARMUL (Duan and Wang, 2023) was implemented by the code in their paper. We included their code in the "benchmarks" folder.
- AdaptRep (Chua et al. 2021) was implemented by the code in their paper. We included their code in the "benchmarks" folder.
- Group Lasso (Yuan and Lin, 2006; Lounici et al. 2009, 2011) was implemented by the R package `RMTL` and we used the Python package `rpy2` to call the functions in R. Please make sure `RMTL` and `rpy2` have been correctly installed.
- All the other methods, including the penalized ERM method (Algorithm 1 in our paper), the spectral method (Algorithm 2 in our paper), ERM (Du et al. 2021; Tripuraneni et al. 2021), Methods-of-moments (Tripuraneni et al. 2021), pooled regression (Crammer et al. 2008; Ben-David et al. 2010), and single-task regression, were implemented by the code in `mtl_func_torch.py`.
- We used `PyTorch` to implement ERM, the penalized ERM, and the spectral method. Please make sure `PyTorch` is correctly installed before running the code.

#### Simulation
- Run each .py code with random seed 0-99
- Section 5.1.1: `sim_h.py`
- Section 5.1.2: `sim_h.py`
- Section 5.1.3: 
	- `sim_T.py`
	- `sim_T_time.py`
- Section 5.1.4: `sim_theta.py`
- Section 5.1.5: `sim_r_adaptive.py`

#### Real data: 
- Download the dataset from UCI Machine Learning Repository: [Link](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- Data pre-processing: `data_prep.py`, running on the local computer, producing `har_standardized.pkl`
- Prediction: `har.py`, run with random seed 0-99

#### Plot:
- `plot.R`: Plot and summarize the results by R packages `ggplot2`, `ggpubr`, `latex2exp`, `dplyr`, and `scales`

#### References
- Tian, Y., Gu, Y., & Feng, Y. (2023). Learning from similar linear representations: Adaptivity, minimaxity, and robustness. _arXiv preprint arXiv:2303.17765_.
- Duan, Y., & Wang, K. (2023). Adaptive and robust multi-task learning. _The Annals of Statistics_, _51_(5), 2015-2039.
- Chua, K., Lei, Q., & Lee, J. D. (2021). How fine-tuning allows for effective meta-learning. _Advances in Neural Information Processing Systems_, _34_, 8871-8884.
- Yuan, M., & Lin, Y. (2006). Model selection and estimation in regression with grouped variables. _Journal of the Royal Statistical Society Series B: Statistical Methodology_, _68_(1), 49-67.
- Lounici, K., Pontil, M., Tsybakov, A. B., & Van De Geer, S. A. (2009, December). Taking advantage of sparsity in multi-task learning. In _COLT 2009-The 22nd Conference on Learning Theory_.
- Lounici, K., Pontil, M., van de Geer, S., & Tsybakov, A. B. (2011). Oracle inequalities and optimal inference under group sparsity. _The Annals of Statistics_, _39_(4), 2164-2204.
- Du, S. S., Hu, W., Kakade, S. M., Lee, J. D., & Lei, Q. (2021). Few-shot learning via learning the representation, provably. In _9th International Conference on Learning Representations, ICLR 2021_.
- Tripuraneni, N., Jin, C., & Jordan, M. (2021, July). Provable meta-learning of linear representations. In _International Conference on Machine Learning_ (pp. 10434-10443). PMLR.
