# Investigating and Mitigating Gender Bias in Static Word Embeddings

### [View the Full Analysis Notebook](./main_analysis.ipynb)

This project conducts a comprehensive analysis of social gender bias in pre-trained word embeddings (GloVe) and implements a mathematical debiasing algorithm to mitigate it.

## Summary

This repository implements and analyzes two foundational methods for quantifying gender bias from the paper by Bolukbasi et al. (2016), "Man is to Computer Programmer as Woman is to Homemaker?":

1.  **Direct Bias:** Measuring the projection of "neutral" words (e.g., 'programmer') onto a gender axis.
2.  **Word Embedding Association Test (WEAT):** A statistical test measuring the association between concepts (e.g., 'Career' vs. 'Family') and attributes (e.g., 'Male' vs. 'Female').

Finally, this project implements and **verifies** a geometric debiasing technique to mitigate the identified biases.

## Methodology

### 1. Defining the Gender Subspace
The gender direction $v_{gender}$ is computed using PCA on the difference vectors for 10 definitional pairs (e.g., $v_{he} - v_{she}$). This creates a robust, stable vector representing the "gender axis" in the embedding space.

* **Math:** $v_{gender} = \text{PCA}_1(\{v_{male} - v_{female}\})$

### 2. Quantifying Bias
-   **Direct Bias:** $Bias(w) = v_w \cdot v_{gender}$
-   **WEAT:** $s(X, Y, A, B) = \sum_{x \in X} s(x, A, B) - \sum_{y \in Y} s(y, A, B)$

### 3. Mitigating Bias
Bias is "neutralized" by removing a word's projection onto the gender axis, making it geometrically orthogonal to the gender concept.
-   **Math:** $v_{debiased} = v_{word} - (v_{word} \cdot v_{gender}) \cdot v_{gender}$

## Key Findings & Limitations

This analysis provides a critical insight into the nature of AI bias and the limits of simple debiasing methods.

1.  **Bias Confirmation:** The original `glove-wiki-gigaword-100` model shows significant, statistically significant gender bias in both Direct and Indirect tests.

2.  **Mitigation Success (Direct Bias):** The geometric debiasing method was **100% successful** at removing *Direct Bias*. All debiased occupation words (e.g., `programmer`, `nurse`) became perfectly neutral (bias $\approx 0.0$).

3.  **Mitigation Failure (Indirect Bias):** This method is **insufficient** for removing *Indirect Associational Bias*.
    * As shown in the "Science vs. Arts" WEAT test, the effect size remained high and statistically significant even after debiasing.
    * **Original Effect Size (d):** 1.124 (p=0.013)
    * **Debiased Effect Size (d):** 1.006 (p=0.026)
    * **Conclusion:** This proves that gender bias is a complex, multi-dimensional subspace. Removing a single direction $v_{gender}$ is not a complete solution, as the debiased words still remain "closer" in the vector space to the cluster of male-associated words.

## How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/anubhuti710/bias-in-embeddings.git](https://github.com/anubhuti710/bias-in-embeddings.git)
    ```
2.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the analysis:
    Open and run the `main_analysis.ipynb` notebook in Jupyter or Google Colab.
