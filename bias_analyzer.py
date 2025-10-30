#
# A Python module for analyzing and mitigating gender bias in static
# word embeddings. Implements methods from Bolukbasi et al. (2016),
# "Man is to Computer Programmer as Woman is to Homemaker? Debiasing
# Word Embeddings."

import numpy as np
from scipy import spatial
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 
# 1. GENDER DIRECTION COMPUTATION
# 

def compute_gender_direction(embeddings, method='pca', definitional_pairs=None):
    """
    Computes the gender direction vector using specified method.
    """
    if definitional_pairs is None:
        definitional_pairs = [
            ('he', 'she'), ('man', 'woman'), ('boy', 'girl'),
            ('father', 'mother'), ('son', 'daughter'), ('husband', 'wife'),
            ('brother', 'sister'), ('uncle', 'aunt'), ('nephew', 'niece'),
            ('king', 'queen')
        ]

    if method == 'difference':
        print("Computing gender direction using 'difference' method (he - she)...")
        gender_vec = embeddings['he'] - embeddings['she']
    
    elif method == 'pca':
        print(f"Computing gender direction using 'pca' method with {len(definitional_pairs)} pairs...")
        differences = []
        for male_word, female_word in definitional_pairs:
            if male_word in embeddings and female_word in embeddings:
                diff = embeddings[male_word] - embeddings[female_word]
                differences.append(diff)
            else:
                print(f"Warning: Skipping pair ({male_word}, {female_word}) - not in vocabulary.")
        
        if len(differences) == 0:
            raise ValueError("No valid definitional pairs found in vocabulary.")
        
        diff_matrix = np.array(differences)
        pca = PCA(n_components=1)
        pca.fit(diff_matrix)
        gender_vec = pca.components_[0]
        
        explained_variance = pca.explained_variance_ratio_[0]
        print(f"PCA: 1st component explained variance: {explained_variance:.4f}")
    
    else:
        raise ValueError("Method must be 'difference' or 'pca'")
    
    gender_vec_normalized = gender_vec / np.linalg.norm(gender_vec)
    return gender_vec_normalized


def check_direction_stability(embeddings, definitional_pairs):
    """
    Performs a leave-one-out stability analysis on the gender direction.
    """
    print("\n--- Running Gender Direction Stability Analysis (Leave-One-Out) ---")
    
    original_direction = compute_gender_direction(
        embeddings, 
        method='pca', 
        definitional_pairs=definitional_pairs
    )
    
    similarities = []
    
    for i in range(len(definitional_pairs)):
        subset_pairs = definitional_pairs[:i] + definitional_pairs[i+1:]
        
        try:
            subset_direction = compute_gender_direction(
                embeddings, 
                method='pca', 
                definitional_pairs=subset_pairs
            )
            sim = cosine_similarity(original_direction, subset_direction)
            similarities.append(sim)
        except ValueError:
            print(f"  Skipping subset {i}, not enough valid pairs.")
            
    if not similarities:
        print("Could not complete stability analysis.")
        return

    mean_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    
    print(f"Stability results (cosine similarity with full-set direction):")
    print(f"  Mean similarity: {mean_sim:.6f}")
    print(f"  Min similarity:  {min_sim:.6f}")
    print("--- Stability Analysis Complete ---")


# 
# 2. DIRECT BIAS MEASUREMENT
#
def compute_direct_bias(embedding_vector, gender_direction):
    """
    Computes the direct bias of a word vector.
    """
    return np.dot(embedding_vector, gender_direction)


def analyze_words_bias(words, embeddings, gender_direction):
    """
    Analyzes and returns the direct bias for a list of words.
    """
    bias_scores = {}
    
    for word in words:
        if word in embeddings:
            vec = embeddings[word]
            bias = compute_direct_bias(vec, gender_direction)
            bias_scores[word] = bias
        else:
            print(f"Warning: '{word}' (for direct bias) not in vocabulary.")
    
    return bias_scores


def display_bias_results(bias_scores, top_n=10):
    """
    Prints a formatted table of the most biased words.
    """
    sorted_words = sorted(bias_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"DIRECT BIAS ANALYSIS (Top {top_n})")
    print(f"{'='*60}\n")
    
    print(f"Top {top_n} MASCULINE-biased words:")
    print(f"{'Word':<20} {'Bias Score':>15}")
    print("-" * 40)
    for word, score in sorted_words[:top_n]:
        print(f"{word:<20} {score:>15.6f}")
    
    print(f"\nTop {top_n} FEMININE-biased words:")
    print(f"{'Word':<20} {'Bias Score':>15}")
    print("-" * 40)
    for word, score in reversed(sorted_words[-top_n:]):
        print(f"{word:<20} {score:>15.6f}")
    print("\n")


# 
# 3. WEAT (INDIRECT BIAS) MEASUREMENT
# 
def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    """
    return 1 - spatial.distance.cosine(vec1, vec2)


def association_difference(word, set_A, set_B, embeddings):
    """
    Computes s(w, A, B) from the WEAT paper.
    """
    if word not in embeddings:
        return None
    
    word_vec = embeddings[word]
    
    similarities_A = [
        cosine_similarity(word_vec, embeddings[attr_word])
        for attr_word in set_A if attr_word in embeddings
    ]
    
    similarities_B = [
        cosine_similarity(word_vec, embeddings[attr_word])
        for attr_word in set_B if attr_word in embeddings
    ]
    
    if len(similarities_A) == 0 or len(similarities_B) == 0:
        return None
    
    mean_sim_A = np.mean(similarities_A)
    mean_sim_B = np.mean(similarities_B)
    
    return mean_sim_A - mean_sim_B


def weat_effect_size(target_set_X, target_set_Y, attribute_set_A, 
                     attribute_set_B, embeddings):
    """
    Computes the WEAT test statistic and effect size (Cohen's d).
    """
    associations_X = [
        association_difference(word, attribute_set_A, attribute_set_B, embeddings)
        for word in target_set_X
    ]
    associations_Y = [
        association_difference(word, attribute_set_A, attribute_set_B, embeddings)
        for word in target_set_Y
    ]
    
    associations_X = [s for s in associations_X if s is not None]
    associations_Y = [s for s in associations_Y if s is not None]
    
    if len(associations_X) == 0 or len(associations_Y) == 0:
        raise ValueError("Not enough valid words in target sets to run WEAT.")
    
    test_statistic = np.sum(associations_X) - np.sum(associations_Y)
    
    mean_X = np.mean(associations_X)
    mean_Y = np.mean(associations_Y)
    
    all_associations = associations_X + associations_Y
    pooled_std = np.std(all_associations, ddof=1)
    
    if pooled_std == 0:
        effect_size = 0.0
    else:
        effect_size = (mean_X - mean_Y) / pooled_std
    
    return {
        'test_statistic': test_statistic,
        'effect_size': effect_size,
        'associations_X': associations_X,
        'associations_Y': associations_Y,
        'mean_X': mean_X,
        'mean_Y': mean_Y
    }


def weat_permutation_test(target_set_X, target_set_Y, attribute_set_A, 
                          attribute_set_B, embeddings, n_permutations=1000):
    """
    Computes the p-value for WEAT using a permutation test.
    """
    try:
        observed_result = weat_effect_size(
            target_set_X, target_set_Y, attribute_set_A, attribute_set_B, embeddings
        )
    except ValueError as e:
        print(f"Error computing observed WEAT: {e}")
        return None, None
        
    observed_stat = observed_result['test_statistic']
    
    all_targets = list(target_set_X) + list(target_set_Y)
    n_X = len(target_set_X)
    np.random.seed(42)
    
    permuted_stats = []
    for _ in range(n_permutations):
        np.random.shuffle(all_targets)
        perm_X = all_targets[:n_X]
        perm_Y = all_targets[n_X:]
        
        try:
            perm_result = weat_effect_size(
                perm_X, perm_Y, attribute_set_A, attribute_set_B, embeddings
            )
            permuted_stats.append(perm_result['test_statistic'])
        except ValueError:
            continue
    
    p_value = np.mean(np.abs(permuted_stats) >= np.abs(observed_stat))
    
    return p_value, observed_result


#
# 4. BIAS MITIGATION (DEBIASING)
# 

def debias_word_vector(word_vec, gender_direction):
    """
    Neutralizes bias in a single word vector using geometric projection.
    """
    projection_scalar = np.dot(word_vec, gender_direction)
    projection_vector = projection_scalar * gender_direction
    debiased_vec = word_vec - projection_vector
    return debiased_vec


# 
# 5. VISUALIZATION
# 
def plot_bias_distribution(bias_scores, title="Word Bias Distribution"):
    """
    Plots a histogram and bar chart for direct bias scores.
    """
    scores = list(bias_scores.values())
    if not scores:
        print(f"Cannot plot '{title}', no scores provided.")
        return
        
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Neutral')
    plt.xlabel('Bias Score (Feminine <-> Masculine)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{title}\nDistribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sorted_items = sorted(bias_scores.items(), key=lambda x: x[1])
    
    num_display = min(10, len(sorted_items) // 2)
    if num_display < 1 and len(sorted_items) > 0:
         num_display = len(sorted_items)
    elif num_display == 0:
        plt.tight_layout()
        plt.show()
        return

    top_fem = sorted_items[:num_display]
    top_masc = sorted_items[-num_display:]
    
    selected_items = top_fem + top_masc
    selected_words = [w for w, _ in selected_items]
    selected_scores = [s for _, s in selected_items]
    
    colors = ['blue'] * num_display + ['red'] * num_display
    
    plt.barh(range(len(selected_words)), selected_scores, color=colors, alpha=0.7)
    plt.yticks(range(len(selected_words)), selected_words, fontsize=10)
    plt.xlabel('Bias Score', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.title('Most Biased Words', fontsize=14)
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()


def plot_weat_associations(results, target_X_name="Target X", target_Y_name="Target Y"):
    """
    Creates a box plot to visualize WEAT association differences.
    """
    if not results or 'associations_X' not in results or 'associations_Y' not in results:
        print("Cannot plot WEAT, results are incomplete.")
        return

    plt.figure(figsize=(8, 6))
    
    data = [results['associations_X'], results['associations_Y']]
    labels = [target_X_name, target_Y_name]
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True, showmeans=False)
    
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    means = [results['mean_X'], results['mean_Y']]
    plt.plot([1, 2], means, 'D', color='darkred', markersize=8, label='Mean', zorder=3)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.ylabel('Association Difference\n(Assoc. w/ Male - Assoc. w/ Female)', fontsize=12)
    plt.title(f'WEAT Association Differences\nEffect Size (d) = {results["effect_size"]:.3f}', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.show()
