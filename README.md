# Entropy for vibration signal
This is a programme designed to calculate the entropy for vibration signal identification.

## Entropy available
* improved Slope Entropy  
* ***Slope Entropy:***
   * Slope Entropy encoding each subsequence of length m drawn from X based on the slope generated by two consecutive data samples. Thus, each subsequence can be transformed into another subsequence of length m-1 with the differences of each pair of consecutive samples, x<sub>i</sub>~-x<sub>i-1</sub>. Then, two thresholds is applied to these differences in order to obtain the corresponding symbolic representation, and will ultimately map these differnces into 5 classes. In the end, probability of patterns with embedding dimension set by users is calculated to obtain the entropy value.
   * **Slopen(seq, m=3, gamma=1, delta=0.001, detail=False)**
   * **Parameters:**
       * **seq:      list**      Time series.
       * **m:        int**       Length of subsequence (Embedding dimension).
       * **gamma:    float**     Upper threshold.
       * **delta:    float**     Lower threshold.
       * **detail:   bool**      Return the dictionary of frequency corresponding to each patterns.
   * **Return:**   float
---
* ***Attention Entropy:***
   * Attention Entropy is calculated in three main steps: (1) define the key patterns (define peak points as the key patterns, including local maxima and local minima); (2) calculate the intervals between two adjacent key patterns (intervals of local maxima to local maxima, local minima to local minima, local maxima to local minima, local minima to local maxima; (3) calculate Shannon entropy of intervals.
   * **Aten(seq, threshold=0.0001, window=False, detail=False)**
   * **Parameters:**
      * **seq:       list**      Time series.
      * **threshold: float**     The difference between two consective local maximum and local minimum must be larger that the value of $threshold*std(seq)$.
      * **window:   bool**       Use sliding wondow to calculate the value of local threshold $threshold*std(seq[i-10:i+10])$. It should be noted that this will seriously reduce calculating speed.
      * **detail:    bool**      Return the dictionary of frequency corresponding to each patterns.
   * **Return:**   float
---
* ***Shannon Entropy***:
   * Discretize the signal sequence $x_i$ and then calculate the Shannon Entropy using the frequency of each segment representing the probability of each segment.
   * **Shen(seq, n=10, detail=False)**
   * **Parameters:**
      * **seq:       list**      Time series.
      * **n:         int**       The number of segments that sequence will be discretized into. The length of each segment is $(max({x_i})-min({x_i}))/n$
      * **detail:    bool**      Return the dictionary of frequency corresponding to each patterns.
   * **Return:**   float
---
* ***Dispersion Entropy:***
   * Dispersion Entrpy firstly apply normal cumulative distribution function (NCDF) to map x<sub>i</sub> to y<sub>i</sub>, and then y<sub>i</sub> are mapped to $c$ classes with integer indices from 1 to $c$ using formula $z_i=round(c*y_i+0.5)$. But it should be noted that there are a number of linear and nonlinear mapping techniques can be used, not just NCDF. In the end, probability of patterns with embedding dimension set by users is calculated to obtain the entropy value.
   * **Disen(seq, m=3, c=3, d=1, detail=False)**
   * **Parameters:**
      * **seq:       list**      Time series.
      * **m:         int**       Length of subsequence (Embedding dimension).
      * **c:         int**       The number of classes that x~i will be mapped to.
      * **d:         int**       Time delay.
      * **detail:    bool**      Return the dictionary of frequency corresponding to each patterns.
   * **Return:**   float
---
* ***Permutation Entropy:***
  * Permutation Entropy is calculated by capturing the order relations between values of a time series and extracting a probability distribution of the ordinal patterns. After partitioning the one-dimensional time series, the subsequences are mapped into unique permutations that capture the ordinal rankings of the data. In the end, probability of permutations (ordinal patterns) is calculated to obtain the entropy value.
  * **Pren(seq, m=3, d=1, detail=False)**
  * **Parameters:**
     * **seq:       list**      Time series.
     * **m:         int**       Length of subsequence (Embedding dimension).
     * **d:         int**       Time delay.
     * **detail:    bool**      Return the dictionary of frequency corresponding to each patterns.
  * **Return:**   float
---
* ***Approximate Entropy & Sample Entropy & Fuzzy Entropy***
  * The principle of these three kinds of entropy is similar, and all measure the probability that a time series will produce a new pattern when its embedding dimension changes. Different from Approximate Entropy, Sample Entropy doesn't contain comparison to its own data segment. Instead of utilizing unit step function as Approximate Entropy and Sample Entropy, Fuzzy Entropy improved the method to evaluate the similarity between different subsequences. All the three entropies use Chebychev distance to calculate the distancu of two subsequences.
  * It should be noted that all the three entropies don't have parameter **detail**, so they doesn't have corresponding Refined Composite Multiscale method.
  * **Apen(seq, m=3, r=0.2)**
  * **Smpen(seq, m=3, r=0.2)**
  * **Fuzen(seq, m=3, r=0.2, n=2)**
  * **Parameters:**
     * **seq:       list**      Time series.
     * **m:         int**       Length of subsequence (Embedding dimension).
     * **r:         float**     Ratio of similarity tolerance to sequence standard deviation.
     * **n:         int**       Gradient of similarity tolerance boundary.
## Methods for Improving
* ***Coarse graining method***
  * This function is used in the fuctions of Multiscale entropy, Composite multiscale entropy and Refined composite multiscale entropy for coarse graining.
  * **ms(seq,k,func=average_coarse)**
  * **Parameters:**
     * **seq:       list**      Time series.
     * **k:         int**       Length of subsequences for coarse graining.
     * **func:      function**  Method used for coarse graining. Includong average_coarse, max_coarse, min_coarse, root_mean_square_coarse and variance_coarse.
* ***Multiscale entropy***
  * **Multiscale(seq,tao,entropy,\*\*kwargs)**
  * **Parameters:**
     * **seq:        list**      Time series.
     * **tao:        int**       Length of partitioned subsequences while employing multiscale method.
     * **entropy:    function**  Method for calculating entropy (in **Entropy available**).
     * **\*\*kwargs: key-value** Parameters of utilized entropy function that need to be revised.
* ***Time shift multiscale entropy***
  * **Time_shift_multiscale(seq,tao,entropy,\*\*kwargs)**
  * **Parameters:**
     * **seq:        list**      Time series.
     * **tao:        int**       Length of partitioned subsequences while employing multiscale method.
     * **entropy:    function**  Method for calculating entropy (in **Entropy available**).
     * **\*\*kwargs: key-value** Parameters of utilized entropy function that need to be revised.
* ***Composite multiscale entropy***
  * **Composite_multiscale(seq,tao,entropy,\*\*kwargs)**
  * **Parameters:**
     * **seq:        list**      Time series.
     * **tao:        int**       Length of partitioned subsequences while employing multiscale method.
     * **entropy:    function**  Method for calculating entropy (in **Entropy available**).
     * **\*\*kwargs: key-value** Parameters of utilized entropy function that need to be revised.
* ***Refined composite multiscale entropy***
  * **Refined_composite_multiscale(seq,tao,entropy,\*\*kwargs)**
  * **Parameters:**
     * **seq:        list**      Time series.
     * **tao:        int**       Length of partitioned subsequences while employing multiscale method.
     * **entropy:    function**  Method for calculating entropy (in **Entropy available**).
     * **\*\*kwargs: key-value** Parameters of utilized entropy function that need to be revised.
