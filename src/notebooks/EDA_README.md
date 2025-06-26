# Congressional Vote Prediction Analysis: Exploratory Findings & Modeling Directions

## Overview

This analysis brings together 15 visualizations across 4 notebooks, each exploring a different angle of congressional voting behavior, party alignment, and financial influence. The goal was to surface meaningful patterns and signals that could directly support our modeling approach and feature engineering efforts.

---

## Notebook 00: Foundational Data Exploration

### Key Insights:

**1. Bill Passage Rate by Congress**

- **Observation**: The average bill passage rate across all Congresses is 72.5%.
- **Modeling Implication**: This provides a performance baseline for evaluating classification models.
- **Recommendation**: Use as a benchmark to assess whether our model is truly learning from features or simply reflecting institutional norms.

**2. Policy Area Distribution Over Time**

- **Observation**: "Economics and Public Finance" emerges as the most frequently addressed policy area.
- **Modeling Implication**: Temporal policy salience can be leveraged in time-aware models to reflect issue-specific voting dynamics.

**3. Ideological Distribution (DW-NOMINATE Scores)**

- **Observation**: The ideological landscape is distinctly bimodal, with a standard deviation of 0.450.
- **Modeling Implication**: NOMINATE scores are viable continuous features representing ideological positioning.
- **Additional Insight**: Liberal (< -0.2): 2,505; Conservative (> 0.2): 2,723; Moderates: 289.

**4. Party Cohesion Trends**

- **Observation**: Average intra-party voting agreement stands at 93.2%, underscoring high partisan discipline.
- **Modeling Implication**: Party affiliation is likely the strongest categorical predictor in classification tasks.

---

## Notebook 01: Financial Influence & Alignment

### Key Insights:

**5. Member–Contributor Ideological Alignment**

- **Observation**: 98.8% of members vote in alignment with the ideological leaning of their financial backers.
- **Modeling Implication**: Introduce a binary feature denoting ideological match to capture this highly predictive signal.

**6. Contribution Size vs. Alignment**

- **Observation**: A weak but statistically suggestive correlation (r = -0.047, p = 0.138) exists between contribution amount and alignment.
- **Modeling Implication**: While noisy, contribution size may provide marginal predictive value when included as a continuous feature.

**7. Policy-Specific Financial Influence**

- **Observation**: Financial alignment is most predictive in policy areas such as Arts, Families, and Sports.
- **Modeling Implication**: Introduce interaction features between policy domain and financial alignment to capture domain-specific influence effects.

---

## Notebook 02: Voting Similarity & Polarization

### Key Insights:

**8. Member Voting Similarity Matrix**

- **Observation**: Mean similarity between members is 0.552, with values ranging from -0.17 to 1.00.
- **Modeling Implication**: Similarity matrices enable collaborative filtering methods for member-specific vote prediction.

**9. Intra- vs. Inter-Party Similarities**

- **Observation**: Average within-party similarity is 74.6%, versus 35.9% between parties.
- **Polarization Index**: 0.387 — indicating a strong partisan divide.
- **Modeling Implication**: Use these metrics to construct swing-vote indicators and quantify ideological proximity.

**10. Financial Evolution by Party**

- **Observation**: Individual contributions dominate (56.7%), followed by PACs (42.9%) and party committees (0.5%).
- **Modeling Implication**: Decompose financial sources as time-series predictors to reflect evolving fundraising strategies.

**11. Most Opposed Member Pairs**

- **Observation**: Surprisingly, 7 of the 15 most opposed pairs belong to the same party, suggesting intra-party ideological diversity.
- **Modeling Implication**: High opposition scores can be used to detect outliers or potential swing members.

---

## Notebook 03: Bill Clustering via Feature Engineering

### Key Insights:

**12. PCA-Based Clustering of Bills**

- **Observation**: 28,762 bills grouped into 12 clusters using TF-IDF; silhouette score = 0.037.
- **Modeling Implication**: Cluster membership can serve as a categorical feature reflecting legislative content type.

**13. Policy Area vs. Cluster Mapping**

- **Observation**: Distinct policy clusters emerge, supporting the interpretability of the clustering model.
- **Modeling Implication**: Use cluster labels to refine policy area features and better group similar bills.

**14. Party Support Patterns by Cluster**

- **Observation**: Preliminary results suggest variable party support across clusters.
- **Modeling Implication**: Once real vote data is used, this can support modeling of conditional party loyalty.

**15. Vote Margin Distribution by Cluster**

- **Observation**: Clusters exhibit distinct vote margin characteristics.
- **Modeling Implication**: This can be incorporated as a proxy for bill competitiveness or legislative consensus.

---

## Priority Visualizations for Modeling (High Value):

| Priority | Visualization                            | Rationale                                                 |
| -------- | ---------------------------------------- | --------------------------------------------------------- |
| ⭐⭐⭐   | Member–Contributor Ideological Alignment | shows dominant behavioral predictor (98.8% alignment)     |
| ⭐⭐⭐   | Within vs Between Party Similarity       | polarization and cohesion—central to political prediction |
| ⭐⭐     | Bill Passage Rate by Congress            | useful for outcome baseline                               |
| ⭐⭐     | Member Voting Similarity Matrix          | enables collaborative filtering                           |
| ⭐⭐     | Policy-Specific Money Influence          | highlights financial dynamics at the domain level         |

---

## Feature Engineering Highlights & Recommendations

During EDA and clustering, I found several features as potentially valuable additions to the modeling pipeline. While these features are not yet integrated into the current model, they may be useful for modeling improvment:

1. **Ideology Alignment Flag**: Boolean feature capturing whether a member's vote aligns with donor ideology.
   - From the 98.8% alignment finding in Visualization 5 in notebook 01
2. **Policy-Specific Financial Scores**: Continuous features that quantify alignment strength within each policy domain.
   - From the policy-area correlations in Visualization 7 in notebook 01
3. **Party Cohesion Index**: Aggregate party agreement level for each member per session.
   - From the 93.2% cohesion analysis across both notebooks
4. **Cluster-Based Bill Embeddings**: Category labels from text clustering that capture legislative content similarity.
   - From the 12-cluster TF-IDF analysis in notebook 03
5. **Member Similarity Rankings**: Distance-based features that model coalition or opposition likelihood.
   - From the 100x100 similarity matrix in notebook 02
6. **Opposition Score Metrics**: Quantifies how ideologically distant a member is from others, within or across parties.
   - From the member opposition analysis in notebook 02
7. **Temporal Finance Trends**: Time-series deltas of funding sources, reflecting strategic shifts in influence.
   - From the financial evolution analysis in notebook 02
8. **Vote Margin Bins**: Categorical indicators for competitiveness and contentiousness.
   - From the vote margin distribution framework (simulated data) in notebook 3

\*\* I've left some additional notes on exactly where to find these features in `features.md`. Please let me know if there's any questions!
