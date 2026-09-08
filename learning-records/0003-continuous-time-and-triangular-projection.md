# Continuous Time and Triangular Categorical Projection Established

The learner explored why integer index shifts (`jnp.roll`) fail under irregular continuous time telemetry, inducing cumulative temporal bias, and how SurvTD's triangular categorical projection $\Pi$ coupled with an absorbing terminal boundary rigorously preserves unit probability mass ($\sum p_k \equiv 1.000000$) on arbitrary continuous shifts.

## Evidence
- `notebooks/02_categorical_projection.ipynb` was created, executed, and baked with visual plots comparing continuous off-grid shifted masses against triangular projected discrete PMFs.
- `lessons/0002-continuous-time-and-categorical-projection.html` was published and opened in the browser.

## Implications
- Establishes the non-expansiveness property of $\Pi$ (Lemma 1), unlocking the formal contraction mapping proof (Theorem 1) in Lesson 3.
