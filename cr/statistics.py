import numpy as np
from scipy.stats import ttest_rel


def seed_summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Seed statistics require at least two finite run values")
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "seeds": len(values)}


def paired_comparison(first, second):
    if set(first) != set(second) or len(first) < 2:
        raise ValueError("Paired comparison has the same independent seed identifiers")
    a, b = [np.asarray([values[seed] for seed in sorted(first)], dtype=float) for values in (first, second)]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired seed values are finite")
    differences = a - b
    summary = seed_summary(differences)
    if np.all(differences == 0):
        return dict(summary, statistic=0.0, p_value=1.0, degrees_of_freedom=len(a) - 1)
    if differences.std(ddof=1) == 0:
        return dict(summary, statistic=None, statistic_kind="unbounded", p_value=0.0, degrees_of_freedom=len(a) - 1)
    result = ttest_rel(a, b)
    return dict(summary, statistic=float(result.statistic), p_value=float(result.pvalue), degrees_of_freedom=len(a) - 1)


def holm_correction(p_values, level=0.05):
    if not p_values or not 0 < level < 1:
        raise ValueError("Holm correction requires a nonempty test family and valid level")
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("Family p-values lie in the unit interval")
    ordered = sorted(p_values, key=lambda key: (p_values[key], str(key)))
    output, previous = {}, 0.0
    for rank, key in enumerate(ordered):
        corrected = max(previous, min(1.0, (len(ordered) - rank) * p_values[key]))
        output[key] = {"raw_p": p_values[key], "holm_p": corrected, "rejected": corrected <= level,
                       "family_size": len(ordered), "level": level}
        previous = corrected
    return output


def resolution_budget(variance, resolution):
    if not np.isfinite([variance, resolution]).all() or variance < 0 or resolution <= 0:
        raise ValueError("Measurement variance is nonnegative and target resolution is positive")
    per_exposure = max(1, int(np.ceil(2 * variance / resolution ** 2)))
    return {"samples_per_exposure": per_exposure, "samples_total": 2 * per_exposure,
            "target_standard_error": resolution, "confidence_level": None}


def measurement_agreement(first, second):
    if first.message_ids != second.message_ids:
        raise ValueError("Reliability audit uses paired message identifiers")
    x, y = first.posteriors.ravel(), second.posteriors.ravel()
    correlation = None if np.std(x) == 0 or np.std(y) == 0 else float(np.corrcoef(x, y)[0, 1])
    return {"posterior_correlation": correlation, "action_agreement": float(np.mean(first.actions == second.actions)),
            "posterior_l1_difference": float(np.mean(np.abs(first.posteriors - second.posteriors).sum(axis=1)))}
