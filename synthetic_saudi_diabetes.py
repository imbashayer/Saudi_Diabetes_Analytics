
# --- Original notebook cell 2 ---

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import truncnorm

SEED = 42
N = 10_000

print("Libraries imported successfully.")
print(f"Sample size: {N:,}")


# --- Original notebook cell 4 ---

# 1. POPULATION: Saudi nationals, census 2022 (GAStat), age 15+

BANDS = [
    (15, 19), (20, 24), (25, 29), (30, 34), (35, 39),
    (40, 44), (45, 49), (50, 54), (55, 59), (60, 64),
    (65, 85)
]

POP_2022 = {
    "M": [
        941_806, 868_712, 859_531, 770_595, 679_028,
        534_889, 381_557, 310_703, 252_068, 186_840,
        324_279
    ],
    "F": [
        916_439, 850_780, 842_731, 758_562, 675_023,
        543_104, 404_356, 343_291, 276_936, 203_802,
        335_592
    ],
}

POP_SAUDI_15PLUS = sum(sum(v) for v in POP_2022.values())

REPORT_GROUPS = [
    "15-24", "25-34", "35-44", "45-54", "55-64", "65+"
]

REPORT_BINS = [14, 24, 34, 44, 54, 64, 120]


print("Population parameters loaded.")


# --- Original notebook cell 6 ---

SCENARIOS = {
    "low": (
        0.085,
        "GASTAT 2017 household survey, self-report (VERIFIED)",
        0.195,
        "meta-analysis lower 95% CI (VERIFIED)"
    ),
    "central": (
        0.134,
        "SHIS 2013, HbA1c>=6.5 or medication (VERIFIED)",
        0.241,
        "meta-analysis pooled 24.1% (VERIFIED)"
    ),
    "high": (
        0.200,
        "illustrative high scenario (ASSUMPTION)",
        0.294,
        "meta-analysis upper 95% CI (VERIFIED)"
    ),
}

CASCADE = {
    "undiagnosed": 0.578,
    "treated_uncontrolled": 0.202,
    "treated_controlled": 0.166,
    "untreated": 0.054,
}

BMI_CATS = [
    "underweight", "normal", "overweight", "obese", "morbid"
]

BMI_SHIS = {
    "M": [0.074, 0.356, 0.331, 0.213, 0.026],
    "F": [0.064, 0.327, 0.278, 0.284, 0.047],
}

OBESITY_TARGET = {"M": 0.241, "F": 0.335}

GDM_RANGE = (0.13, 0.176)


# --- Original notebook cell 8 ---

LN = np.log

B = {
    "male": 0.20,
    "family_hx": 0.50,

    "bmi": {
        "underweight": 0.0,
        "normal": 0.0,
        "overweight": LN(3.20),
        "obese": LN(3.20) + 0.30,
        "morbid": LN(3.20) + 0.60,
    },

    "pre_overweight_plus": LN(2.50),
    "pre_age_per_decade": 0.30,
}

AGE_OR_TARGET = 4.47
ENABLE_GDM = False

SCENARIO_LABELS = {
    "low": "Low - exploratory (different definition: GASTAT 2017 self-report)",
    "central": "Central - historical SHIS 2013 benchmark",
    "high": "High - illustrative assumption (NOT an estimate)",
}

P_FAMILY_HX = 0.35

P_PREGNANT_15_49 = 0.06

P_T1_UNDER25 = 0.10
P_T1_25PLUS = 0.01

BMI_AGE_TILT = [-0.60, -0.10, 0.15, 0.35, 0.35]

COMORB = {
    "hypertension": {
        "b0": -2.2,
        "age": 0.45,
        "obese": 0.5,
        "diabetes": LN(5.08),
    },
    "hypercholesterolemia": {
        "b0": -2.4,
        "age": 0.35,
        "obese": 0.4,
        "diabetes": 0.8,
    },
}


# --- Original notebook cell 10 ---

def tn(rng, lo, hi, loc, sd, size):
    a, b = (lo - loc) / sd, (hi - loc) / sd
    return truncnorm.rvs(
        a, b, loc=loc, scale=sd, size=size, random_state=rng
    )


def sample_population(rng, n):
    cells, p = [], []

    for s in ("M", "F"):
        for band, cnt in zip(BANDS, POP_2022[s]):
            cells.append((s, band))
            p.append(cnt)

    p = np.array(p) / sum(p)

    pick = rng.choice(len(cells), n, p=p)

    sex = np.array([cells[i][0] for i in pick])

    age = np.array([
        rng.integers(
            cells[i][1][0],
            cells[i][1][1] + 1
        )
        for i in pick
    ])

    return sex, age


def bmi_categories(rng, sex, age):
    """Age-tilted BMI category, with per-sex marginals re-fitted to SHIS 2013."""

    z = (np.minimum(age, 60) - 40) / 15
    cat = np.empty(len(age), dtype=object)

    for s in ("M", "F"):
        idx = sex == s
        target = np.array(BMI_SHIS[s])
        base = target.copy()

        for _ in range(40):
            w = base[None, :] * np.exp(
                np.outer(z[idx], BMI_AGE_TILT)
            )
            w /= w.sum(1, keepdims=True)
            base *= target / w.mean(0)

        w = base[None, :] * np.exp(
            np.outer(z[idx], BMI_AGE_TILT)
        )
        w /= w.sum(1, keepdims=True)

        u = rng.random(idx.sum())[:, None]

        cat[idx] = np.array(BMI_CATS)[
            (u > np.cumsum(w, 1)).sum(1).clip(0, 4)
        ]

    return cat


def solve_intercept(lp, target):
    return brentq(
        lambda t: expit(t + lp).mean() - target,
        -25,
        5
    )


def expected_or_ge40(p, age):
    hi = age >= 40

    a, b = p[hi].sum(), (1 - p[hi]).sum()
    c, d = p[~hi].sum(), (1 - p[~hi]).sum()

    return (a * d) / (b * c)


def calibrate_age_slope(
    lp_other, age, d_prev, target_or=AGE_OR_TARGET
):
    """Calibrate the age slope to a target crude OR."""

    def f(sl):
        lp = lp_other + sl * (age - 45) / 10
        t = solve_intercept(lp, d_prev)

        return (
            np.log(expected_or_ge40(expit(t + lp), age))
            - np.log(target_or)
        )

    return brentq(f, 0.01, 4.0)


# --- Original notebook cell 12 ---

def generate(scenario="central", n=N, seed=SEED):
    rng = np.random.default_rng(seed)

    d_prev, _, p_pre, _ = SCENARIOS[scenario]

    sex, age = sample_population(rng, n)
    male = (sex == "M").astype(int)

    grp = pd.cut(
        age,
        REPORT_BINS,
        labels=REPORT_GROUPS
    ).astype(str)

    cat = bmi_categories(rng, sex, age)

    lo_hi = {
        "underweight": (15, 18.5),
        "normal": (18.5, 25),
        "overweight": (25, 30),
        "obese": (30, 40),
    }

    bmi = np.empty(n)

    for c in BMI_CATS:
        m = cat == c

        bmi[m] = (
            np.minimum(
                40 + rng.exponential(3, m.sum()),
                60
            )
            if c == "morbid"
            else rng.uniform(*lo_hi[c], m.sum())
        )

    obese = np.isin(
        cat, ["obese", "morbid"]
    ).astype(int)

    ow_plus = np.isin(
        cat, ["overweight", "obese", "morbid"]
    ).astype(int)

    family_hx = rng.binomial(1, P_FAMILY_HX, n)

    lp_other = (
        B["male"] * male
        + B["family_hx"] * family_hx
        + np.array([B["bmi"][c] for c in cat])
    )

    slope = calibrate_age_slope(
        lp_other, age, d_prev
    )

    lp = lp_other + slope * (age - 45) / 10

    a = solve_intercept(lp, d_prev)
    generate.age_slope = slope

    diabetic = rng.binomial(
        1, expit(a + lp)
    )

    # Prediabetes among non-diabetics
    nd = diabetic == 0

    lp_pre = (
        B["pre_age_per_decade"] * (age - 45) / 10
        + B["pre_overweight_plus"] * ow_plus
    )

    a2 = brentq(
        lambda t: expit(t + lp_pre[nd]).sum() / n - p_pre,
        -20,
        5
    )

    pre = np.zeros(n, dtype=int)

    pre[nd] = rng.binomial(
        1, expit(a2 + lp_pre[nd])
    )

    status = np.where(
        diabetic == 1,
        "diabetes",
        np.where(pre == 1, "prediabetes", "normal")
    )


# --- Original notebook cell 14 ---

# Type 1 / Type 2
p_t1 = np.where( age < 25,
        P_T1_UNDER25,
        P_T1_25PLUS
    )

dtype = np.where(
        diabetic == 1,
        np.where(rng.random(n) < p_t1, "T1", "T2"),
        "none"
    )

extra = {}

if ENABLE_GDM:
        pregnant = (
            (sex == "F")
            & (age <= 49)
            & (rng.random(n) < P_PREGNANT_15_49)
        ).astype(int)

        gdm = (
            (pregnant == 1)
            & (diabetic == 0)
            & (rng.random(n) < np.mean(GDM_RANGE))
        ).astype(int)

        extra = {
            "pregnant": pregnant,
            "gdm": gdm,
        }

    # Care cascade among diabetes cases
cascade = np.full(n, "none", dtype=object)

di = diabetic == 1

cascade[di] = rng.choice(
        list(CASCADE),
        di.sum(),
        p=list(CASCADE.values())
    )


# --- Original notebook cell 16 ---

B = {
    "age_per_decade": 0.55,
    "male": 0.20,
    "family_hx": 0.50,

    "bmi": {
        "underweight": 0.0,
        "normal": 0.0,
        "overweight": 0.40,
        "obese": 0.90,
        "morbid": 1.30
    },

    "pre_age_per_decade": 0.30,
    "pre_obese": 0.50
}

P_FAMILY_HX = 0.35
P_PREGNANT_15_49 = 0.06

P_T1_UNDER25 = 0.10
P_T1_25PLUS = 0.01

COMORB = {
    "hypertension": {
        "b0": -2.2,
        "age": 0.45,
        "obese": 0.5,
        "diabetes": 0.9
    },

    "hypercholesterolemia": {
        "b0": -2.4,
        "age": 0.35,
        "obese": 0.4,
        "diabetes": 0.8
    }
}

print("Assumptions and comorbidity parameters loaded.")


# --- Original notebook cell 18 ---

def calibrate_intercept(lp, target):
    """
    Calibrate the intercept so the mean predicted probability
    equals the target prevalence.
    """
    lp = np.asarray(lp)

    def objective(intercept):
        return expit(intercept + lp).mean() - target

    return brentq(objective, -15, 5)


def tn(lo, hi, loc, sd, size):
    """
    Generate samples from a truncated normal distribution.
    """
    a = (lo - loc) / sd
    b = (hi - loc) / sd

    return truncnorm.rvs(
        a,
        b,
        loc=loc,
        scale=sd,
        size=size,
        random_state=rng
    )


print("Helper functions defined.")
