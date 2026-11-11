"""Checks for common-row LOO sensitivity, censoring, and sampling retry policy."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.splines import ordinal_shape_robustness as a


class ShapeRobustnessTests(unittest.TestCase):
    def test_common_mask_changes_every_model_consistently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in a.FORMULAS:
                is_linear = name == "linear"
                pd.DataFrame(
                    {
                        "row": range(4),
                        "year": range(1700, 1704),
                        "loo_i": [-1, -10, -2, -1] if is_linear else [-2, -1, -2, -2],
                        "pareto_k": [0.2, 0.9, 0.2, 0.2]
                        if is_linear
                        else [0.2, 0.2, 0.8, 0.2],
                        "lppd_i": np.zeros(4),
                    }
                ).to_csv(root / f"{name}_loo_pointwise.csv", index=False)
            full, reduced = a.comparison_tables(root, root)
            self.assertNotEqual(full.iloc[0].model, "linear")
            self.assertEqual(reduced.iloc[0].model, "linear")
            self.assertTrue(reduced.n_scored.eq(2).all())
            self.assertTrue(reduced.n_excluded.eq(2).all())
            self.assertEqual(reduced.set_index("model").loc["linear", "elpd_loo"], -2)
            self.assertEqual(
                reduced.set_index("model").loc["quadratic", "elpd_loo"], -4
            )
            self.assertEqual(reduced.equivalent_2se.sum(), 1)

    def test_retry_preserves_draws_seed_and_uses_requested_settings(self):
        class FakeData:
            def to_netcdf(self, path):
                Path(path).write_bytes(b"test checkpoint")

        calls = []

        class FakeModel:
            def fit(self, include_response_params=True, **kwargs):
                calls.append(kwargs)
                return FakeData()

        failed = dict(
            divergences=1,
            max_rhat=1.02,
            min_ess_bulk=300,
            min_ess_tail=500,
            nonfinite_diagnostic_cells=0,
            fails_requested_checks=True,
        )
        passed = {
            **failed,
            "divergences": 0,
            "max_rhat": 1.001,
            "min_ess_bulk": 900,
            "fails_requested_checks": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(
                a,
                "diagnostics",
                side_effect=[(failed, pd.DataFrame()), (passed, pd.DataFrame())],
            ):
                _, attempts = a.fit_checked(
                    FakeModel(),
                    root,
                    "test",
                    draws=4000,
                    tune=2000,
                    chains=4,
                    cores=2,
                    seed=0,
                    target_accept=0.8,
                )
            self.assertEqual(len(attempts), 2)
            self.assertEqual([c["tune"] for c in calls], [2000, 4000])
            self.assertEqual([c["target_accept"] for c in calls], [0.8, 0.95])
            self.assertTrue(
                all(c["draws"] == 4000 and c["random_seed"] == 0 for c in calls)
            )
            self.assertTrue((root / "test.nc").exists())
            self.assertFalse((root / "test.partial.nc").exists())
            self.assertEqual(
                len(json.loads((root / "test_attempts.json").read_text())), 2
            )

    def test_bulk_not_tail_is_the_requested_retry_trigger(self):
        summary = pd.DataFrame(
            {"r_hat": [1.001], "ess_bulk": [1000.0], "ess_tail": [200.0]}
        )
        fake = SimpleNamespace(
            posterior={"beta": None},
            sample_stats=SimpleNamespace(diverging=np.zeros(5)),
        )
        with patch.object(a.az, "summary", return_value=summary):
            diag, _ = a.diagnostics(fake)
        self.assertFalse(diag["fails_requested_checks"])
        summary.loc[0, "r_hat"] = 1.02
        with patch.object(a.az, "summary", return_value=summary):
            diag, _ = a.diagnostics(fake)
        self.assertTrue(diag["fails_requested_checks"])

    def test_long_quantities_include_censoring_and_empty_contrast(self):
        years = np.arange(1576, 1801)
        expected = np.tile(np.linspace(1, 4, len(years)), (10, 1))
        long, _ = a.long_shape_table(expected, years, "spline_4")
        self.assertEqual(len(long), 6)
        last = long[long.quantity.eq("contrast_1800_1950")].iloc[0]
        self.assertTrue(np.isnan(last["median"]))
        self.assertIn("not estimable", last.status)
        decline = long[long.quantity.str.startswith("decline")]
        self.assertTrue(decline.proportion_never_within_range.eq(1).all())
        self.assertTrue(decline["median"].isna().all())
        self.assertTrue(long.interior_knots.eq(1).all())

    def test_reloo_basis_and_priors_remain_fixed(self):
        raw = pd.DataFrame(
            {
                "year": np.r_[np.arange(1500, 1570), np.arange(1800, 1807)],
                "score": np.tile(np.arange(7), 11),
                "author": [f"a{i % 11}" for i in range(77)],
            }
        )
        data, _, _ = a.prepare_data(raw)
        for k in range(3, 7):
            name = f"spline_{k}"
            base = a.bmb.Model(a.FORMULAS[name], data, family="cumulative")
            refit, _ = a.fixed_basis_refit_model(data, name, 0)
            for old, new in zip(
                base.components["p"].common_terms.values(),
                refit.components["p"].common_terms.values(),
            ):
                np.testing.assert_allclose(
                    old.prior.args["sigma"], new.prior.args["sigma"]
                )
            for arg in ("mu", "sigma"):
                np.testing.assert_allclose(
                    base.components["threshold"].prior.args[arg],
                    refit.components["threshold"].prior.args[arg],
                )

    def test_reloo_method_check_does_not_sample(self):
        wrapper = a.OrdinalReLOOWrapper(None, "linear", Path("/tmp"), {})
        with patch.object(
            wrapper, "sample", side_effect=AssertionError("Must not sample")
        ):
            self.assertEqual(
                wrapper.check_implemented_methods(
                    [
                        "sel_observations",
                        "sample",
                        "get_inference_data",
                        "log_likelihood__i",
                    ]
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()
