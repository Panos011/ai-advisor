from pathlib import Path
import unittest


class CloudBuildCostControlTests(unittest.TestCase):
    def test_production_deploy_keeps_scale_to_zero_and_bounded_burst(self):
        cloudbuild = (Path(__file__).parents[1] / "cloudbuild.yaml").read_text()

        self.assertIn("--min-instances=0", cloudbuild)
        self.assertIn("--max-instances=2", cloudbuild)

    def test_experiment_does_not_query_production_by_default(self):
        experiment = (Path(__file__).parents[1] / "experiment.sh").read_text()

        self.assertIn("COMPARE_PRODUCTION=${COMPARE_PRODUCTION:-0}", experiment)
        self.assertIn('if [ "${COMPARE_PRODUCTION}" = "1" ]', experiment)
        self.assertIn("--min-instances 0 --max-instances 2", experiment)


if __name__ == "__main__":
    unittest.main()
