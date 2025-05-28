# This is  trivial test to verify GitHub Actions Workflow Works
# As development progresses, developers need to use pytest.

from django.test import TestCase

class BaselineTest(TestCase):

    def test_ci_pipeline_runs(self):
        self.assertTrue(True)

