import unittest


class TestServiceContract(unittest.TestCase):
    def test_service_routes_exist(self):
        try:
            from drug_agent.gad import service
        except RuntimeError:
            self.skipTest("fastapi is only available on the GPU worker")
        paths = {route.path for route in service.app.routes}
        self.assertTrue({"/health", "/metrics", "/score-and-update", "/checkpoint"}.issubset(paths))


if __name__ == "__main__":
    unittest.main()
