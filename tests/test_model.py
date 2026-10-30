import unittest
import numpy as np
from src.model import build_classifier

class ModelTests(unittest.TestCase):
    def test_pipeline_learns_simple_separable_data(self):
        X = np.array([[0., 0.], [0., 1.], [10., 10.], [10., 11.]])
        y = np.array(['jazz', 'jazz', 'rock', 'rock'])
        model = build_classifier().fit(X, y)
        self.assertEqual(model.predict([[0., .2]])[0], 'jazz')
        self.assertEqual(model.predict([[10., 10.2]])[0], 'rock')
        self.assertEqual(model.predict_proba([[0., .2]]).shape, (1, 2))

if __name__ == '__main__': unittest.main()
