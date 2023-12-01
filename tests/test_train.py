import unittest
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import sys
sys.path.append("/Users/paulchauvin/Documents/GitHub/lunit/src/")

from train import train_model
from models import SimpleConvNet
from config_loader import load_config

class TestTrainingScript(unittest.TestCase):
    def setUp(self):
        self.config_path = "config/initial_experiment.yaml"
        self.config = load_config(self.config_path)
        self.sample_data = torch.randn((self.config['batch_size'], 1, 28, 28))
        self.sample_label = torch.randint(0, 10, (self.config['batch_size'],))

    def test_load_config(self):
        self.assertIsInstance(self.config, dict)
        self.assertIn('batch_size', self.config)
        self.assertIn('lr', self.config)
        self.assertIn('max_epochs', self.config)

    def test_model_initializaton(self):
        model = SimpleConvNet()
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'forward'))

    def test_loss_calculation(self):
        model = SimpleConvNet()
        criterion = torch.nn.CrossEntropyLoss()
        output = model(self.sample_data)
        loss = criterion(output, self.sample_label)
        self.assertIsInstance(loss.item(), float)

    def test_metrics_calculation(self):
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 1, 1]

        # Calculate Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

        # Calculate F1 Score
        f1 = f1_score(y_true, y_pred, average='macro')
        self.assertGreaterEqual(f1, 0)
        self.assertLessEqual(f1, 1)

        # Calculate Precision
        precision = precision_score(y_true, y_pred, average='macro')
        self.assertGreaterEqual(precision, 0)
        self.assertLessEqual(precision, 1)

        # Calculate Recall
        recall = recall_score(y_true, y_pred, average='macro')
        self.assertGreaterEqual(recall, 0)
        self.assertLessEqual(recall, 1)

    def test_invalid_config_path(self):
        with self.assertRaises(Exception):
            train_model("invalid_path.yaml")

if __name__ == '__main__':
    unittest.main()
