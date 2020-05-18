import unittest
import torch
import torch.nn as nn
import sys
import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PytorchStorage import ForwardModuleStorage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PytorchStorageTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.rand(1,3,224,224).to(device) # random input, this can be an image
        self.y = torch.rand(1,3,224,224).to(device) # random input, this can be an image
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2)
        ).to(device).eval()

    def test(self):
        # self.test_layer_inputs()
        # self.test_layers_input()
        self.test_named_layers()

    def test_layer_inputs(self):
        layer = self.cnn[2]

        storage = ForwardModuleStorage(self.cnn, [layer],debug=True)
        # check if layer is a correct key
        self.assertTrue(layer in storage.state)
        storage(self.x)

        self.assertTrue(type(storage[layer][0]) is dict)


        self.assertTrue(type(storage[layer][0]['input']) is tuple)
        # it must be a tensor
        self.assertTrue(type(storage[layer][0]['input'][0]) is torch.Tensor)
        # it sholud be different because layer = self.cnn[2]
        self.assertFalse(torch.equal(self.x,storage[layer][0]['input'][0]))
        # it must be a tensor
        self.assertTrue(type(storage[layer][0]['output']) is torch.Tensor)

        self.assertTrue(len(storage[layer]), 1)

        storage(self.y)
        # it must be a tensor
        self.assertTrue(type(storage[layer][1]['output']) is torch.Tensor)

        self.assertTrue(len(storage[layer]), 2)


    def test_layers_input(self):
        layer = self.cnn[0]
        layer1 = self.cnn[2]

        storage = ForwardModuleStorage(self.cnn, [layer, layer1])
        # check if layer is a correct key
        self.assertTrue(layer in storage.state)
        self.assertTrue(layer1 in storage.state)
        self.assertEqual(len(list(storage.keys())), 2)

        storage(self.x)
        # only one per layer
        self.assertEqual(len(storage[layer]), 1)
        self.assertEqual(len(storage[layer1]), 1)
        # it must be a tensor
        self.assertTrue(type(storage[layer][0]['output']) is torch.Tensor)
        self.assertTrue(type(storage[layer1][0]['output']) is torch.Tensor)


    def test_named_layers(self):
        layer = self.cnn[0]
        layer1 = self.cnn[2]

        storage = ForwardModuleStorage(self.cnn, { 'a' : [layer], 'b': [layer, layer1] })
        self.assertTrue('a' in storage.state)
        self.assertTrue('b' in storage.state)

        storage(self.x, 'a')

        self.assertTrue(type(storage['a'][layer]['output']) is torch.Tensor)
        self.assertFalse(layer in storage['b'])

        storage(self.y, 'b')
        self.assertTrue(layer in storage['b'])
        self.assertTrue(layer1 in storage['b'])
        self.assertTrue(type(storage['b'][layer]['output']) is torch.Tensor)
        self.assertTrue(type(storage['b'][layer1]['output']) is torch.Tensor)

if __name__ == '__main__':
    unittest.main()