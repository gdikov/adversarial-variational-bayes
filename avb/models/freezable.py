from __future__ import absolute_import

import logging

from keras.models import Model

logger = logging.getLogger(__name__)


class FreezableModel(Model):
    """
    A Keras model which supports layer freezing, i.e. making them untrainable, before compilation. 
    This is useful in alternating model optimisation, when it consists of multiple sub-models. 
    """
    def __init__(self, inputs, outputs, name='freezable'):
        """
        Args:
            inputs: the inputs of the model 
            outputs: the outputs of the model
            name: str, the name of the model
        """
        super(FreezableModel, self).__init__(inputs=inputs, outputs=outputs, name=name)
        self._trainable_layers = []

    def _crawl_trainable_layers(self, freezable_layers_prefix, deep_freeze=True):
        """
        (Recursively) crawl the model and find the trainable layers which can be frozen.

        Args:
            freezable_layers_prefix: see docstring of public method
            deep_freeze: see docstring of public method

        Returns:
            A list of all trainable layers in the model.
        """
        if not deep_freeze:
            trainable_layers = [layer for layer in self.layers if layer.trainable]
        else:
            def recursive_model_crawl(current_layer):
                deeper_layers = []
                if isinstance(current_layer, Model):
                    for l in current_layer.layers:
                        if l.trainable:
                            deeper_layers += recursive_model_crawl(l)
                if current_layer.trainable and (current_layer.name.split('_')[0] in freezable_layers_prefix):
                    deeper_layers.append(current_layer)
                return deeper_layers

            # flatten the list of lists of layer names
            trainable_layers = sum([recursive_model_crawl(layer) for layer in self.layers if layer.trainable], [])
        logger.debug("Model {} has freezable layers: {}".format(self.name, [l.name for l in trainable_layers]))
        return trainable_layers

    def get_trainable_layers(self, freezable_layers_prefix=None, deep_crawl=True):
        """
        Return the trainable layers if already crawled, else find, save them globally and return them.

        Args:
            freezable_layers_prefix: str, a prefix of the layer/sub-model names which will be frozen before compilation.
            deep_crawl: bool, whether the layers of the model and its sub-models should be recursively crawled or not.

        Returns:
            A list of all trainable layers in the model.
        """
        if not hasattr(self, '_trainable_layers'):
            trainable_layers = self._crawl_trainable_layers(freezable_layers_prefix, deep_crawl)
            setattr(self, '_trainable_layers', trainable_layers)
            return self._trainable_layers
        else:
            return self._trainable_layers

    def freeze(self, freezable_layers_prefix=None, deep_freeze=True):
        """
        Make all the layers trainable.

        Args:
            freezable_layers_prefix: str, a prefix of the layer/sub-model names which will be frozen before compilation.
            deep_freeze: bool, whether the layers of the model and its sub-models should be recursively frozen or not.

        Returns:
            In-place method.
        """
        trainable_layers = self.get_trainable_layers(freezable_layers_prefix, deep_freeze)
        for layer in trainable_layers:
            layer.trainable = False

    def unfreeze(self, unfreezable_layers_prefix=None, deep_unfreeze=True):
        """
        Make all the layers not trainable.

        Args:
            unfreezable_layers_prefix: str, a prefix of the layer/sub-model names which will be frozen before compilation.
            deep_unfreeze: bool, whether the layers of the model and its sub-models should be recursively frozen or not.
        
        Returns:
            In-place method.
        """
        trainable_layers = self.get_trainable_layers(unfreezable_layers_prefix, deep_unfreeze)
        for layer in trainable_layers:
            layer.trainable = True
