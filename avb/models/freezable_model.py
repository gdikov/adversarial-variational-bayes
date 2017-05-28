from keras.models import Model

from utils.logger_config import logger


class FreezableModel(Model):
    """
    A Keras model which supports layer freezing, i.e. making them untrainable, before compilation. 
    This is useful in alternating model optimisation, when it consists of multiple sub-models. 
    """
    def __init__(self, inputs, outputs, name_prefix, deep_freeze=True):
        """
        Args:
            inputs: the inputs of the model 
            outputs: the outputs of the model
            name_prefix: str, a prefix of the layer/sub-model names which are to be frozen before compilation.
            deep_freeze: bool, whether the layers of the model and its sub-models should be recursively frozen or not.
        """
        super(FreezableModel, self).__init__(inputs=inputs, outputs=outputs)
        if not deep_freeze:
            self._trainable_layers = [layer for layer in self.layers if layer.trainable]
        else:
            def recursive_model_crawl(current_layer):
                deeper_layers = []
                if isinstance(current_layer, Model):
                    for l in current_layer.layers:
                        if l.trainable:
                            deeper_layers += recursive_model_crawl(l)
                if current_layer.trainable and (current_layer.name.split('_')[0] in name_prefix):
                    deeper_layers.append(current_layer)
                return deeper_layers
            # flatten the list of lists of layer names
            self._trainable_layers = sum([recursive_model_crawl(layer)
                                          for layer in self.layers if layer.trainable], [])
        logger.debug("Model {} has freezable layers: {}".format(self.name, [l.name for l in self._trainable_layers]))

    def freeze(self):
        """
        Make all the layers trainable.
        
        Returns:
            In-place method.
        """
        for layer in self._trainable_layers:
            layer.trainable = False

    def unfreeze(self):
        """
        Make all the layers not trainable.
        
        Returns:
            In-place method.
        """
        for layer in self._trainable_layers:
            layer.trainable = True
