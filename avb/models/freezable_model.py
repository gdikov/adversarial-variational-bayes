from keras.models import Model


class FreezableModel(Model):
    def __init__(self, inputs, outputs, name_prefix, deep_freeze=True):
        super(FreezableModel, self).__init__(inputs=inputs, outputs=outputs)
        if not deep_freeze:
            self._trainable_layers = [layer for layer in self.layers if layer.trainable]
        else:
            def recursive_model_crawl(layer):
                deeper_layers = []
                if isinstance(layer, Model):
                    for l in layer.layers:
                        if l.trainable:
                            deeper_layers += recursive_model_crawl(l)
                if layer.trainable and (layer.name.split('_')[0] in name_prefix):
                    deeper_layers.append(layer)
                return deeper_layers

            self._trainable_layers = sum([recursive_model_crawl(layer) for layer in self.layers if layer.trainable], [])
            print [l.name for l in self._trainable_layers]

    def freeze(self):
        for layer in self._trainable_layers:
            layer.trainable = False

    def unfreeze(self):
        for layer in self._trainable_layers:
            layer.trainable = True
