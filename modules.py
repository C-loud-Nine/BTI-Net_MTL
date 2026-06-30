import tensorflow as tf
from tensorflow.keras import layers


class MultiScaleContextFusion(layers.Layer):
    """Hierarchical Multi-Scale Fusion with dynamic scale weighting."""
    def __init__(self, channels, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.dilation_convs = [
            layers.SeparableConv2D(self.channels, 3, padding='same', dilation_rate=1, name=f'{self.name}_sep_conv_1'),
            layers.SeparableConv2D(self.channels, 3, padding='same', dilation_rate=2, name=f'{self.name}_sep_conv_2'),
            layers.SeparableConv2D(self.channels, 3, padding='same', dilation_rate=4, name=f'{self.name}_sep_conv_4'),
        ]
        self.scale_attention = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.channels // self.reduction_ratio, activation='relu'),
            layers.Dense(3, activation='softmax'),
        ], name=f'{self.name}_scale_attention')
        self.cross_scale_fusion = layers.Conv2D(self.channels, 1, padding='same', name=f'{self.name}_fusion')
        super().build(input_shape)

    def call(self, inputs):
        scale_features = [conv(inputs) for conv in self.dilation_convs]
        scale_weights = self.scale_attention(inputs)
        scale_weights = tf.reshape(scale_weights, [-1, 1, 1, 3])
        weighted = [scale_features[i] * scale_weights[:, :, :, i:i+1] for i in range(3)]
        return self.cross_scale_fusion(tf.add_n(weighted))

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels, 'reduction_ratio': self.reduction_ratio})
        return config


class DualPathAttention(layers.Layer):
    """Dual-Path Attention: Channel + Spatial."""
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        self.channel_path = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(max(8, self.channels // 16), activation='relu'),
            layers.Dense(self.channels, activation='sigmoid'),
        ], name=f'{self.name}_channel_path')
        self.spatial_path = layers.Conv2D(1, 7, padding='same', activation='sigmoid', name=f'{self.name}_spatial') 
        super().build(input_shape)

    def call(self, inputs):
        channel_weights = tf.reshape(self.channel_path(inputs), [-1, 1, 1, self.channels])
        spatial_input = tf.concat([
            tf.reduce_max(inputs, axis=3, keepdims=True),
            tf.reduce_mean(inputs, axis=3, keepdims=True),
        ], axis=3)
        spatial_weights = self.spatial_path(spatial_input)
        return inputs * channel_weights * spatial_weights

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels})
        return config


class ResidualBlock(layers.Layer):
    """Residual block with multi-scale fusion and optional dual-path attention."""
    def __init__(self, channels, use_attention=True, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.use_attention = use_attention

    def build(self, input_shape):
        self.conv1 = layers.SeparableConv2D(self.channels, 3, padding='same', activation='relu', name=f'{self.name}_conv1')
        self.bn1   = layers.BatchNormalization(name=f'{self.name}_bn1')
        self.conv2 = layers.SeparableConv2D(self.channels, 3, padding='same', name=f'{self.name}_conv2')
        self.bn2   = layers.BatchNormalization(name=f'{self.name}_bn2')
        self.multiscale   = MultiScaleContextFusion(self.channels, name=f'{self.name}_multiscale')
        if self.use_attention:
            self.attention = DualPathAttention(self.channels, name=f'{self.name}_attention')
        self.residual_proj = layers.Conv2D(self.channels, 1, padding='same', name=f'{self.name}_proj')
        super().build(input_shape)

    def call(self, inputs):
        residual = inputs
        x = self.bn1(self.conv1(inputs))
        x = self.bn2(self.conv2(x))
        x = self.multiscale(x)
        if self.use_attention:
            x = self.attention(x)
        if residual.shape[-1] != self.channels:
            residual = self.residual_proj(residual)
        return tf.keras.activations.relu(x + residual)

    def get_config(self):
        config = super().get_config()
        config.update({'channels': self.channels, 'use_attention': self.use_attention})
        return config


class AttentionGate(layers.Layer):
    """Attention gate with multi-scale context for skip connections."""
    def __init__(self, inter_channels, **kwargs):
        super().__init__(**kwargs)
        self.inter_channels = inter_channels

    def build(self, input_shape):
        self.x_transform = layers.Conv2D(self.inter_channels, 1, padding='same', name=f'{self.name}_x_transform')
        self.g_transform = layers.Conv2D(self.inter_channels, 1, padding='same', name=f'{self.name}_g_transform')
        self.context_attention = tf.keras.Sequential([
            layers.SeparableConv2D(self.inter_channels, 3, padding='same', dilation_rate=1),
            layers.SeparableConv2D(self.inter_channels, 3, padding='same', dilation_rate=2),
            layers.Conv2D(1, 1, padding='same', activation='sigmoid'),
        ], name=f'{self.name}_context_attention')
        super().build(input_shape)

    def call(self, x, g):
        theta_x = self.x_transform(x)
        phi_g   = self.g_transform(g)
        if x.shape[1] != g.shape[1] or x.shape[2] != g.shape[2]:
            phi_g = tf.image.resize(phi_g, [x.shape[1], x.shape[2]])
        combined = tf.nn.relu(theta_x + phi_g)
        return x * self.context_attention(combined)

    def get_config(self):
        config = super().get_config()
        config.update({'inter_channels': self.inter_channels})
        return config


class TaskInteractionModule(layers.Layer):
    """Bidirectional cross-task interaction: Seg↔Clf with multiplicative modulation."""
    def __init__(self, seg_channels=192, clf_channels=256, **kwargs):
        super().__init__(**kwargs)
        self.seg_channels = seg_channels
        self.clf_channels = clf_channels

    def build(self, input_shape):
        self.seg_to_clf = tf.keras.Sequential([
            layers.Conv2D(self.seg_channels // 2, 1, activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.clf_channels, activation='relu'),
            layers.BatchNormalization(),
        ], name=f'{self.name}_seg_to_clf')
        self.clf_to_seg = tf.keras.Sequential([
            layers.Dense(self.seg_channels, activation='sigmoid'),
            layers.Reshape((1, 1, self.seg_channels)),
        ], name=f'{self.name}_clf_to_seg')
        self.seg_gate = layers.Dense(self.seg_channels, activation='sigmoid', name=f'{self.name}_seg_gate')
        self.clf_gate = layers.Dense(self.clf_channels, activation='sigmoid', name=f'{self.name}_clf_gate')
        self.seg_gap  = layers.GlobalAveragePooling2D()
        super().build(input_shape)

    def call(self, seg_features, clf_features, training=None):
        # Seg → Clf
        seg_context      = self.seg_to_clf(seg_features)
        clf_gate_weights = self.clf_gate(seg_context)
        enhanced_clf     = clf_features + clf_gate_weights * seg_context

        # Clf → Seg
        clf_context      = self.clf_to_seg(clf_features)
        seg_global       = tf.reshape(self.seg_gate(self.seg_gap(seg_features)), [-1, 1, 1, self.seg_channels])
        modulation       = 1.0 + 0.7 * seg_global * clf_context
        enhanced_seg     = seg_features * modulation

        return enhanced_seg, enhanced_clf

    def get_config(self):
        config = super().get_config()
        config.update({'seg_channels': self.seg_channels, 'clf_channels': self.clf_channels})
        return config


class UncertaintyProxyAttention(layers.Layer):
    """Uncertainty Proxy Attention (UPA): per-instance, per-level gating between base and TIM-enhanced features using 3 signals (alignment, seg energy, clf energy)."""
    def __init__(self, seg_channels, clf_channels, **kwargs):
        super().__init__(**kwargs)
        self.seg_channels = seg_channels
        self.clf_channels = clf_channels
        self._last_w_seg  = None
        self._last_w_clf  = None

    def build(self, input_shape):
        self.seg_proj = layers.Dense(64, use_bias=False, name=f'{self.name}_sp')
        self.clf_proj = layers.Dense(64, use_bias=False, name=f'{self.name}_cp')
        self.weight_network = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(2),
        ], name=f'{self.name}_wn')
        super().build(input_shape)

    def _alignment(self, seg_base, seg_enhanced, clf_base, clf_enhanced):
        seg_res = tf.reduce_mean(seg_enhanced - seg_base, axis=[1, 2])
        clf_res = clf_enhanced - clf_base
        seg_p   = self.seg_proj(seg_res)
        clf_p   = self.clf_proj(clf_res)
        return tf.reduce_sum(
            tf.nn.l2_normalize(seg_p, axis=1) *
            tf.nn.l2_normalize(clf_p, axis=1),
            axis=1, keepdims=True)   # [B, 1]

    def _seg_energy(self, seg_base):
        H = seg_base.shape[1]
        if H is not None and H < 8:
            return tf.zeros_like(
                tf.reduce_mean(seg_base, axis=[1, 2, 3])[..., :1])
        dy = tf.pad(seg_base[:, 1:] - seg_base[:, :-1],
                    [[0, 0], [0, 1], [0, 0], [0, 0]])
        dx = tf.pad(seg_base[:, :, 1:] - seg_base[:, :, :-1],
                    [[0, 0], [0, 0], [0, 1], [0, 0]])
        energy = (tf.reduce_mean(tf.abs(dy), axis=[1, 2, 3], keepdims=True) +
                  tf.reduce_mean(tf.abs(dx), axis=[1, 2, 3], keepdims=True))
        return tf.reshape(energy, [-1, 1])   # [B, 1]

    def _clf_energy(self, clf_enhanced):
        mean     = tf.reduce_mean(clf_enhanced, axis=-1, keepdims=True)
        variance = tf.reduce_mean(
            tf.square(clf_enhanced - mean), axis=-1, keepdims=True)
        return tf.math.log1p(variance)   # [B, 1]

    def call(self, seg_base, seg_enhanced, clf_base, clf_enhanced, training=None):
        align      = self._alignment(seg_base, seg_enhanced, clf_base, clf_enhanced)
        seg_energy = self._seg_energy(seg_base)
        clf_energy = self._clf_energy(clf_enhanced)
        gate_input = tf.concat([align, seg_energy, clf_energy], axis=-1)  # [B, 3]
        logits     = self.weight_network(gate_input)
        weights    = tf.sigmoid(logits)
        self._last_w_seg = weights[:, 0]
        self._last_w_clf = weights[:, 1]
        w_seg = tf.reshape(weights[:, 0:1], [-1, 1, 1, 1])
        w_clf = tf.reshape(weights[:, 1:2], [-1, 1])
        seg_final = seg_base + w_seg * (seg_enhanced - seg_base)
        clf_final = clf_base + w_clf * (clf_enhanced - clf_base)
        return seg_final, clf_final

    def get_config(self):
        config = super().get_config()
        config.update({
            'seg_channels': self.seg_channels,
            'clf_channels': self.clf_channels,
        })
        return config