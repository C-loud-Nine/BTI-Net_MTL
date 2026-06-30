# ============================================================================
# MODEL BUILDING FUNCTION (modular)
# ============================================================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4

from modules import (
    ResidualBlock,
    AttentionGate,
    TaskInteractionModule,
    UncertaintyProxyAttention,
)


def build_decoder_block(prev, skip, channels, dropout_rate, block_name):
    """Upsample -> attention gate -> concat -> residual block -> dropout."""
    x = layers.Conv2DTranspose(channels, 3, strides=2, padding='same')(prev)
    att = AttentionGate(channels // 2, name=f'att_gate_{block_name}')(skip, x)
    x = layers.Concatenate()([x, att])
    x = ResidualBlock(channels, use_attention=False, name=f'bti_res_{block_name}')(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_tim_upa(decoder_out, encoder_feat, seg_channels, level_name):
    """Attach TIM + UPA to a decoder level. Returns (seg_final, clf_final)."""
    clf_raw = layers.GlobalAveragePooling2D()(encoder_feat)
    clf_raw = layers.Dense(256, activation='relu', name=f'clf_{level_name}_features')(clf_raw)

    tim = TaskInteractionModule(seg_channels=seg_channels, clf_channels=256,
                                 name=f'task_interaction_{level_name}')
    seg_enh, clf_enh = tim(decoder_out, clf_raw)

    upa = UncertaintyProxyAttention(seg_channels=seg_channels, clf_channels=256,
                                     name=f'upa_{level_name}')
    seg_final, clf_final = upa(decoder_out, seg_enh, clf_raw, clf_enh)

    return seg_final, clf_final


def enhanced_bti_model(
    input_size=(224, 224, 3),
    num_seg_classes=1,
    num_clf_classes=3,
    dropout_rate=0.3,
    l2_lambda=1e-5,
):
    """BTI-Net: TIM + UPA applied at all four decoder levels (D1-D4)."""

    # ------------------------------------------------------------------
    # ENCODER (EfficientNetB4, ImageNet weights)
    # ------------------------------------------------------------------
    base_model = EfficientNetB4(
        input_shape=input_size, include_top=False, weights="imagenet"
    )
    for layer in base_model.layers[:50]:
        layer.trainable = False

    s1 = base_model.get_layer("block2a_expand_activation").output  # 56x56
    s2 = base_model.get_layer("block3a_expand_activation").output  # 28x28
    s3 = base_model.get_layer("block4a_expand_activation").output  # 14x14
    s4 = base_model.get_layer("block6a_expand_activation").output  # 7x7
    bridge = base_model.get_layer("top_activation").output         # 7x7

    s1_e = ResidualBlock(144,  use_attention=True, name='bti_res_s1')(s1)
    s2_e = ResidualBlock(192,  use_attention=True, name='bti_res_s2')(s2)
    s3_e = ResidualBlock(336,  use_attention=True, name='bti_res_s3')(s3)
    s4_e = ResidualBlock(960,  use_attention=True, name='bti_res_s4')(s4)
    br_e = ResidualBlock(1792, use_attention=True, name='bti_res_bridge')(bridge)

    # ------------------------------------------------------------------
    # DECODER (4 levels, each followed by TIM + UPA)
    # ------------------------------------------------------------------
    decoder_channels = [384, 192, 96, 48]

    # D1 -- 7x7 -> 14x14
    d1 = build_decoder_block(br_e, s4_e, decoder_channels[0], dropout_rate, 'd1')
    d1_final, clf_d1 = build_tim_upa(d1, br_e, decoder_channels[0], 'd1')

    # D2 -- 14x14 -> 28x28
    d2 = build_decoder_block(d1_final, s3_e, decoder_channels[1], dropout_rate, 'd2')
    d2_final, clf_d2 = build_tim_upa(d2, s4_e, decoder_channels[1], 'd2')

    # D3 -- 28x28 -> 56x56
    d3 = build_decoder_block(d2_final, s2_e, decoder_channels[2], dropout_rate, 'd3')
    d3_final, clf_d3 = build_tim_upa(d3, s3_e, decoder_channels[2], 'd3')

    # D4 -- 56x56 -> 112x112
    d4 = build_decoder_block(d3_final, s1_e, decoder_channels[3], dropout_rate, 'd4')
    d4_final, clf_d4 = build_tim_upa(d4, s2_e, decoder_channels[3], 'd4')

    # ------------------------------------------------------------------
    # SEGMENTATION HEAD (112x112 -> 224x224)
    # ------------------------------------------------------------------
    seg = layers.Conv2DTranspose(24, 3, strides=2, padding='same')(d4_final)
    seg = layers.Conv2D(16, 3, padding='same', activation='relu')(seg)
    seg_output = layers.Conv2D(num_seg_classes, 1, activation='sigmoid',
                                name='segmentation_output')(seg)

    # ------------------------------------------------------------------
    # CLASSIFICATION HEAD
    # ------------------------------------------------------------------
    gap_s2 = layers.GlobalAveragePooling2D()(s2_e)
    gap_s3 = layers.GlobalAveragePooling2D()(s3_e)
    gap_s4 = layers.GlobalAveragePooling2D()(s4_e)
    gap_br = layers.GlobalAveragePooling2D()(br_e)

    multi_level = layers.Concatenate()([gap_s2, gap_s3, gap_s4, gap_br,
                                         clf_d1, clf_d2, clf_d3, clf_d4])
    clf = layers.Dense(256, activation='relu')(multi_level)
    clf = layers.BatchNormalization()(clf)
    clf = layers.Dropout(0.3)(clf)
    clf_output = layers.Dense(num_clf_classes, activation='softmax',
                               name='classification_output')(clf)

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------
    model = models.Model(
        inputs=base_model.input,
        outputs=[seg_output, clf_output],
        name='BTI_Net',
    )
    return model