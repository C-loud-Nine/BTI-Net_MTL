"""
complexity.py - parameter counts and FLOPs at 224 x 224 for one image.

Usage:
    python complexity.py --num-classes 3   # 7 for HAM10000, 4 for BRISC
FLOPs are counted by the TensorFlow profiler on the frozen graph (one
multiply-add = 2 FLOPs). The SRG share is reported separately.
"""
import argparse
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

import config
from model import enhanced_bti_model


def flops(model):
    spec = tf.TensorSpec([1, *config.INPUT_SIZE], tf.float32)
    fn = tf.function(lambda x: model(x, training=False)).get_concrete_function(spec)
    frozen = convert_variables_to_constants_v2(fn)
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    info = tf.compat.v1.profiler.profile(graph=frozen.graph, run_meta=tf.compat.v1.RunMetadata(),
                                         cmd="op", options=opts)
    return info.total_float_ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-classes", type=int, default=3)
    args = ap.parse_args()
    model = enhanced_bti_model(input_size=config.INPUT_SIZE, num_clf_classes=args.num_classes)
    total = model.count_params()
    srg = sum(l.count_params() for l in model.layers if "srg" in l.name.lower())
    print(f"parameters: total {total:,} | SRG {srg:,}")
    print(f"FLOPs per 224x224 image: {flops(model) / 1e9:.2f} G")


if __name__ == "__main__":
    main()
