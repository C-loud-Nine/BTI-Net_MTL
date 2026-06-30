import tensorflow as tf


# ============================================================================
# GEOMETRIC FILTERS
# ============================================================================

class GeometricFilters:
    @staticmethod
    def curvature_filter(kernel_size=5, eps=1e-8):
        k   = int(kernel_size)
        x   = tf.cast(tf.range(k) - (k // 2), tf.float32)
        y   = tf.cast(tf.range(k) - (k // 2), tf.float32)
        xx, yy = tf.meshgrid(x, y)
        r2  = xx * xx + yy * yy
        sigma   = tf.cast(k / 4.0, tf.float32)
        gaussian = tf.exp(-r2 / (2.0 * (sigma * sigma + eps)))
        f   = (xx * xx - yy * yy) * gaussian
        f   = f / (tf.reduce_sum(tf.abs(f)) + eps)
        return tf.reshape(f, [k, k, 1, 1])


# ============================================================================
# SEGMENTATION LOSS
# ============================================================================

def stabilized_tversky_index(y_true, y_pred, alpha=0.5, beta=0.5, smooth=1e-7):
    """Per-image Tversky index. Returns shape [B]."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
    if y_pred.shape.rank == 3:
        y_pred = tf.expand_dims(y_pred, -1)

    axes = [1, 2, 3]
    tp = tf.reduce_sum(y_true * y_pred,            axis=axes)
    fp = tf.reduce_sum((1 - y_true) * y_pred,      axis=axes)
    fn = tf.reduce_sum(y_true * (1 - y_pred),      axis=axes)
    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def stabilized_focal_tversky(y_true, y_pred, gamma=0.75):
    """Focal Tversky loss (scalar)."""
    eps   = 1e-7
    t_idx = stabilized_tversky_index(y_true, y_pred, smooth=eps)
    focal = tf.pow(tf.maximum(1.0 - t_idx, eps), gamma)
    focal = tf.where(tf.math.is_finite(focal), focal, tf.ones_like(focal))
    return tf.reduce_mean(focal)


def efficient_boundary_detection(y_true, y_pred, kernel_size=5):
    """Curvature-based boundary loss (scalar)."""
    eps    = 1e-8
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
    if y_pred.shape.rank == 3:
        y_pred = tf.expand_dims(y_pred, -1)

    y_true_bin = tf.clip_by_value(tf.cast(y_true > 0.5, tf.float32), 0.0, 1.0)
    y_pred_bin = tf.clip_by_value(tf.cast(y_pred > 0.5, tf.float32), 0.0, 1.0)

    kernel = GeometricFilters.curvature_filter(kernel_size)
    b_true = tf.nn.conv2d(y_true_bin, kernel, strides=1, padding='SAME')
    b_pred = tf.nn.conv2d(y_pred_bin, kernel, strides=1, padding='SAME')

    per_sample = tf.clip_by_value(
        tf.reduce_mean(tf.abs(b_true - b_pred), axis=[1, 2, 3]), 0.0, 1.0
    )
    per_sample = tf.where(tf.math.is_finite(per_sample), per_sample, tf.zeros_like(per_sample))
    return tf.reduce_mean(per_sample)


def efficient_texture_consistency(y_true, y_pred):
    """Sobel gradient std consistency loss (scalar)."""
    eps    = 1e-8
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
    if y_pred.shape.rank == 3:
        y_pred = tf.expand_dims(y_pred, -1)

    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])

    gt = tf.nn.conv2d(y_true, sobel_x, strides=1, padding='SAME')
    gp = tf.nn.conv2d(y_pred, sobel_x, strides=1, padding='SAME')

    axes = [1, 2, 3]
    std_gt = tf.sqrt(tf.maximum(tf.reduce_mean(tf.square(gt - tf.reduce_mean(gt, axis=axes, keepdims=True)), axis=axes), 0.0) + eps)
    std_gp = tf.sqrt(tf.maximum(tf.reduce_mean(tf.square(gp - tf.reduce_mean(gp, axis=axes, keepdims=True)), axis=axes), 0.0) + eps)

    per_sample = tf.abs(std_gt - std_gp)
    per_sample = tf.where(tf.math.is_finite(per_sample), per_sample, tf.zeros_like(per_sample))
    return tf.reduce_mean(per_sample)


def enhanced_lesion_focus_loss(y_true, y_pred):
    """
    Combined segmentation loss:
        L_seg = L_FocalTversky + 0.25 * L_boundary + 0.15 * L_texture
    with per-sample lesion-size boost.
    """
    eps    = 1e-7
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
    if y_pred.shape.rank == 3:
        y_pred = tf.expand_dims(y_pred, -1)

    y_true = tf.clip_by_value(y_true, 0.0, 1.0)
    y_pred = tf.clip_by_value(
        tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred)), eps, 1.0 - eps
    )

    # Per-sample focal Tversky
    t_idx   = stabilized_tversky_index(y_true, y_pred, smooth=eps)
    base    = tf.pow(tf.maximum(1.0 - t_idx, eps), 0.75)

    # Boundary + texture (scalars broadcast to [B])
    b_loss = efficient_boundary_detection(y_true, y_pred)
    t_loss = efficient_texture_consistency(y_true, y_pred)
    b_vec  = tf.fill(tf.shape(base), tf.cast(b_loss, tf.float32))
    t_vec  = tf.fill(tf.shape(base), tf.cast(t_loss, tf.float32))

    # Lesion-size boost: small lesions get higher weight
    axes         = [1, 2, 3]
    total_pixels = tf.cast(tf.shape(y_true)[1] * tf.shape(y_true)[2] * tf.shape(y_true)[3], tf.float32)
    lesion_ratio = tf.clip_by_value(tf.reduce_sum(y_true, axis=axes) / tf.maximum(total_pixels, 1.0), 0.0, 1.0)
    size_boost   = 1.0 + tf.clip_by_value(tf.exp(-10.0 * lesion_ratio), 0.0, 2.0)

    loss = tf.clip_by_value(base * size_boost + 0.25 * b_vec + 0.15 * t_vec, 0.0, 10.0)
    loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
    return tf.reduce_mean(loss)


# ============================================================================
# CLASSIFICATION LOSS
# ============================================================================

def enhanced_multi_modal_focal_loss(y_true_clf, y_pred_clf):
    """Focal cross-entropy for multi-class classification (scalar)."""
    eps    = 1e-7
    y_pred = tf.cast(y_pred_clf, tf.float32)
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.fill(tf.shape(y_pred), eps))
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    # Re-normalise
    denom  = tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    y_pred = y_pred / tf.where(denom > 0.0, denom, tf.ones_like(denom))
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    y_true = tf.cast(tf.reshape(y_true_clf, [-1]), tf.int32)
    ce     = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    ce     = tf.where(tf.math.is_finite(ce), ce, tf.fill(tf.shape(ce), 1e3))

    pt     = tf.exp(-ce)
    focal  = tf.pow(tf.maximum(1.0 - pt, eps), 2.0) * ce
    return tf.reduce_mean(tf.clip_by_value(focal, 0.0, 10.0))


# ============================================================================
# SOFT IoU HELPER
# ============================================================================

def soft_iou_per_sample(y_true, y_pred):
    """Differentiable soft IoU, no threshold. Returns [B]."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # FIX: Added rank check to prevent axis 3 out of bounds error on 3D tensors
    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
    if y_pred.shape.rank == 3:
        y_pred = tf.expand_dims(y_pred, -1)
        
    axes   = [1, 2, 3]
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union        = tf.reduce_sum(
        y_true + y_pred - y_true * y_pred, axis=axes)
    return intersection / (union + 1e-7)


# ============================================================================
# UPA GATE LOSSES
# ============================================================================

def make_seg_loss(upa_layers, lambda_gate=0.01):
    def loss(y_true, y_pred):
        task     = enhanced_lesion_focus_loss(y_true, y_pred)
        soft_iou = soft_iou_per_sample(y_true, tf.stop_gradient(y_pred))

        # Rank-normalise within the batch: hard=1.0, easy=0.0
        iou_min  = tf.reduce_min(soft_iou)
        iou_max  = tf.reduce_max(soft_iou)
        iou_norm = (soft_iou - iou_min) / (iou_max - iou_min + 1e-7)
        target   = 1.0 - iou_norm   # hard samples → target close to 1.0
        target   = tf.clip_by_value(target, 1e-6, 1.0 - 1e-6)

        gate = 0.0; n = 0
        for upa in upa_layers:
            if upa._last_w_seg is not None:
                w = tf.clip_by_value(upa._last_w_seg, 1e-6, 1.0 - 1e-6)
                gate += tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        target[:, tf.newaxis], w[:, tf.newaxis]))
                n += 1
        if n > 0: gate /= n
        return task + lambda_gate * gate
    return loss


def make_clf_loss(upa_layers, lambda_gate=0.01):
    def loss(y_true, y_pred):
        task       = enhanced_multi_modal_focal_loss(y_true, y_pred)
        confidence = tf.reduce_max(tf.stop_gradient(y_pred), axis=-1)

        # Rank-normalise: confident=1.0, uncertain=0.0 within batch
        conf_min  = tf.reduce_min(confidence)
        conf_max  = tf.reduce_max(confidence)
        conf_norm = (confidence - conf_min) / (conf_max - conf_min + 1e-7)
        target    = 1.0 - conf_norm  # NOW hard/uncertain samples → 1.0
        target    = tf.clip_by_value(target, 1e-6, 1.0 - 1e-6)

        gate = 0.0; n = 0
        for upa in upa_layers:
            if upa._last_w_clf is not None:
                w = tf.clip_by_value(upa._last_w_clf, 1e-6, 1.0 - 1e-6)
                gate += tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        target[:, tf.newaxis], w[:, tf.newaxis]))
                n += 1
        if n > 0: gate /= n
        return task + lambda_gate * gate
    return loss