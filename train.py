import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import inception_preprocessing as inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import time

slim = tf.contrib.slim

dataset_dir = '.'

log_dir = './logs'

checkpoint_file = './ckpt/inception_resnet_v2_2016_08_30.ckpt'

image_size = 299

num_classes = 2

labels_to_name = {0: "abnormal", 1: "normal"}


file_pattern = 'X_ray_%s_*.tfrecord'

items_to_descriptions = {
    'image': 'A 3-channel RGB x-ray chest image of a patient.',
    'label': 'A label that is as such -- 0:abnromal/sick, 1:normal/healthy'
}
num_epochs = 30

batch_size = 16

initial_learning_rate = 0.0001
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2


def get_split(split_name, dataset_dir, file_pattern=file_pattern, file_pattern_for_counting='X_ray'):

    if split_name not in ['train', 'validation']:
        raise ValueError(
            'The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(
        dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_name_dict = labels_to_name

    dataset = slim.dataset.Dataset(
        data_sources=file_pattern_path,
        decoder=decoder,
        reader=reader,
        num_readers=4,
        num_samples=num_samples,
        num_classes=num_classes,
        labels_to_name=labels_to_name_dict,
        items_to_descriptions=items_to_descriptions)

    return dataset


def load_batch(dataset, batch_size, height=image_size, width=image_size, is_training=True):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity=24 + 3 * batch_size,
        common_queue_min=24)

    raw_image, label = data_provider.get(['image', 'label'])

    image = inception_preprocessing.preprocess_image(
        raw_image, height, width, is_training)

    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=4 * batch_size,
        allow_smaller_final_batch=True)

    return images, raw_images, labels


def run():
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        dataset = get_split('train', dataset_dir, file_pattern=file_pattern)
        images, _, labels = load_batch(dataset, batch_size=batch_size)

        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(
                images, num_classes=dataset.num_classes, is_training=True)

        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=one_hot_labels, logits=logits)
        total_loss = tf.losses.get_total_loss()

        global_step = get_or_create_global_step()

        lr = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        train_op = slim.learning.create_train_op(total_loss, optimizer)

        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(
            predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)

        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        def train_step(sess, train_op, global_step):
            start_time = time.time()
            total_loss, global_step_count, _ = sess.run(
                [train_op, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            logging.info('global step %s: loss: %.4f (%.2f sec/step)',
                         global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        sv = tf.train.Supervisor(
            logdir=log_dir, summary_op=None, init_fn=restore_fn)

        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * num_epochs):
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step /
                                 num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value, accuracy_value = sess.run(
                        [lr, accuracy])
                    logging.info('Current Learning Rate: %s',
                                 learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s',
                                 accuracy_value)

                    logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                        [logits, probabilities, predictions, labels])
                    print('logits: \n', logits_value)
                    print('Probabilities: \n', probabilities_value)
                    print('predictions: \n', predictions_value)
                    print('Labels:\n:', labels_value)

                if step % 10 == 0:
                    loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                else:
                    loss, _ = train_step(sess, train_op, sv.global_step)

            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)


if __name__ == '__main__':
    run()
