import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import inception_preprocessing as inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import time
import os
from train import get_split, load_batch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
slim = tf.contrib.slim

log_dir = './logs'

log_eval = './log_test_eval'

dataset_dir = '.'

batch_size = 36

num_epochs = 3

checkpoint_file = tf.train.latest_checkpoint(log_dir)


def run():
    if not os.path.exists(log_eval):
        os.mkdir(log_eval)

    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        dataset = get_split('validation', dataset_dir)
        images, raw_images, labels = load_batch(
            dataset, batch_size=batch_size, is_training=False)

        num_batches_per_epoch = dataset.num_samples / batch_size
        num_steps_per_epoch = num_batches_per_epoch

        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2(
                images, num_classes=dataset.num_classes, is_training=False)

        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        predictions = tf.argmax(end_points['Predictions'], 1)
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(
            predictions, labels)
        metrics_op = tf.group(accuracy_update)

        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1)

        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value = sess.run(
                [metrics_op, global_step_op, accuracy])
            time_elapsed = time.time() - start_time

            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)',
                         global_step_count, accuracy_value, time_elapsed)

            return accuracy_value

        tf.summary.scalar('Validation_Accuracy', accuracy)
        my_summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(
            logdir=log_eval, summary_op=None, saver=None, init_fn=restore_fn)

        with sv.managed_session() as sess:
            for step in range(int(num_steps_per_epoch * num_epochs)):
                sess.run(sv.global_step)
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch: %s/%s', step /
                                 num_batches_per_epoch + 1, num_epochs)
                    logging.info('Current Streaming Accuracy: %.4f',
                                 sess.run(accuracy))

                if step % 10 == 0:
                    eval_step(sess, metrics_op=metrics_op,
                              global_step=sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                else:
                    eval_step(sess, metrics_op=metrics_op,
                              global_step=sv.global_step)

            logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))

            raw_images, labels, predictions = sess.run(
                [raw_images, labels, predictions])
            for i in range(10):
                image, label, prediction = raw_images[i], labels[i], predictions[i]
                prediction_name, label_name = dataset.labels_to_name[
                    prediction], dataset.labels_to_name[label]
                text = 'Prediction: %s \n Ground Truth: %s' % (
                    prediction_name, label_name)
                img_plot = plt.imshow(image)

                plt.title(text)
                img_plot.axes.get_yaxis().set_ticks([])
                img_plot.axes.get_xaxis().set_ticks([])
                plt.show()

            logging.info(
                'Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')


if __name__ == '__main__':
    run()
