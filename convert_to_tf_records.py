import random
import tensorflow as tf
from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_to_tf_records


def main():

    if not FLAGS.tfrecord_filename:
        raise ValueError(
            'tfrecord_filename is empty. Please state a tfrecord_filename argument.')

    if _dataset_exists(dataset_file=FLAGS.dataset_file, _NUM_SHARDS=FLAGS.num_shards, output_filename=FLAGS.tfrecord_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return None
    photo_filenames, class_names = _get_filenames_and_classes(
        FLAGS.dataset_file)

    class_ids = [1 if label == "normal" else 0 for label in class_names]

    num_validation = int(FLAGS.validation_size * len(photo_filenames))

    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]
    training_labels = class_ids[num_validation:]
    validation_labels = class_ids[:num_validation]
    _convert_to_tf_records('train', training_filenames, training_labels,
                           dataset_file=FLAGS.dataset_file,
                           tfrecord_filename=FLAGS.tfrecord_filename,
                           _NUM_SHARDS=FLAGS.num_shards)
    _convert_to_tf_records('validation', validation_filenames, validation_labels,
                           dataset_file=FLAGS.dataset_file,
                           tfrecord_filename=FLAGS.tfrecord_filename,
                           _NUM_SHARDS=FLAGS.num_shards)

    print('\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename))


flags = tf.app.flags

flags.DEFINE_string('dataset_file', "images_data.txt",
                    'String: Your dataset txt file')

flags.DEFINE_integer(
    'num_shards', 2, 'Int: Number of shards to split the TFRecord files')

flags.DEFINE_float('validation_size', 0.3,
                   'Float: The proportion of examples in the dataset to be used for validation')
flags.DEFINE_integer(
    'random_seed', 0, 'Int: Random seed to use for repeatability.')

flags.DEFINE_string('tfrecord_filename', "X_ray",
                    'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS
if __name__ == "__main__":
    main()
