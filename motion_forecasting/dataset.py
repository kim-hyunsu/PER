from emlp.reps import Scalar, Vector
from emlp.groups import O

import tensorflow as tf
import os
from tqdm import tqdm

tpu_name = os.environ.get("TPU_NAME")
data_path = os.environ.get("WAYMO_PATH")


if tpu_name is not None:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)


# Example field definition
roadgraph_features = {
    'roadgraph_samples/dir':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    'roadgraph_samples/id':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/type':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/valid':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/xyz':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
}

necessary_features = {
    # 'state/id':
    #     tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    # 'state/is_sdc':
    #     tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    # 'state/tracks_to_predict':
    #     tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    # 'state/current/bbox_yaw':
    #     tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    # 'state/current/timestamp_micros':
    #     tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    # 'state/current/vel_yaw':
    #     tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    # 'state/future/bbox_yaw':
    #     tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    # 'state/future/timestamp_micros':
    #     tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    # 'state/future/vel_yaw':
    #     tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    # 'state/past/bbox_yaw':
    #     tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    # 'state/past/timestamp_micros':
    #     tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    # 'state/past/vel_yaw':
    #     tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

features_description = {}
features_description.update(state_features)

if data_path is not None:
    DATASET_FOLDER = data_path
else:
    # DATASET_FOLDER = "/mnt/disks/disk/tf_example"
    # DATASET_FOLDER = "/siml/Datasets/tf_example"
    DATASET_FOLDER = "/mnt/gsai/Datasets/WaymoOpenDataset/tf_example"
# 13851
TRAIN_FILES = f"{DATASET_FOLDER}/training/training_tfexample.tfrecord*"
# 8456
VALID_FILES = f"{DATASET_FOLDER}/validation/validation_tfexample.tfrecord*"
TEST_FILES = f"{DATASET_FOLDER}/testing/testing_tfexample.tfrecord*"  # 8789
# PATH = {
#     "training": TRAIN_FILES,
#     "validation": VALID_FILES,
#     "testing": VALID_FILES
# }
PATH = {
    "training": TRAIN_FILES,
    "validation": TRAIN_FILES,
    "testing": TRAIN_FILES
}
LENGTH = {
    "training": 13039,
    "validation": 7808,
    "testing": 8263
}

##@### z diff 0.1 ###########
# s = 5614.92
# MEAN = [-345.08, 248.97, -8.57]
##### radius 2000 ##########
# s = 733.66
# MEAN = [109.37, 81.18, -22.86]
##### radius 5000 active 5 ###
# s = 1662
# MEAN = [483, -547, -14]
# STD = [s, s, s]
##### gather and z diff 0.1####
# s = 4.8
# MEAN = [0, 0, 0]
# STD = [s, s, s]
##### z diff 0.1 x diff y diff 5
# s = 1.38
# MEAN = [0, 0, 0]
# STD = [s, s, s]
##### discriminative gathering [1,1,0.993] ###
s = 1.307
MEAN = [0, 0, -0.231]
STD = [s, s, s]


def parse_tf_example(tf_example):
    return tf.io.parse_single_example(tf_example, features_description)


def vectorize(data):
    features = ["width", "length", "height",
                "velocity_x", "velocity_y", "x", "y", "z"]
    moments = ["past", "current", "future"]
    states = []
    valids = []
    for m in moments:
        substates = []
        for f in features:
            substates.append(data[f"state/{m}/{f}"])
        st = tf.stack(substates, axis=-1)
        states.append(st)
        valids.append(data[f"state/{m}/valid"])
    vector = tf.concat(states, axis=1)
    valid_mask = tf.concat(valids, axis=1)
    car_type = data["state/type"]
    return {
        "car_type": car_type,
        "valid_mask": valid_mask,
        "vector": vector
    }


def collect_last(data):
    """
    collect vehicles that have fully valid trajectoy and move actively
    car_type: (batchsize,)
    valid_mask: (batchsize, 91)
    vector: (batchsize, 91, 8)
    """
    # car_type = data["car_type"]
    valid_mask = data["valid_mask"]
    vector = data["vector"]
    total_length = 24
    count = tf.reduce_sum(valid_mask[:, :total_length], axis=1)
    mask = count == total_length
    vector = tf.boolean_mask(vector[:, :total_length, -3:], mask) # index -3 filter "x","y","z"
    # vector = tf.boolean_mask(vector[:, :total_length:2, -3:], mask) # index -3 filter "x","y","z"
    diff = (vector[:, 0, :]-vector[:, 5, :])**2
    # diff = (vector[:, 0, -1]-vector[:, 5, -1])**2
    norm = (vector[:, :, :])**2
    activeness = tf.math.sqrt(tf.reduce_sum(diff, axis=1))
    # activeness = tf.math.sqrt(diff)
    radius = tf.reduce_mean(tf.math.sqrt(tf.reduce_sum(norm, axis=2)), axis=1)
    mask1 = (activeness > 5)
    # mask = (activeness > 0.1)
    mask2 = (radius < 5000)
    mask = mask1 & mask2
    vector = tf.boolean_mask(vector, mask)

    vector = tf.concat((vector[:,:6,:], vector[:,-6:,:]), axis=1)
    return vector

def collect(data):
    """
    collect vehicles that have fully valid trajectoy and move actively
    car_type: (batchsize,)
    valid_mask: (batchsize, 91)
    vector: (batchsize, 91, 8)
    """
    # car_type = data["car_type"]
    valid_mask = data["valid_mask"]
    vector = data["vector"]
    total_length = 24
    count = tf.reduce_sum(valid_mask[:, :total_length], axis=1)
    mask = count == total_length
    vector = tf.boolean_mask(vector[:, :total_length:2, -3:], mask) # index -3 filter "x","y","z"
    xdiff = (vector[:, 0, 0]-vector[:, 5, 0])**2
    ydiff = (vector[:, 0, 1]-vector[:, 5, 1])**2
    zdiff = (vector[:, 0, 2]-vector[:, 5, 2])**2
    xactive = tf.math.sqrt(xdiff)
    yactive = tf.math.sqrt(ydiff)
    zactive = tf.math.sqrt(zdiff)
    mask = (zactive > 0.05) & (xactive < 5) & (yactive<5)
    vector = tf.boolean_mask(vector, mask)

    vector = tf.concat((vector[:,:6,:], vector[:,-6:,:]), axis=1)
    return vector

def normalize(data):
    mean = tf.constant(MEAN, dtype=tf.float32)
    mean = mean[tf.newaxis, tf.newaxis, :]
    std = tf.constant(STD, dtype=tf.float32)
    std = std[tf.newaxis, tf.newaxis, :]
    return (data - mean)/std


def gather(data):
    # trajectory-wise mean
    mean = tf.reduce_mean(data, axis=1)
    mean = mean[:, tf.newaxis, :]*tf.constant([1.,1.,0.993])
    return data - mean


def labeling(data):
    return data[:, :18], data[:, 18:]


def preprocess(path, key):
    filenames = tf.io.matching_files(path)
    assert len(filenames) > 0, "No Dataset Detected"
    n_files = len(filenames)//10
    filenames = filenames[:n_files]
    split = n_files//6
    if "training" in key:
        dataset = tf.data.TFRecordDataset(filenames[:-2*split])
    elif "validation" in key:
        dataset = tf.data.TFRecordDataset(filenames[-2*split:-1*split])
    elif "testing" in key:
        dataset = tf.data.TFRecordDataset(filenames[-1*split:])
    dataset = dataset.map(parse_tf_example)
    dataset = dataset.map(vectorize)
    dataset = dataset.map(collect)
    dataset = dataset.map(gather)
    return dataset

class Dataloader():

    def __init__(self, key, batch_size, tpu=False):
        self.tpu = tpu
        path = PATH[key]
        self.key = key
        self.rep_in = 6*Vector
        self.rep_out = 6*Vector
        self.symmetry = O(3)
        def func():
            dataset = preprocess(path, key)
            dataset = dataset.map(normalize)
            dataset = dataset.map(lambda v: tf.reshape(v, (-1, 36)))
            dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
            # dataset = dataset.repeat()
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(labeling)
            return dataset
        if tpu:
            with strategy.scope():
                self.loader = func()
        else:
            self.loader = func()

    def __iter__(self):
        if self.tpu:
            with strategy.scope():
                for data in self.loader:
                    yield data
        else:
            for data in self.loader:
                yield data

    def __len__(self):
        return LENGTH[self.key]

    # def __getitem__(self, idx):
    #     instance =  self.dataset[idx]
    #     return instance[:,:18], instance[:,18:]


# def count_length(key):
#     trainloader = Dataloader(key, 1)
#     count = 0
#     initial_x = None
#     initial_y = None
#     for i, (x, y) in enumerate(trainloader):
#         # x_mean = tf.reduce_mean(x, axis=1)
#         # x -= x_mean[:, tf.newaxis]
#         print("x[0]", x[0])
#         print("y[0]", y[0])
#         if i == 0:
#             initial_x = x
#             initial_y = y
#         else:
#             if tf.math.reduce_all(initial_x == x) and tf.math.reduce_all(initial_y == y):
#                 break
#         count += 1
#     return count


def cal_mean_std(key, tpu=False):
    path = PATH[key]
    def func():
        dataset = preprocess(path, key)
        mean = 0
        length = 0
        print("Calculate Mean..")
        for v in tqdm(dataset):
            length += v.shape[0]
            mean += tf.reduce_mean(tf.reduce_sum(v, axis=0), axis=0)
        mean /= length
        # _mean = mean[tf.newaxis, tf.newaxis, :]
        _mean = tf.reduce_mean(mean)
        std = 0
        print("Calculate STD..")
        for v in tqdm(dataset):
            std += tf.reduce_mean(tf.reduce_sum((v-_mean)**2, axis=0), axis=0)
        std /= length
        total_std = tf.reduce_mean(std)
        std = tf.math.sqrt(std)
        total_std = tf.sqrt(total_std)
        return length, mean, std, total_std

    if tpu:
        with strategy.scope():
            return func()
    else:
        return func()


if __name__ == "__main__":
    train_stat = cal_mean_std("training")
    print("train", train_stat)
    valid_stat = cal_mean_std("validation")
    print("valid", valid_stat)
    test_stat = cal_mean_std("testing")
    print("test", test_stat)
