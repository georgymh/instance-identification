import os
import xmltodict
import pickle
import random
from collections import Counter
from multiprocessing.pool import ThreadPool

from imageio import imread
from PIL import Image

ORIGINAL_DATA_PATH_STR = 'data/ILSVRC2015/Data/VID/'
ORIGINAL_ANNOTATIONS_PATH_STR = 'data/ILSVRC2015/Annotations/VID/'
CROPPED_DATA_PATH_STR = 'data/ILSVRC2015-VID-Curation/Data/VID/'
RANDOM_CROPS_PATH_STR = 'data/ILSVRC2015-VID-Random/Data/VID/'
NUM_RANDOM_CROPS = 5

SUBDIR_MAP = {'ILSVRC2015_VID_train_0000': 'a',
            'ILSVRC2015_VID_train_0001': 'b',
            'ILSVRC2015_VID_train_0002': 'c',
            'ILSVRC2015_VID_train_0003': 'd',
            '': 'e'}

TRAIN_STATS_PKL = 'dataset/imagenetvid/train_data_stats.pkl'
VAL_STATS_PKL = 'dataset/imagenetvid/val_data_stats.pkl'

TRAIN_SET_PKL = 'dataset/imagenetvid/train_set.pkl'
TRAIN_SET_STATS_PKL = 'dataset/imagenetvid/train_stats.pkl'

TRAIN_EVAL_SET_PKL = 'dataset/imagenetvid/train_eval_set.pkl'
TRAIN_EVAL_STATS_PKL = 'dataset/imagenetvid/train_eval_stats.pkl'

VAL_EVAL_SET_PKL = 'dataset/imagenetvid/val_eval_set.pkl'
VAL_EVAL_STATS_PKL = 'dataset/imagenetvid/val_eval_stats.pkl'

EASY_VAL_EVAL_SET_PKL = 'dataset/imagenetvid/easy_val_eval_set.pkl'
EASY_VAL_EVAL_STATS_PKL = 'dataset/imagenetvid/easy_val_eval_stats.pkl'


def make_training_pickle(num_batches, num_snippets, num_instances):
    output = open(TRAIN_SET_PKL, 'ab')
    stats = Counter()
    snippets = _get_snippets_dict(training=True)

    # Repeat the following for num_batches steps...
    update = num_batches // 10
    for step in range(num_batches):
        batch_images, batch_labels = [], []
        label = 0

        # For random video snippet from 1...num_snippets
        i = 0
        while i < num_snippets:
            snippet_path, _ = random.choice(snippets['multiple_bboxes'])

            # Choose a random frame
            annotation1 = _get_random_frame_annotation(snippet_path)
            if annotation1 == None: continue

            # Pick a random object in the frame with occluded = false
            object1 = _get_random_object_annotation(annotation1, allow_occluded=False)
            if object1 == None: continue
            trackid = object1['trackid']

            # Pick num_instances - 1 other random frames where the same object
            # has occluded = false
            all_frames = [annotation1]
            all_objects = [object1]
            timeout = 100
            while len(all_objects) < num_instances and timeout > 0:
                timeout -= 1
                other_annotation = _get_random_frame_annotation(snippet_path)
                if other_annotation == None or other_annotation in all_frames: continue
                o = _get_object(other_annotation, trackid)
                #o = _get_random_object_annotation(annotation2, allow_occluded=False)
                if o == None or o['trackid'] != trackid or o['occluded'] == '1' : continue
                all_frames.append(other_annotation)
                all_objects.append(o)
            if timeout == 0: continue
            for o in all_objects: stats[o['name']] += 1
            stats[object1['name']] += 1

            # Label each object with <label> (and update label += 1)
            # Add the num_instances elements into batch
            all_images = [_get_cropped_framepath(annotation1, trackid, training=True)]
            all_images.extend([_get_cropped_framepath(ann, trackid, training=True) \
                for ann in all_frames[1:]])

            batch_images.extend(all_images)
            batch_labels.extend([label] * num_instances)

            label += 1
            i += 1

        # Store the batch of batch size B = PK
        batch = (batch_images, batch_labels)
        pickle.dump(batch, output, -1)
        if step % update == 0: print("Batch #: {0}".format(step))

    # Store the metadata into another pickle
    output.close()
    output = open(TRAIN_SET_STATS_PKL, 'ab')
    pickle.dump(stats, output, -1)
    output.close()


def make_training_eval_pickle():
    training_eval_dataset, stats = [], Counter()
    output = open(TRAIN_EVAL_SET_PKL, 'ab')
    snippets = _get_snippets_dict(training=True)

    # For each random multiobject video snippets
    for video_snippet, _ in snippets['multiple_bboxes']:
        # Choose a random frame
        frame_annotation1 = _get_random_frame_annotation(video_snippet)

        # Choose a random, non-occluded object in the image and call it anchor
        anchor = _get_random_object_annotation(frame_annotation1, allow_occluded=False)
        if anchor == None: continue
        trackid = anchor['trackid']

        found_positive = False
        timeout = 10000
        while not found_positive and timeout > 0:
            timeout -= 1
            # Find a second frame
            frame_annotation2 = _get_random_frame_annotation(video_snippet)
            while frame_annotation1['filename'] == frame_annotation2['filename']:
                frame_annotation2 = _get_random_frame_annotation(video_snippet)

            # Find the same object in this frame and verify it's not occluded
            positive = _get_object(frame_annotation2, trackid)
            if positive == None: continue
            # (if it is, repeat the last step until we find a good frame)
            if positive['occluded'] == '0':
                found_positive = True
        if timeout == 0 or found_positive == False: continue

        # Store all the objects of the last frame in a list
        all_objects = _get_all_objects(frame_annotation2)
        if len(all_objects) < 3: continue
        # (Put the object class in a dictionary for stats)
        for o in all_objects: stats[o['name']] += 1
        stats[anchor['name']] += 1

        # Label the first object and the same one from the list
        # as 0, and the rest as 1
        zero_if_equal_one_otherwise = lambda o: 0 if o['trackid'] == trackid else 1
        labels = [0] + [zero_if_equal_one_otherwise(o) for o in all_objects]

        # Prepare batch
        images = [_get_cropped_framepath(frame_annotation1, trackid, training=True)] + \
          [_get_cropped_framepath(frame_annotation2, o['trackid'], training=True) for o in all_objects]
        batch = (images, labels)

        # Put this batch into the pickle
        pickle.dump(batch, output, -1)

    # Store the metadata into another pickle
    output.close()
    output2 = open(TRAIN_EVAL_STATS_PKL, 'ab')
    pickle.dump(stats, output2, -1)
    output2.close()


def make_validation_pickle():
    val_eval_dataset, stats = [], Counter()
    output = open(VAL_EVAL_SET_PKL, 'ab')
    snippets = _get_snippets_dict(training=False)

    # For each random multiobject video snippets
    for video_snippet, _ in snippets['multiple_bboxes']:
        # Choose a random frame
        frame_annotation1 = _get_random_frame_annotation(video_snippet)

        # Choose a random, non-occluded object in the image and call it anchor
        anchor = _get_random_object_annotation(frame_annotation1, allow_occluded=False)
        if anchor == None: continue
        trackid = anchor['trackid']

        found_positive = False
        timeout = 10000
        while not found_positive and timeout > 0:
            timeout -= 1
            # Find a second frame
            frame_annotation2 = _get_random_frame_annotation(video_snippet)
            while frame_annotation1['filename'] == frame_annotation2['filename']:
                frame_annotation2 = _get_random_frame_annotation(video_snippet)

            # Find the same object in this frame and verify it's not occluded
            positive = _get_object(frame_annotation2, trackid)
            if positive == None: continue
            # (if it is, repeat the last step until we find a good frame)
            if positive['occluded'] == '0':
                found_positive = True
        if timeout == 0 or found_positive == False: continue

        # Store all the objects of the last frame in a list
        all_objects = _get_all_objects(frame_annotation2)
        if len(all_objects) < 3: continue
        # (Put the object class in a dictionary for stats)
        for o in all_objects: stats[o['name']] += 1
        stats[anchor['name']] += 1

        # Label the first object and the same one from the list
        # as 0, and the rest as 1
        zero_if_equal_one_otherwise = lambda o: 0 if o['trackid'] == trackid else 1
        labels = [0] + [zero_if_equal_one_otherwise(o) for o in all_objects]

        # Prepare batch
        images = [_get_cropped_framepath(frame_annotation1, trackid, training=False)] + \
          [_get_cropped_framepath(frame_annotation2, o['trackid'], training=False) for o in all_objects]
        batch = (images, labels)

        # Put this batch into the pickle
        pickle.dump(batch, output, -1)

    # Store the metadata into another pickle
    output.close()
    output = open(VAL_EVAL_STATS_PKL, 'ab')
    pickle.dump(stats, output, -1)
    output.close()


def make_easy_validation_pickle(target_num_batches=9000):
    val_eval_dataset, stats = [], Counter()
    output = open(EASY_VAL_EVAL_SET_PKL, 'ab')
    snippets = _get_snippets_dict(training=False)

    total_batches = 0
    while total_batches < target_num_batches:
        # For each random multiobject video snippets
        for video_snippet, _ in snippets['multiple_bboxes'] + snippets['single_bboxes']:
            # Choose a random frame
            frame_annotation1 = _get_random_frame_annotation(video_snippet)

            # Choose a random, non-occluded object in the image and call it anchor
            anchor = _get_random_object_annotation(frame_annotation1, allow_occluded=False)
            if anchor == None: continue
            trackid = anchor['trackid']

            found_positive = False
            timeout = 10000
            while not found_positive and timeout > 0:
                timeout -= 1
                # Find a second frame
                frame_annotation2 = _get_random_frame_annotation(video_snippet)
                while frame_annotation1['filename'] == frame_annotation2['filename']:
                    frame_annotation2 = _get_random_frame_annotation(video_snippet)

                # Find the same object in this frame and verify it's not occluded
                positive = _get_object(frame_annotation2, trackid)
                if positive == None: continue
                # (if it is, repeat the last step until we find a good frame)
                if positive['occluded'] == '0':
                    found_positive = True
            if timeout == 0 or found_positive == False: continue

            # Get NUM_RANDOM_CROPS random crops from the second frame.
            random_crops = _make_return_random_crops(frame_annotation2, NUM_RANDOM_CROPS,
                training=False, black_area=positive, iou_limit=0.3)
            if len(random_crops) == 0:
                print("\tCouldn't find enough random crops...")
                continue

            # Label the first object and second object the same (anchor and positive),
            # and the rest with another label.
            labels = [0, 0] + [1 for _ in range(len(random_crops))]

            # Prepare batch
            images = [_get_cropped_framepath(frame_annotation1, trackid, training=False)] + \
              [_get_cropped_framepath(frame_annotation2, trackid, training=False)] + \
              random_crops
            batch = (images, labels)

            # Put this batch into the pickle
            pickle.dump(batch, output, -1)
            total_batches += len(labels)
            stats[anchor['name']] += 1

        print("Created {}/{} batches so far.".format(total_batches, target_num_batches))

    # Store the metadata into another pickle
    output.close()
    output = open(EASY_VAL_EVAL_STATS_PKL, 'ab')
    pickle.dump(stats, output, -1)
    output.close()


def save_training_snippets_dict():
    if os.path.isfile(TRAIN_STATS_PKL):
        return

    TYPE = "train"
    paths = {
        "missing_bbox": [],
        "multiple_bboxes": [],
        "single_bboxes": []
    }
    stats = {
        "missing_bbox": 0,
        "multiple_bboxes": 0,
        "single_bboxes": 0
    }
    annotations_base_path = os.path.join(ORIGINAL_ANNOTATIONS_PATH_STR, TYPE)
    for dir1 in os.listdir(annotations_base_path):
        dir1 = os.path.join(annotations_base_path + "/" + dir1)
        for snippet_path in os.listdir(dir1):
            snippet_path = os.path.join(dir1 + "/" + snippet_path)

            dict_obj = { snippet_path : [] }
            counts = {
                "missing_bbox": 0,
                "multiple_bboxes": 0,
                "single_bboxes": 0,
                "total": 0
            }

            for f in os.listdir(snippet_path):
                if f[-3:] != 'xml':
                    continue
                filepath = snippet_path + "/" + f
                with open(filepath) as file:
                    fobj = xmltodict.parse(file.read())

                annotation = fobj['annotation']
                if "object" not in annotation:
                    # no bounding boxes
                    counts["missing_bbox"] += 1
                    stats["missing_bbox"] += 1
                elif isinstance(annotation["object"], list):
                    # multiple objects in a frame
                    counts["multiple_bboxes"] += 1
                    stats["multiple_bboxes"] += 1
                else:
                    # single object in a frame
                    counts["single_bboxes"] += 1
                    stats["single_bboxes"] += 1
                counts["total"] += 1

            if counts["missing_bbox"] > counts["multiple_bboxes"] and \
               counts["missing_bbox"] > counts["single_bboxes"]:
                # missing_bbox -> discard
                ratio = counts["missing_bbox"] / counts["total"]
                paths["missing_bbox"].append((snippet_path, ratio))
            elif counts["multiple_bboxes"] > counts["missing_bbox"] and \
               counts["multiple_bboxes"] > counts["single_bboxes"]:
                # multiple bboxes
                ratio = counts["multiple_bboxes"] / counts["total"]
                paths["multiple_bboxes"].append((snippet_path, ratio))
            else:
                # single bbox
                ratio = counts["single_bboxes"] / counts["total"]
                paths["single_bboxes"].append((snippet_path, ratio))

    paths['all'] = [e + ('single_bboxes',) for e in paths['single_bboxes']] + \
       [e + ('multiple_bboxes',) for e in paths['multiple_bboxes']] + \
       [e + ('missing_bbox',) for e in paths['missing_bbox']]

    paths['all_but_missing'] = [e + ('single_bboxes',) for e in paths['single_bboxes']] + \
       [e + ('multiple_bboxes',) for e in paths['multiple_bboxes']]

    # Save into Pickle
    output = open(TRAIN_STATS_PKL, 'wb')
    pickle.dump(stats, output, -1)
    pickle.dump(paths, output, -1)
    output.close()


def save_testing_snippets_dict():
    if os.path.isfile(VAL_STATS_PKL):
        return

    TYPE = "val"
    paths = {
        "missing_bbox": [],
        "multiple_bboxes": [],
        "single_bboxes": []
    }
    stats = {
        "missing_bbox": 0,
        "multiple_bboxes": 0,
        "single_bboxes": 0
    }
    annotations_base_path = os.path.join(ANNOTATIONS_PATH_STR, TYPE)
    for dir1 in os.listdir(annotations_base_path):
        dir1 = os.path.join(annotations_base_path + "/" + dir1)
        for snippet_path in os.listdir(dir1):
            snippet_path = os.path.join(dir1 + "/" + snippet_path)

            counts = {
                "missing_bbox": 0,
                "multiple_bboxes": 0,
                "single_bboxes": 0,
                "total": 0
            }

            filepath = snippet_path
            if filepath[-3:] != 'xml':
                continue
            with open(filepath) as file:
                fobj = xmltodict.parse(file.read())

            annotation = fobj['annotation']
            if "object" not in annotation:
                # no bounding boxes
                counts["missing_bbox"] += 1
                stats["missing_bbox"] += 1
            elif isinstance(annotation["object"], list):
                # multiple objects in a frame
                counts["multiple_bboxes"] += 1
                stats["multiple_bboxes"] += 1
            else:
                # single object in a frame
                counts["single_bboxes"] += 1
                stats["single_bboxes"] += 1
            counts["total"] += 1

        if counts["missing_bbox"] > counts["multiple_bboxes"] and \
           counts["missing_bbox"] > counts["single_bboxes"]:
            # missing_bbox -> discard
            ratio = counts["missing_bbox"] / counts["total"]
            paths["missing_bbox"].append((dir1, ratio))
        elif counts["multiple_bboxes"] > counts["missing_bbox"] and \
           counts["multiple_bboxes"] > counts["single_bboxes"]:
            # multiple bboxes
            ratio = counts["multiple_bboxes"] / counts["total"]
            paths["multiple_bboxes"].append((dir1, ratio))
        else:
            # single bbox
            ratio = counts["single_bboxes"] / counts["total"]
            paths["single_bboxes"].append((dir1, ratio))

    paths['all'] = [e + ('single_bboxes',) for e in paths['single_bboxes']] + \
       [e + ('multiple_bboxes',) for e in paths['multiple_bboxes']] + \
       [e + ('missing_bbox',) for e in paths['missing_bbox']]

    paths['all_but_missing'] = [e + ('single_bboxes',) for e in paths['single_bboxes']] + \
       [e + ('multiple_bboxes',) for e in paths['multiple_bboxes']]

    # Save into Pickle
    output = open(VAL_STATS_PKL, 'wb')
    pickle.dump(stats, output, -1)
    pickle.dump(paths, output, -1)
    output.close()


def _get_snippets_dict(training=True):
    """
    Returns a structure that contains the video snippets categorized by
    'missing_bbox', 'single_bboxes', 'multiple_bboxes', and 'all'.
    """
    if training:
        pkl_file = open(TRAIN_STATS_PKL, 'rb')
        print("Loading training pickle...")
    else:
        pkl_file = open(VAL_STATS_PKL, 'rb')
        print("Loading validation pickle...")

    stats = pickle.load(pkl_file)
    paths = pickle.load(pkl_file)
    pkl_file.close()

    return paths


def _get_random_frame_annotation(snippet_path):
    frame_files = os.listdir(snippet_path)
    random.shuffle(frame_files)
    for file in frame_files:
        if file[-3:] != 'xml': continue
        annotation_path = os.path.join(snippet_path, file)
        return _get_annotation(annotation_path)


def _get_pair_of_frames(snippet_path):
    frames = [_get_random_frame_annotation(snippet_path) for _ in range(2)]
    order_by_time = lambda f: int(f['filename'])
    return tuple(sorted(frames, key=order_by_time))


def _get_random_object_annotation(annotation, allow_occluded=True):
    if 'object' in annotation:
        obj = annotation['object']
        if type(obj) == list and len(obj) > 0:
            # multiple objects
            random.shuffle(obj)
            for o in obj:
                if allow_occluded or o['occluded'] == '0':
                    return o
        else:
            # single object
            if allow_occluded or obj['occluded'] == '0':
                return obj


def _get_all_objects(annotation):
    all_objects = []
    if 'object' in annotation:
        obj = annotation['object']
        if type(obj) == list and len(obj) > 0:
            # multiple objects
            all_objects = obj
        else:
            # single object
            all_objects.append(obj)
    return all_objects


def _get_annotation(annotation_path):
    with open(annotation_path) as file:
        xmlobj = xmltodict.parse(file.read())
    annotation = xmlobj['annotation']
    return annotation


def _get_object(annotation, trackid):
    if 'object' in annotation:
        obj = annotation['object']
        if type(obj) == list and len(obj) > 0:
            # multiple objects
            for o in obj:
                if o['trackid'] == trackid:
                    return o
        else:
            # single object
            if obj['trackid'] == trackid:
                return obj


def _get_cropped_framepath(annotation, track_id, training=True):
    """Assumes track_id is a string and not an int."""
    def _get_full_snippetpath(snippet_path):
        return os.path.join(CROPPED_DATA_PATH_STR, 'train', snippet_path)

    if training:
        folder1, folder2 = annotation["folder"].split('/')
        snippet_path = os.path.join(SUBDIR_MAP[folder1], folder2)
    else:
        snippet_path = 'e/' + annotation["folder"]
    full_snippet_path = _get_full_snippetpath(snippet_path)

    if len(track_id) == 1:
        track_id = "0" + track_id

    filename = annotation["filename"]
    real_filename = filename + "." + track_id + ".crop.x.jpg"

    return os.path.join(full_snippet_path, real_filename)


def _get_original_framepath(annotation, training=True):
    if training:
        folder1, folder2 = annotation["folder"].split('/')
        snippet_path = os.path.join(ORIGINAL_DATA_PATH_STR, 'train', folder1, folder2)
    else:
        snippet_path = os.path.join(ORIGINAL_DATA_PATH_STR, 'val', annotation["folder"])
    filename = annotation['filename'] + '.JPEG'
    return os.path.join(snippet_path, filename)


def _make_return_random_crops(annotation, num_crops, training, size=(255, 255),
    black_area=None, iou_limit=0.2):
    transform_bbox = lambda bbox: (bbox[0], bbox[2], bbox[1], bbox[3])
    framepath = _get_original_framepath(annotation, training)

    im = Image.open(framepath)
    max_x, max_y = im.size
    w, h = size

    if max_x - w < num_crops or max_y - h < num_crops:
        return []

    if black_area != None:
        bndbox = black_area['bndbox']
        black_bbox = [bndbox['xmin'], bndbox['xmax'], bndbox['ymin'], bndbox['ymax']]
        black_bbox = [int(x) for x in black_bbox]

    i = 0
    random_crops = []
    while len(random_crops) < num_crops:
        random_x = random.randint(0, max_x - w)
        random_y = random.randint(0, max_y - h)
        random_bbox = [random_x, random_x + w, random_y, random_y + h]
        if black_area != None and _iou(random_bbox, black_bbox) < iou_limit:
            # crop the box into a new image file
            new_file_name = "{}.{}.{}.JPEG".format(
                annotation["folder"], annotation["filename"], str(i))
            new_file = os.path.join(RANDOM_CROPS_PATH_STR, 'e', new_file_name)
            im.crop(transform_bbox(random_bbox)).save(new_file)
            # add the file path to the array
            random_crops.append(new_file)
        # else:
        #     print("\tSkipping bbox... ({})".format(annotation["filename"]))
        i += 1
        if i > 200: break
    return random_crops


def _iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    bb1 or bb2: [x1, x2, y1, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner.
    """
    def _get_iou(bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2'], str(bb2['x1']) + "," + str(bb2['x2'])
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    keys = ['x1', 'x2', 'y1', 'y2']
    box1 = {k : v for k, v in zip(keys, bb1)}
    box2 = {k : v for k, v in zip(keys, bb2)}
    return _get_iou(box1, box2)


if __name__ == '__main__':
    print("Making the snippet dictionaries...")
    pool = ThreadPool(processes=2)
    results = [
        pool.apply_async(save_training_snippets_dict, []),
        pool.apply_async(save_testing_snippets_dict, [])
    ]
    ans = [res.get() for res in results]
    print("Done!")

    print("Making the evaluation pickles...")
    pool = ThreadPool(processes=3)
    results = [
        pool.apply_async(make_training_eval_pickle, []),
        pool.apply_async(make_validation_pickle, []),
        pool.apply_async(make_easy_validation_pickle, []),
    ]
    ans = [res.get() for res in results]
    print("Done!")

    print("Making the training pickle...")
    pool = ThreadPool(processes=4)
    results = [
        # TODO: Make the parameters (num_batches, num_snippets, num_instances)
        # be configurable using argparse.
        pool.apply_async(make_training_pickle, [25000, 15, 3]),
        pool.apply_async(make_training_pickle, [25000, 15, 3]),
        pool.apply_async(make_training_pickle, [25000, 15, 3]),
        pool.apply_async(make_training_pickle, [25000, 15, 3]),
    ]
    ans = [res.get() for res in results]
    print("Done!")
