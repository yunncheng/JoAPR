import random
import math
import numpy as np


def generate_fewshot_dataset_with_symflip_noise(data_source, num_shots, num_fp, seed):
    if num_shots < 1:
        return data_source
    random.seed(seed)
    np.random.seed(seed)
    print(f"Creating a 16-shot dataset with {num_fp} symflip noisy shots")
    output = []

    label_set = set()
    img_paths = []
    for item in data_source:
        label_set.add(item.label)
        img_paths.append(item.impath)
    img_paths = np.array(img_paths)
    num_classes = max(label_set) + 1
    img_paths_dict = {k: v for v, k in enumerate(img_paths)}
    rng = np.random.default_rng(seed=seed)
    label_imgpath_dict = {}
    for idx in range(num_classes):
        label_imgpath_dict[idx] = np.array([])
    for item in data_source:
        label_imgpath_dict[item.label] = np.append(label_imgpath_dict[item.label], np.array(item.impath))
    tp_all_img_index_dict = {}
    fp_all_img_index_dict = {}
    fp_all_img_index_list = []

    for id in range(num_classes):
        split = int(math.ceil((len(label_imgpath_dict[id]) * (0.5))))
        gt_class_img_index = []
        for img in list(label_imgpath_dict[id]):
            gt_class_img_index.append(img_paths_dict[img])
        # if num_fp == 0:
        #    tp_all_img_index_dict[id] = gt_class_img_index[:]
        # else:
        tp_all_img_index_dict[id] = gt_class_img_index[:split]
        fp_all_img_index_dict[id] = gt_class_img_index[split:]
        fp_all_img_index_list.extend(gt_class_img_index[split:])
    fp_all_img_index_set = set(fp_all_img_index_list)

    fp_ids_chosen = set()
    for id in range(num_classes):
        class_imgpath = label_imgpath_dict[id][0]
        class_imgpath_idx = img_paths_dict[class_imgpath]
        classname = data_source[class_imgpath_idx].classname
        class_img_index = []
        for img in list(label_imgpath_dict[id]):
            class_img_index.append(img_paths_dict[img])
        # noisy lebels - randomly draw FP samples with their indice
        class_img_index = tp_all_img_index_dict[id]
        fp_ids_set = fp_all_img_index_set.difference(class_img_index, fp_all_img_index_dict[id], fp_ids_chosen)
        fp_ids = random.choices(list(fp_ids_set), k=num_fp)
        fp_ids_chosen.update(fp_ids)
        # noisy lebels - randomly draw FP samples with their indice
        img_paths_class = img_paths[class_img_index]
        if len(img_paths_class) < num_shots:
            is_replace = True
        else:
            is_replace = False
        num_shots_array = rng.choice(len(img_paths_class), size=num_shots, replace=is_replace)
        img_paths_class = img_paths_class[num_shots_array]
        # noisy lebels - dilute with FP samples
        for i in range(num_fp):
            img_paths_class[i] = img_paths[fp_ids][i]

        for img_path in img_paths_class:
            index = img_paths_dict[img_path]
            data_source[index]._label = id
            data_source[index]._classname = classname
            output.append(data_source[index])
    for item in output:
        print(item.label, item.impath, item.classname)
    return output


def generate_fewshot_dataset_with_pairflip_noise(data_source, num_shots, num_fp, seed):
    if num_shots < 1:
        return data_source
    random.seed(seed)
    np.random.seed(seed)
    print(f"Creating a 16-shot dataset with {num_fp} pairflip noisy shots")
    output = []

    label_set = set()
    img_paths = []
    for item in data_source:
        label_set.add(item.label)
        img_paths.append(item.impath)
    img_paths = np.array(img_paths)
    num_classes = max(label_set) + 1
    img_paths_dict = {k: v for v, k in enumerate(img_paths)}
    rng = np.random.default_rng(seed=seed)
    label_imgpath_dict = {}
    for idx in range(num_classes):
        label_imgpath_dict[idx] = np.array([])
    for item in data_source:
        label_imgpath_dict[item.label] = np.append(label_imgpath_dict[item.label], np.array(item.impath))
    tp_all_img_index_dict = {}
    fp_all_img_index_dict = {}
    fp_all_img_index_list = []

    for id in range(num_classes):
        split = int(math.ceil((len(label_imgpath_dict[id]) * (0.5))))
        gt_class_img_index = []
        for img in list(label_imgpath_dict[id]):
            gt_class_img_index.append(img_paths_dict[img])
        # if num_fp == 0:
        #    tp_all_img_index_dict[id] = gt_class_img_index[:]
        # else:
        tp_all_img_index_dict[id] = gt_class_img_index[:split]
        fp_all_img_index_dict[id] = gt_class_img_index[split:]
        random.shuffle(tp_all_img_index_dict[id])
        random.shuffle(fp_all_img_index_dict[id])

    # noise_T_matrix
    P = np.zeros((num_classes, num_classes))
    if num_fp > 0:
        # 0 -> 1
        P[0, 1] = 1
        for i in range(1, num_classes - 1):
            P[i, i + 1] = 1
        P[num_classes - 1, 0] = 1

    label_list = []
    for idx in range(num_classes):
        label_list.append(idx)
    fp_label_idx = np.zeros(num_classes)

    for id in range(num_classes):
        class_imgpath = label_imgpath_dict[id][0]
        class_imgpath_idx = img_paths_dict[class_imgpath]
        classname = data_source[class_imgpath_idx].classname
        # noisy lebels - randomly draw FP samples with their indice
        class_img_index = tp_all_img_index_dict[id]

        fp_ids = []
        label_id = random.choices(label_list, weights=P[id, :], k=num_fp)
        for index in label_id:
            fp_ids.append(fp_all_img_index_dict[index][int(fp_label_idx[index])])
            if fp_label_idx[index] + 1 < len(fp_all_img_index_dict[index]):
                fp_label_idx[index] += 1

        # noisy lebels - randomly draw FP samples with their indice
        img_paths_class = img_paths[class_img_index]
        if len(img_paths_class) < num_shots:
            is_replace = True
        else:
            is_replace = False
        num_shots_array = rng.choice(len(img_paths_class), size=num_shots, replace=is_replace)
        img_paths_class = img_paths_class[num_shots_array]
        # noisy lebels - dilute with FP samples
        for i in range(num_fp):
            img_paths_class[i] = img_paths[fp_ids][i]

        for img_path in img_paths_class:
            index = img_paths_dict[img_path]
            data_source[index]._label = id
            data_source[index]._classname = classname
            output.append(data_source[index])
    for item in output:
        print(item.label, item.impath, item.classname)
    return output