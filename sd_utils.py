import re
import os
import csv
import shutil
import subprocess
import numpy as np
import cv2 as cv
import networkx
from networkx.algorithms.components.connected import connected_components
from streamduet_utils import list_frames,get_images_length,get_image_extension
import time
def remove_before_first_underscore(s):
    return s.split('_', 1)[-1]
class ServerConfig:
    def __init__(self, low_res, high_res, low_qp, high_qp, bsize,
                 h_thres, l_thres, max_obj_size, min_obj_size,
                 tracker_length, boundary, intersection_threshold,
                 tracking_threshold, suppression_threshold, simulation,
                 rpn_enlarge_ratio, prune_score, objfilter_iou, size_obj):
        self.low_resolution = low_res
        self.high_resolution = high_res
        self.low_qp = low_qp
        self.high_qp = high_qp
        self.batch_size = bsize
        self.high_threshold = h_thres
        self.low_threshold = l_thres
        self.max_object_size = max_obj_size
        self.min_object_size = min_obj_size
        self.tracker_length = tracker_length
        self.boundary = boundary
        self.intersection_threshold = intersection_threshold
        self.simulation = simulation
        self.tracking_threshold = tracking_threshold
        self.suppression_threshold = suppression_threshold
        self.rpn_enlarge_ratio = rpn_enlarge_ratio
        self.prune_score = prune_score
        self.objfilter_iou = objfilter_iou
        self.size_obj = size_obj


class Region:
    def __init__(self, fid, x, y, w, h, conf, label, resolution,
                 origin="generic",feature=None):
        self.fid = int(fid)
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.conf = float(conf)
        self.label = label
        self.resolution = float(resolution)
        self.origin = origin
        self.feature=feature

    @staticmethod
    def convert_from_server_response(r, res, phase):
        return Region(r[0], r[1], r[2], r[3], r[4], r[5], r[6], res, phase)

    def __str__(self):
        string_rep = (f"{self.fid}, {self.x:0.3f}, {self.y:0.3f}, "
                      f"{self.w:0.3f}, {self.h:0.3f}, {self.conf:0.3f}, "
                      f"{self.label}, {self.origin}")
        return string_rep

    def is_same(self, region_to_check, threshold=0.5):
        if (self.fid != region_to_check.fid or
                ((self.label != "-1" and region_to_check.label != "-1") and
                 (self.label != region_to_check.label))):
            return False


        if calc_iou(self, region_to_check) > threshold:
            return True
        else:
            return False

    def enlarge(self, ratio):
        x_min = max(self.x - self.w * ratio, 0.0)
        y_min = max(self.y - self.h * ratio, 0.0)
        x_max = min(self.x + self.w * (1 + ratio), 1.0)
        y_max = min(self.y + self.h * (1 + ratio), 1.0)
        self.x = x_min
        self.y = y_min
        self.w = x_max - x_min
        self.h = y_max - y_min

    def copy(self):
        return Region(self.fid, self.x, self.y, self.w, self.h, self.conf,
                      self.label, self.resolution, self.origin)
    def calc_iou(self,region1, region2):
        x1_min = region1.x
        y1_min = region1.y
        x1_max = region1.x + region1.w
        y1_max = region1.y + region1.h

        x2_min = region2.x
        y2_min = region2.y
        x2_max = region2.x + region2.w
        y2_max = region2.y + region2.h

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)


        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)


        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area


        if union_area == 0:
            return 0.0
        else:
            return inter_area / union_area
    @staticmethod
    def merge_regions(regions, iou_threshold=0.7):
        merged_regions = []
        used = [False] * len(regions)

        for i in range(len(regions)):
            if used[i]:
                continue

            current_region = regions[i]
            for j in range(i + 1, len(regions)):
                if used[j]:
                    continue

                candidate_region = regions[j]


                if calc_iou(current_region, candidate_region) > iou_threshold:

                    x_min = min(current_region.x, candidate_region.x)
                    y_min = min(current_region.y, candidate_region.y)
                    x_max = max(current_region.x + current_region.w, candidate_region.x + candidate_region.w)
                    y_max = max(current_region.y + current_region.h, candidate_region.y + candidate_region.h)


                    current_region.x = x_min
                    current_region.y = y_min
                    current_region.w = x_max - x_min
                    current_region.h = y_max - y_min


                    used[j] = True


            merged_regions.append(current_region)

        return merged_regions

class Results:
    def __init__(self):
        self.regions = []
        self.regions_dict = {}

    def __len__(self):
        return len(self.regions)

    def results_high_len(self, threshold):
        count = 0
        for r in self.regions:
            if r.conf > threshold:
                count += 1
        return count

    def is_dup(self, result_to_add, threshold=0.5):

        if result_to_add.fid not in self.regions_dict:
            return None

        max_conf = -1
        max_conf_result = None
        for existing_result in self.regions_dict[result_to_add.fid]:
            if existing_result.is_same(result_to_add, threshold):
                if existing_result.conf > max_conf:
                    max_conf = existing_result.conf
                    max_conf_result = existing_result
        return max_conf_result

    def combine_results(self, additional_results, threshold=0.5):
        for result_to_add in additional_results.regions:
            self.add_single_result(result_to_add, threshold)

    def add_single_result(self, region_to_add, threshold=0.5):
        if threshold == 1:
            self.append(region_to_add)
            return
        dup_region = self.is_dup(region_to_add, threshold)
        if (not dup_region or
                ("tracking" in region_to_add.origin and
                 "tracking" in dup_region.origin)):
            self.regions.append(region_to_add)
            if region_to_add.fid not in self.regions_dict:
                self.regions_dict[region_to_add.fid] = []
            self.regions_dict[region_to_add.fid].append(region_to_add)
        else:
            final_object = None
            if dup_region.origin == region_to_add.origin:
                final_object = max([region_to_add, dup_region],
                                   key=lambda r: r.conf)
            elif ("low" in dup_region.origin and
                  "high" in region_to_add.origin):
                final_object = region_to_add
            elif ("high" in dup_region.origin and
                  "low" in region_to_add.origin):
                final_object = dup_region
            dup_region.x = final_object.x
            dup_region.y = final_object.y
            dup_region.w = final_object.w
            dup_region.h = final_object.h
            dup_region.conf = final_object.conf
            dup_region.origin = final_object.origin
            dup_region.feature=None

    def suppress(self, threshold=0.5):
        new_regions_list = []
        while len(self.regions) > 0:
            max_conf_obj = max(self.regions, key=lambda e: e.conf)
            new_regions_list.append(max_conf_obj)
            self.remove(max_conf_obj)
            objs_to_remove = []
            for r in self.regions:
                if r.fid != max_conf_obj.fid:
                    continue
                if calc_iou(r, max_conf_obj) > threshold:
                    objs_to_remove.append(r)
            for r in objs_to_remove:
                self.remove(r)
        new_regions_list.sort(key=lambda e: e.fid)
        for r in new_regions_list:
            self.append(r)

    def append(self, region_to_add):
        self.regions.append(region_to_add)
        if region_to_add.fid not in self.regions_dict:
            self.regions_dict[region_to_add.fid] = []
        self.regions_dict[region_to_add.fid].append(region_to_add)

    def remove(self, region_to_remove):
        self.regions_dict[region_to_remove.fid].remove(region_to_remove)
        self.regions.remove(region_to_remove)


    def fill_gaps(self, number_of_frames):
        if len(self.regions) == 0:
            return
        results_to_add = Results()
        max_resolution = max([e.resolution for e in self.regions])
        fids_in_results = [e.fid for e in self.regions]
        for i in range(number_of_frames):
            if i not in fids_in_results:
                results_to_add.regions.append(Region(i, 0, 0, 0, 0,
                                                     0.1, "no obj",
                                                     max_resolution))
        self.combine_results(results_to_add)
        self.regions.sort(key=lambda r: r.fid)

    def write_results_txt(self, fname):
        results_file = open(fname, "w")
        for region in self.regions:
            # prepare the string to write
            str_to_write = (f"{region.fid},{region.x},{region.y},"
                            f"{region.w},{region.h},"
                            f"{region.label},{region.conf},"
                            f"{region.resolution},{region.origin}\n")
            results_file.write(str_to_write)
        results_file.close()

    def write_results_csv(self, fname):
        results_files = open(fname, "w")
        csv_writer = csv.writer(results_files)
        for region in self.regions:
            row = [region.fid, region.x, region.y,
                   region.w, region.h,
                   region.label, region.conf,
                   region.resolution, region.origin]
            csv_writer.writerow(row)
        results_files.close()

    def write(self, fname):
        if re.match(r"\w+[.]csv\Z", fname):
            self.write_results_csv(fname)
        else:
            self.write_results_txt(fname)


def to_graph(l):
    G = networkx.Graph()
    for part in l:

        G.add_nodes_from(part)

        G.add_edges_from(to_edges(part))
    return G


def to_edges(l):

    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current


def filter_bbox_group(bb1, bb2, iou_threshold):
    if calc_iou(bb1, bb2) > iou_threshold and bb1.label == bb2.label:
        return True
    else:
        return False


def overlap(bb1, bb2):

    x_left = max(bb1.x, bb2.x)
    y_top = max(bb1.y, bb2.y)
    x_right = min(bb1.x+bb1.w, bb2.x+bb2.w)
    y_bottom = min(bb1.y+bb1.h, bb2.y+bb2.h)

    if x_right < x_left or y_bottom < y_top:
        return False
    else:
        return True


def pairwise_overlap_indexing_list(single_result_frame, iou_threshold):
    pointwise = [[i] for i in range(len(single_result_frame))]
    pairwise = [[i, j] for i, x in enumerate(single_result_frame)
                for j, y in enumerate(single_result_frame)
                if i != j if filter_bbox_group(x, y, iou_threshold)]
    return pointwise + pairwise


def simple_merge(single_result_frame, index_to_merge):

    bbox_large = []
    for i in index_to_merge:
        i2np = np.array([j for j in i])
        left = min(np.array(single_result_frame)[i2np], key=lambda x: x.x)
        top = min(np.array(single_result_frame)[i2np], key=lambda x: x.y)
        right = max(
            np.array(single_result_frame)[i2np], key=lambda x: x.x + x.w)
        bottom = max(
            np.array(single_result_frame)[i2np], key=lambda x: x.y + x.h)

        fid, x, y, w, h, conf, label, resolution, origin = (
            left.fid, left.x, top.y, right.x + right.w - left.x,
            bottom.y + bottom.h - top.y, left.conf, left.label,
            left.resolution, left.origin)
        single_merged_region = Region(fid, x, y, w, h, conf,
                                      label, resolution, origin)
        bbox_large.append(single_merged_region)
    return bbox_large


def merge_boxes_in_results(results_dict, min_conf_threshold, iou_threshold):
    final_results = Results()


    for _, regions in results_dict.items():
        to_remove = []
        for r in regions:
            if r.conf < min_conf_threshold:
                to_remove.append(r)
        for r in to_remove:
            regions.remove(r)

    for fid, regions in results_dict.items():
        overlap_pairwise_list = pairwise_overlap_indexing_list(
            regions, iou_threshold)
        overlap_graph = to_graph(overlap_pairwise_list)
        grouped_bbox_idx = [c for c in sorted(
            connected_components(overlap_graph), key=len, reverse=True)]
        merged_regions = simple_merge(regions, grouped_bbox_idx)
        for r in merged_regions:
            final_results.append(r)
    return final_results


def read_results_csv_dict(fname):

    results_dict = {}

    rows = []
    with open(fname) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            rows.append(row)

    for row in rows:
        fid = int(row[0])
        x, y, w, h = [float(e) for e in row[1:5]]
        conf = float(row[6])
        label = row[5]
        resolution = float(row[7])
        origin = float(row[8])

        region = Region(fid, x, y, w, h, conf, label, resolution, origin)

        if fid not in results_dict:
            results_dict[fid] = []

        if label != "no obj":
            results_dict[fid].append(region)

    return results_dict


def read_results_txt_dict(fname):

    results_dict = {}

    with open(fname, "r") as f:
        lines = f.readlines()
        f.close()

    for line in lines:
        line = line.split(",")
        fid = int(line[0])
        x, y, w, h = [float(e) for e in line[1:5]]
        conf = float(line[6])
        label = line[5]
        resolution = float(line[7])
        origin = "generic"
        if len(line) > 8:
            origin = line[8].strip()
        single_result = Region(fid, x, y, w, h, conf, label,
                               resolution, origin.rstrip())

        if fid not in results_dict:
            results_dict[fid] = []

        if label != "no obj":
            results_dict[fid].append(single_result)

    return results_dict


def read_results_dict(fname):
    # TODO: Need to implement a CSV function
    if re.match(r"\w+[.]csv\Z", fname):
        return read_results_csv_dict(fname)
    else:
        return read_results_txt_dict(fname)


def calc_intersection_area(a, b):
    to = max(a.y, b.y)
    le = max(a.x, b.x)
    bo = min(a.y + a.h, b.y + b.h)
    ri = min(a.x + a.w, b.x + b.w)

    w = max(0, ri - le)
    h = max(0, bo - to)

    return w * h


def calc_area(a):
    w = max(0, a.w)
    h = max(0, a.h)

    return w * h


def calc_iou(a, b):
    intersection_area = calc_intersection_area(a, b)
    union_area = calc_area(a) + calc_area(b) - intersection_area


    if union_area == 0:
        return 0.0

    return intersection_area / union_area

def get_interval_area(width, all_yes):
    area = 0
    for y1, y2 in all_yes:
        area += (y2 - y1) * width
    return area


def insert_range_y(all_yes, y1, y2):
    ranges_length = len(all_yes)
    idx = 0
    while idx < ranges_length:
        if not (y1 > all_yes[idx][1] or all_yes[idx][0] > y2):
            # Overlapping
            y1 = min(y1, all_yes[idx][0])
            y2 = max(y2, all_yes[idx][1])
            del all_yes[idx]
            ranges_length = len(all_yes)
        else:
            idx += 1

    all_yes.append((y1, y2))


def get_y_ranges(regions, j, x1, x2):
    all_yes = []
    while j < len(regions):
        if (x1 < (regions[j].x + regions[j].w) and
                x2 > regions[j].x):
            y1 = regions[j].y
            y2 = regions[j].y + regions[j].h
            insert_range_y(all_yes, y1, y2)
        j += 1
    return all_yes


def compute_area_of_frame(regions):

    regions.sort(key=lambda r: r.x + r.w)

    all_xes = []
    for r in regions:
        all_xes.append(r.x)
        all_xes.append(r.x + r.w)
    all_xes.sort()

    area = 0
    j = 0
    for i in range(len(all_xes) - 1):
        x1 = all_xes[i]
        x2 = all_xes[i + 1]

        if x1 < x2:

            while (regions[j].x + regions[j].w) < x1:
                j += 1
            all_yes = get_y_ranges(regions, j, x1, x2)
            area += get_interval_area(x2 - x1, all_yes)

    return area


def compute_area_of_regions(results):
    if len(results.regions) == 0:
        return 0

    min_frame = min([r.fid for r in results.regions])
    max_frame = max([r.fid for r in results.regions])

    total_area = 0
    for fid in range(min_frame, max_frame + 1):
        regions_for_frame = [r for r in results.regions if r.fid == fid]
        total_area += compute_area_of_frame(regions_for_frame)

    return total_area


def compress_and_get_size(images_path, start_id, end_id, qp,
                          enforce_iframes=False, resolution=None):
    image_extension = get_image_extension(images_path)
    number_of_frames = end_id - start_id
    encoded_vid_path = os.path.join(images_path, "temp.mp4")
    if resolution and enforce_iframes:
        scale = f"scale=trunc(iw*{resolution}/2)*2:trunc(ih*{resolution}/2)*2"
        if not qp:
            encoding_result = subprocess.run(["ffmpeg", "-y",
                                              "-loglevel", "error",
                                              "-start_number", str(start_id),
                                              '-i', f"{images_path}/%08d.{image_extension}",
                                              "-vcodec", "libx264", "-g", "15",
                                              "-keyint_min", "15",
                                              "-pix_fmt", "yuv420p",
                                              "-vf", scale,
                                              "-frames:v",
                                              str(number_of_frames),
                                              encoded_vid_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             universal_newlines=True)
        else:
            encoding_result = subprocess.run(["ffmpeg", "-y",
                                              "-loglevel", "error",
                                              "-start_number", str(start_id),
                                              '-i', f"{images_path}/%08d.{image_extension}",
                                              "-vcodec", "libx264",
                                              "-g", "15",
                                              "-keyint_min", "15",
                                              "-qp", f"{qp}",
                                              "-pix_fmt", "yuv420p",
                                              "-vf", scale,
                                              "-frames:v",
                                              str(number_of_frames),
                                              encoded_vid_path],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             universal_newlines=True)
    else:
        encoding_result = subprocess.run(["ffmpeg", "-y",
                                          "-start_number", str(start_id),
                                          "-i", f"{images_path}/%08d.{image_extension}",
                                          "-loglevel", "error",
                                          "-vcodec", "libx264",
                                          "-pix_fmt", "yuv420p", "-crf", "23",
                                          encoded_vid_path],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         universal_newlines=True)

    size = 0
    if encoding_result.returncode != 0:
        # Encoding failed
        print("ENCODING FAILED")
        print(encoding_result.stdout)
        print(encoding_result.stderr)
        exit()
    else:
        size = os.path.getsize(encoded_vid_path)

    return size


def extract_images_from_video(images_path, req_regions):
    if not os.path.isdir(images_path):
        return
    image_extension=get_image_extension(images_path)
    for fname in os.listdir(images_path):
        if "image_extension" not in fname:
            continue
        else:
            os.remove(os.path.join(images_path, fname))
    encoded_vid_path = os.path.join(images_path, "temp.mp4")
    extacted_images_path = os.path.join(images_path, f"%08d.{image_extension}")
    decoding_result = subprocess.run(["ffmpeg", "-y",
                                      "-i", encoded_vid_path,
                                      "-pix_fmt", "yuvj420p",
                                      "-g", "8", "-q:v", "2",
                                      "-vsync", "0", "-start_number", "0",
                                      extacted_images_path],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
    if decoding_result.returncode != 0:
        print("DECODING FAILED")
        print(decoding_result.stdout)
        print(decoding_result.stderr)
        exit()

    fnames = sorted(
        [os.path.join(images_path, name)
         for name in os.listdir(images_path) if f"{image_extension}" in name])
    fids = sorted(list(set([r.fid for r in req_regions.regions])))
    fids_mapping = zip(fids, fnames)
    for fname in fnames:
        # Rename temporarily
        os.rename(fname, f"{fname}_temp")

    for fid, fname in fids_mapping:
        os.rename(os.path.join(f"{fname}_temp"),
                  os.path.join(images_path, f"{str(fid).zfill(8)}.{image_extension}"))
def cache_images(images_direc, image_extension, region_list):
    image_cache = {}
    for region in region_list:
        if region.fid not in image_cache:
            image_path = os.path.join(images_direc, f"{str(region.fid).zfill(8)}.{image_extension}")
            image_cache[region.fid] = cv.imread(image_path)
    return image_cache


def crop_images(results, vid_name, images_direc, image_extension, resolution=None):

    image_cache = cache_images(images_direc, image_extension, results.regions)
    cropped_images = {}

    for region in results.regions:
        cached_image = (region.fid, image_cache[region.fid])


        if region.x == 0 and region.y == 0 and region.w == 1 and region.h == 1:
            cropped_images[region.fid] = cached_image[1]
            continue

        width = cached_image[1].shape[1]
        height = cached_image[1].shape[0]
        x0 = int(region.x * width)
        y0 = int(region.y * height)
        x1 = int((region.w * width) + x0 - 1)
        y1 = int((region.h * height) + y0 - 1)


        if region.fid not in cropped_images:
            cropped_images[region.fid] = np.zeros_like(cached_image[1])

        cropped_image = cropped_images[region.fid]
        cropped_image[y0:y1, x0:x1, :] = cached_image[1][y0:y1, x0:x1, :]


    os.makedirs(vid_name, exist_ok=True)
    frames_count = len(cropped_images)
    frames = sorted(cropped_images.items(), key=lambda e: e[0])
    for idx, (_, frame) in enumerate(frames):
        if resolution:
            w = int(frame.shape[1] * resolution)
            h = int(frame.shape[0] * resolution)
            frame = cv.resize(frame, (w, h), fx=0, fy=0, interpolation=cv.INTER_CUBIC)

        if image_extension == "png":
            cv.imwrite(os.path.join(vid_name, f"{str(idx).zfill(8)}.{image_extension}"), frame,
                       [cv.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv.imwrite(os.path.join(vid_name, f"{str(idx).zfill(8)}.{image_extension}"), frame,
                       [cv.IMWRITE_JPEG_QUALITY, 100])

    return frames_count



def merge_images(cropped_images_direc, low_images_direc, req_regions):
    images = {}
    image_extension=get_image_extension(cropped_images_direc)
    for fname in os.listdir(cropped_images_direc):
        if  not fname.endswith(image_extension):
            continue
        fid = int(fname.split(".")[0])

        high_image = cv.imread(os.path.join(cropped_images_direc, fname))
        width = high_image.shape[1]
        height = high_image.shape[0]


        low_image = cv.imread(os.path.join(low_images_direc, fname))

        enlarged_image = cv.resize(low_image, (width, height), fx=0, fy=0,
                                   interpolation=cv.INTER_CUBIC)

        for r in req_regions.regions:
            if fid != r.fid:
                continue
            x0 = int(r.x * width)
            y0 = int(r.y * height)
            x1 = int((r.w * width) + x0 - 1)
            y1 = int((r.h * height) + y0 - 1)

            enlarged_image[y0:y1, x0:x1, :] = high_image[y0:y1, x0:x1, :]
        image_extension = get_image_extension(cropped_images_direc)
        if image_extension=="png":
            cv.imwrite(os.path.join(cropped_images_direc, fname), enlarged_image,
                       [cv.IMWRITE_PNG_COMPRESSION, 0])
        else:
            cv.imwrite(os.path.join(cropped_images_direc, fname), enlarged_image,
                       [cv.IMWRITE_JPEG_QUALITY, 100])
        images[fid] = enlarged_image
    return images


def compute_regions_size(results, vid_name, images_direc, resolution, qp,
                         enforce_iframes, estimate_banwidth=True):
    if estimate_banwidth:

        vid_name = f"{vid_name}-cropped"
        image_extension=get_image_extension(images_direc)


        frames_count = crop_images(results, vid_name, images_direc, image_extension, resolution)

        size = compress_and_get_size(vid_name, 0, frames_count, qp=qp, enforce_iframes=enforce_iframes, resolution=1)

        pixel_size = compute_area_of_regions(results)
        return size, pixel_size
    else:
        size = compute_area_of_regions(results)

        return size


def cleanup(vid_name, debug_mode=False, start_id=None, end_id=None):
    if not os.path.isdir(vid_name + "-cropped"):
        return

    if not debug_mode:
        shutil.rmtree(vid_name + "-base-phase-cropped")
        shutil.rmtree(vid_name + "-cropped")
    else:
        if start_id is None or end_id is None:
            print("Need start_fid and end_fid for debugging mode")
            exit()
        os.makedirs("debugging", exist_ok=True)
        leaf_direc = vid_name.split("/")[-1] + "-cropped"
        shutil.move(vid_name + "-cropped", "debugging")
        shutil.move(os.path.join("debugging", leaf_direc),
                    os.path.join("debugging",
                                 f"{leaf_direc}-{start_id}-{end_id}"),
                    copy_function=os.rename)


def get_size_from_mpeg_results(results_log_path, images_path, resolution):
    with open(results_log_path, "r") as f:
        lines = f.readlines()
    lines = [line for line in lines if line.rstrip().lstrip() != ""]
    num_frames = get_images_length(images_path)

    bandwidth = 0
    for idx, line in enumerate(lines):
        if f"RES {resolution}" in line:
            bandwidth = float(lines[idx + 2])
            break
    size = bandwidth * 1024.0 * (num_frames / 10.0)
    return size


def filter_results(bboxes, gt_flag, gt_confid_thresh, mpeg_confid_thresh,
                   max_area_thresh_gt, max_area_thresh_mpeg):
    relevant_classes = ["vehicle"]
    if gt_flag:
        confid_thresh = gt_confid_thresh
        max_area_thresh = max_area_thresh_gt

    else:
        confid_thresh = mpeg_confid_thresh
        max_area_thresh = max_area_thresh_mpeg

    result = []
    for b in bboxes:
        b = b.x, b.y, b.w, b.h, b.label, b.conf
        (x, y, w, h, label, confid) = b
        if (confid >= confid_thresh and w*h <= max_area_thresh and
                label in relevant_classes):
            result.append(b)
    return result


def iou(b1, b2):
    (x1, y1, w1, h1, label1, confid1) = b1
    (x2, y2, w2, h2, label2, confid2) = b2
    x3 = max(x1, x2)
    y3 = max(y1, y2)
    x4 = min(x1+w1, x2+w2)
    y4 = min(y1+h1, y2+h2)
    if x3 > x4 or y3 > y4:
        return 0
    else:
        overlap = (x4-x3)*(y4-y3)
        return overlap/(w1*h1+w2*h2-overlap)


def evaluate(max_fid, map_dd, map_gt, gt_confid_thresh, mpeg_confid_thresh,
             max_area_thresh_gt, max_area_thresh_mpeg, iou_thresh=0.3):
    tp_list = []
    fp_list = []
    fn_list = []
    count_list = []
    for fid in range(max_fid+1):
        if fid not in map_dd:
            print(f"Warning: fid {fid} not found in map_dd")
            bboxes_dd = []
        else:
            bboxes_dd = map_dd[fid]

        if fid not in map_gt:
            print(f"Warning: fid {fid} not found in map_gt")
            bboxes_gt = []
        else:
            bboxes_gt = map_gt[fid]

        bboxes_dd = filter_results(
            bboxes_dd, gt_flag=False, gt_confid_thresh=gt_confid_thresh,
            mpeg_confid_thresh=mpeg_confid_thresh,
            max_area_thresh_gt=max_area_thresh_gt,
            max_area_thresh_mpeg=max_area_thresh_mpeg)
        bboxes_gt = filter_results(
            bboxes_gt, gt_flag=True, gt_confid_thresh=gt_confid_thresh,
            mpeg_confid_thresh=mpeg_confid_thresh,
            max_area_thresh_gt=max_area_thresh_gt,
            max_area_thresh_mpeg=max_area_thresh_mpeg)
        tp = 0
        fp = 0
        fn = 0
        count = 0
        for b_dd in bboxes_dd:
            found = False
            for b_gt in bboxes_gt:
                if iou(b_dd, b_gt) >= iou_thresh:
                    found = True
                    break
            if found:
                tp += 1
            else:
                fp += 1
        for b_gt in bboxes_gt:
            found = False
            for b_dd in bboxes_dd:
                if iou(b_dd, b_gt) >= iou_thresh:
                    found = True
                    break
            if not found:
                fn += 1
            else:
                count += 1
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        count_list.append(count)
    tp = sum(tp_list)
    fp = sum(fp_list)
    fn = sum(fn_list)
    count = sum(count_list)

    precision = round(tp / (tp + fp), 3) if (tp + fp) != 0 else 0
    recall = round(tp / (tp + fn), 3) if (tp + fn) != 0 else 0
    f1 = round(2.0 * tp / (2.0 * tp + fp + fn), 3) if (2.0 * tp + fp + fn) != 0 else 0

    return (tp, fp, fn, count, precision, recall, f1)



def write_stats_txt(fname, vid_name, config, f1, stats, bw, frames_count, mode, lt):
    header = ("video-name,low-resolution,high-resolution,low_qp,high_qp,"
              "batch-size,low-threshold,high-threshold,"
              "tracker-length,TP,FP,FN,F1,"
              "low-size,high-size,total-size,frames,mode,"
              "transmission_time,inference_reuse_time,"
              "roi_prediction_time,inference_sharing_time,total_time")
    stats = (f"{vid_name},{config.low_resolution},{config.high_resolution},"
             f"{config.low_qp},{config.high_qp},{config.batch_size},"
             f"{config.low_threshold},{config.high_threshold},"
             f"{config.tracker_length},{stats[0]},{stats[1]},{stats[2]},"
             f"{f1},{bw[0]},{bw[1]},{bw[0] + bw[1]},"
             f"{frames_count},{mode},"
             f"{lt['transmission']},{lt['inference_reuse']},"
             f"{lt['roi_prediction']},{lt['inference_sharing']},{lt['total_time']}")

    if not os.path.isfile(fname):
        str_to_write = f"{header}\n{stats}\n"
    else:
        str_to_write = f"{stats}\n"

    with open(fname, "a") as f:
        f.write(str_to_write)

def write_stats_csv(fname, vid_name, config, f1, stats, bw, frames_count, mode, lt):
    header = ("video-name,low-resolution,high-resolution,low-qp,high-qp,"
              "batch-size,low-threshold,high-threshold,"
              "tracker-length,TP,FP,FN,F1,"
              "low-size,high-size,total-size,frames,mode,"
              "transmission_time,inference_reuse_time,"
              "roi_prediction_time,inference_sharing_time,total_time").split(",")
    stats = (f"{vid_name},{config.low_resolution},{config.high_resolution},"
             f"{config.low_qp},{config.high_qp},{config.batch_size},"
             f"{config.low_threshold},{config.high_threshold},"
             f"{config.tracker_length},{stats[0]},{stats[1]},{stats[2]},"
             f"{f1},{bw[0]},{bw[1]},{bw[0] + bw[1]},"
             f"{frames_count},{mode},"
             f"{lt['transmission']},{lt['inference_reuse']},"
             f"{lt['roi_prediction']},{lt['inference_sharing']},{lt['total_time']}").split(",")

    with open(fname, "a") as results_files:
        csv_writer = csv.writer(results_files)
        if os.path.getsize(fname) == 0:

            csv_writer.writerow(header)
        csv_writer.writerow(stats)

def write_stats(fname, vid_name, config, f1, stats, bw, frames_count, mode, lt):

    if re.match(r"\w+[.]csv\Z", fname):
        write_stats_csv(fname, vid_name, config, f1, stats, bw, frames_count, mode, lt)
    else:
        write_stats_txt(fname, vid_name, config, f1, stats, bw, frames_count, mode, lt)

def visualize_regions(results, images_direc,
                      low_conf=0.0, high_conf=1.0,
                      label="debugging"):
    image_extension = get_image_extension(images_direc)
    idx = 0
    fids = sorted(list(set([r.fid for r in results.regions])))
    while idx < len(fids):
        image_np = cv.imread(
            os.path.join(images_direc, f"{str(fids[idx]).zfill(8)}.{image_extension}"))
        width = image_np.shape[1]
        height = image_np.shape[0]
        regions = [r for r in results.regions if r.fid == fids[idx]]
        for r in regions:
            if r.conf < low_conf or r.conf > high_conf:
                continue
            x0 = int(r.x * width)
            y0 = int(r.y * height)
            x1 = int(r.w * width + x0)
            y1 = int(r.h * height + y0)
            cv.rectangle(image_np, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv.putText(image_np, f"{fids[idx]}", (10, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv.imshow(label, image_np)
        key = cv.waitKey()
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("k"):
            idx -= 2

        idx += 1
    cv.destroyAllWindows()


def visualize_single_regions(region, images_direc, label="debugging"):
    image_extension = get_image_extension(images_direc)
    image_path = os.path.join(images_direc, f"{str(region.fid).zfill(8)}.{image_extension}")
    image_np = cv.imread(image_path)
    width = image_np.shape[1]
    height = image_np.shape[0]

    x0 = int(region.x * width)
    y0 = int(region.y * height)
    x1 = int((region.w * width) + x0)
    y1 = int((region.h * height) + y0)

    cv.rectangle(image_np, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv.putText(image_np, f"{region.fid}, {region.label}, {region.conf:0.2f}, "
               f"{region.w * region.h}",
               (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv.imshow(label, image_np)
    cv.waitKey()
    cv.destroyAllWindows()
