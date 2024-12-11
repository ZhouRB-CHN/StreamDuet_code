import time
import pandas as pd

import os
import cv2
import numpy as np
from sd_utils import Results, Region

from concurrent.futures import ThreadPoolExecutor
class RoICache:
    def __init__(self, time_window, conf_threshold, relevant_classes, residual_threshold, lowres_threshold):
        self.time_window = time_window
        self.conf_threshold = conf_threshold
        self.relevant_classes = relevant_classes
        self.residual_threshold = residual_threshold
        self.lowres_threshold = lowres_threshold
        self.sift = cv2.SIFT_create()

        self.memory_cache = {}

    def _current_time(self):
        return time.time()

    def add_results(self, frame_id, results, frame_image):

        if frame_id not in self.memory_cache:

            categorized_results = self._categorize_results(results)
            self.memory_cache[frame_id] = {
                'results': categorized_results,
                'image': frame_image,
                'timestamp': self._current_time()
            }
        self._cleanup()

    def get_regions_of_interest(self, current_frame_id, current_frame_image):
        prev_frame = self._get_previous_frame(current_frame_id)


        if prev_frame is None:
            req_regions = Results()
            background_regions = Results()
            current_frame_results = Results()
            req_regions.append(Region(current_frame_id, 0, 0, 1, 1, 1.0, 2, self.lowres_threshold))
            return req_regions, background_regions, current_frame_results

        roi_regions, background_regions, current_frame_results = self._process_cached_results(
            current_frame_id, current_frame_image)


        roi_regions.combine_results(current_frame_results)

        return roi_regions, background_regions, current_frame_results

    def _process_cached_results(self, current_frame_id, current_frame_image):
        roi_regions = Results()
        background_regions = Results()
        current_frame_results = Results()

        for frame_id, data in self.memory_cache.items():
            if frame_id < current_frame_id - self.time_window:
                continue
            categorized_results = data['results']
            for category, results in categorized_results.items():




                predicted_bboxes = self.predict_bounding_boxes(data['image'], current_frame_image, results,current_frame_id)
                predicted_bboxes.suppress()

                self._categorize_predicted_bboxes(predicted_bboxes, category,current_frame_results, roi_regions, background_regions,current_frame_id)
        current_frame_results.suppress()

        return roi_regions, background_regions, current_frame_results

    def _categorize_predicted_bboxes(self, predicted_bboxes, category, current_frame_results,roi_regions, background_regions,current_frame_id):
        for predicted_bbox in predicted_bboxes.regions:
            predicted_bbox.frame_id=current_frame_id
            if category == 'high_conf_target' and predicted_bbox.conf > self.conf_threshold:
                current_frame_results.add_single_result(predicted_bbox, self.conf_threshold)
            elif category == 'high_conf_non_target' and predicted_bbox.conf > self.conf_threshold:
                background_regions.add_single_result(predicted_bbox, self.conf_threshold)
            elif category == 'low_conf':
                if predicted_bbox.conf > self.conf_threshold:
                    roi_regions.add_single_result(predicted_bbox, self.conf_threshold)
                else:
                    roi_regions.add_single_result(self._expand_bbox(predicted_bbox), self.conf_threshold)

    def predict_bounding_boxes(self, prev_frame, current_frame_image, results, current_frame_id, threshold=30,
                               expand_pixels=10):

        gpu_prev_frame = cv2.cuda_GpuMat()
        gpu_prev_frame.upload(prev_frame)

        gpu_current_frame = cv2.cuda_GpuMat()
        gpu_current_frame.upload(current_frame_image)


        gpu_prev_frame_gray = cv2.cuda.cvtColor(gpu_prev_frame, cv2.COLOR_BGR2GRAY)
        gpu_current_frame_gray = cv2.cuda.cvtColor(gpu_current_frame, cv2.COLOR_BGR2GRAY)


        if gpu_prev_frame_gray.size() != gpu_current_frame_gray.size():

            raise ValueError("Previous frame and current frame have different sizes.")


        gpu_flow = cv2.cuda.calcOpticalFlowFarneback(
            gpu_prev_frame_gray, gpu_current_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )


        flow = gpu_flow.download()

        height, width = gpu_prev_frame_gray.rows, gpu_prev_frame_gray.cols

        predicted_bboxes = Results()

        for bbox in results.regions:
            x_min = int(bbox.x * width)
            y_min = int(bbox.y * height)
            x_max = int((bbox.x + bbox.w) * width)
            y_max = int((bbox.y + bbox.h) * height)


            flow_x = np.mean(flow[y_min:y_max, x_min:x_max, 0])
            flow_y = np.mean(flow[y_min:y_max, x_min:x_max, 1])


            new_x = bbox.x + flow_x / width
            new_y = bbox.y + flow_y / height


            gpu_residual_block = cv2.cuda.abs(
                gpu_current_frame_gray[y_min:y_max, x_min:x_max] - gpu_prev_frame_gray[y_min:y_max, x_min:x_max])


            residual_block = gpu_residual_block.download()

            mean_residual = np.mean(residual_block)

            if mean_residual > threshold:

                gpu_residual_x = cv2.cuda.mean(gpu_residual_block * flow[y_min:y_max, x_min:x_max, 0])
                gpu_residual_y = cv2.cuda.mean(gpu_residual_block * flow[y_min:y_max, x_min:x_max, 1])


                residual_x = gpu_residual_x.download()
                residual_y = gpu_residual_y.download()

                new_x += residual_x / width
                new_y += residual_y / height


            x_min_new = max(int(new_x * width) - expand_pixels, 0)
            y_min_new = max(int(new_y * height) - expand_pixels, 0)
            x_max_new = min(int((new_x + bbox.w) * width) + expand_pixels, width)
            y_max_new = min(int((new_y + bbox.h) * height) + expand_pixels, height)


            new_bbox = Region(
                current_frame_id,
                x=x_min_new / width,
                y=y_min_new / height,
                w=(x_max_new - x_min_new) / width,
                h=(y_max_new - y_min_new) / height,
                conf=bbox.conf,
                label=bbox.label,
                resolution=bbox.resolution,
                origin="fronted_predicted"
            )
            predicted_bboxes.add_single_result(new_bbox)

        return predicted_bboxes


    def _expand_bbox(self, bbox):
        expanded_bbox = Region(
            bbox.fid,
            x=max(0, bbox.x - 0.01),
            y=max(0, bbox.y - 0.01),
            w=min(1, bbox.w*1.1),
            h=min(1, bbox.h*1.1),
            conf=bbox.conf,
            label=bbox.label,
            resolution=bbox.resolution,
            origin=bbox.origin
        )
        return expanded_bbox

    def _compute_moving_blocks(self, roi_regions, prev_frame, current_frame_image, current_frame_id):

        gpu_prev_frame = cv2.cuda_GpuMat()
        gpu_prev_frame.upload(prev_frame)

        gpu_current_frame = cv2.cuda_GpuMat()
        gpu_current_frame.upload(current_frame_image)

        gpu_prev_gray = cv2.cuda.cvtColor(gpu_prev_frame, cv2.COLOR_BGR2GRAY)
        gpu_curr_gray = cv2.cuda.cvtColor(gpu_current_frame, cv2.COLOR_BGR2GRAY)


        gpu_flow = cv2.cuda.calcOpticalFlowFarneback(
            gpu_prev_gray, gpu_curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )


        flow = gpu_flow.download()

        h, w = gpu_curr_gray.rows, gpu_curr_gray.cols
        block_size = 64


        for y in range(0, h, block_size):
            for x in range(0, w, block_size):

                y_end = min(y + block_size, h)
                x_end = min(x + block_size, w)
                flow_block = flow[y:y_end, x:x_end]
                mean_flow = np.mean(flow_block, axis=(0, 1))


                if np.linalg.norm(mean_flow) > self.residual_threshold:
                    roi_regions.add_single_result(
                        Region(current_frame_id, x / w, y / h, (x_end - x) / w, (y_end - y) / h, 0, 'moving_block', 1,
                               'motion'),
                        self.conf_threshold
                    )

    def _categorize_results(self, results):
        categorized_results = {
            'high_conf_target': Results(),
            'high_conf_non_target':  Results(),
            'low_conf':   Results()
        }

        for region in results:
            if region.conf > self.conf_threshold:
                if region.label in self.relevant_classes:
                    categorized_results['high_conf_target'].append(region)
                else:
                    categorized_results['high_conf_non_target'].append(region)
            else:
                categorized_results['low_conf'].append(region)
        return categorized_results

    def _cleanup(self):
        min_frame_id = max(self.memory_cache.keys()) - self.time_window
        keys_to_delete = [frame_id for frame_id in self.memory_cache if frame_id < min_frame_id]
        for frame_id in keys_to_delete:
            del self.memory_cache[frame_id]

    def _get_previous_frame(self, current_frame_id):

        previous_frame_ids = [frame_id for frame_id in self.memory_cache if frame_id < current_frame_id]

        if not previous_frame_ids:
            return None


        prev_frame_id = max(previous_frame_ids)

        return self.memory_cache[prev_frame_id]['image']

    def _crop_image(self, image, bbox):
        height, width = image.shape[:2]
        x_min = int(bbox.x * width)
        y_min = int(bbox.y * height)
        x_max = int((bbox.x + bbox.w) * width)
        y_max = int((bbox.y + bbox.h) * height)
        return image[y_min:y_max, x_min:x_max]

    def _match_features(self, query_image, cached_image):
        query_keypoints, query_descriptors = self.sift.detectAndCompute(query_image, None)
        cached_keypoints, cached_descriptors = self.sift.detectAndCompute(cached_image, None)

        if query_descriptors is None or cached_descriptors is None:
            return False

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(query_descriptors, cached_descriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return len(good_matches) > 10

    def draw_bbox_on_mask(self, mask, bbox):
        height, width = mask.shape[:2]
        x_min = int(bbox.x * width)
        y_min = int(bbox.y * height)
        x_max = int((bbox.x + bbox.w) * width)
        y_max = int((bbox.y + bbox.h) * height)
        mask[y_min:y_max, x_min:x_max] = 255
        return mask

    def is_region_in_mask(self, region, mask):
        height, width = mask.shape[:2]
        x_min = int(region.x * width)
        y_min = int(region.y * height)
        x_max = int((region.x + region.w) * width)
        y_max = int((region.y + region.h) * height)
        return np.any(mask[y_min:y_max, x_min:x_max] == 255)

    def process_frame(self, current_frame_id, current_frame_image):
        roi_regions, background_regions, current_frame_results = self.get_regions_of_interest(
            current_frame_id, current_frame_image)

        final_results = Results()
        final_results.combine_results(current_frame_results, self.conf_threshold)

        return roi_regions, final_results

    def update_cache(self, start_fid, end_fid, final_results, high_images_path):
        for fid in range(start_fid, end_fid):
            if fid in final_results.regions_dict:
                results= final_results.regions_dict[fid]
            else:
                results=[]
            if fid not in self.memory_cache:
                current_frame_image = cv2.imread(os.path.join(high_images_path, f"{fid:08d}.jpg"))
                self.add_results(fid, results, current_frame_image)
            else:
                self.add_results(fid, results, self.memory_cache[fid]['image'])
        self._cleanup()


    def compute_optical_flow_farneback(self,image1, image2):
        gray1 = self.ensure_grayscale(image1)
        gray2 = self.ensure_grayscale(image2)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion_vectors = np.dstack((flow[..., 0], flow[..., 1]))
        return motion_vectors

    def resize_image(self,image, target_size):
        return cv2.resize(self,image, target_size)

    def ensure_grayscale(self,image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def warp_image(self,img, flow, scale_factor):
        h, w = img.shape
        flow_map = np.zeros_like(flow)
        flow_map[..., 0] = np.meshgrid(np.arange(w), np.arange(h))[0]
        flow_map[..., 1] = np.meshgrid(np.arange(w), np.arange(h))[1]
        flow_map += flow * scale_factor
        flow_map = flow_map.astype(np.float32)
        warped_img = cv2.remap(img, flow_map, None, cv2.INTER_LINEAR)
        return warped_img

        flow_map = np.zeros_like(flow)
        flow_map[..., 0] = np.meshgrid(np.arange(w), np.arange(h))[0]
        flow_map[..., 1] = np.meshgrid(np.arange(w), np.arange(h))[1]
        flow_map += flow
        flow_map = flow_map.astype(np.float32)
        warped_img = cv2.remap(img, flow_map, None, cv2.INTER_LINEAR)
        return warped_img

    def highlight_large_errors(self, img, residual, threshold, block_size, region_size, current_frame_id, expand_pixels=5):
        if len(residual.shape) == 3:
            h, w, _ = residual.shape
        else:
            h, w = residual.shape

        mask = np.zeros_like(img)
        highlighted_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        regions = []


        for y in range(0, h, region_size):
            for x in range(0, w, region_size):
                region_has_large_error = False


                for by in range(y, min(y + region_size, h), block_size):
                    for bx in range(x, min(x + region_size, w), block_size):
                        block_h = min(block_size, h - by)
                        block_w = min(block_size, w - bx)

                        block = residual[by:by + block_h, bx:bx + block_w]
                        mean_error = np.mean(block)


                        if mean_error > threshold:
                            region_has_large_error = True
                            mask[by:by + block_h, bx:bx + block_w] = 255
                            overlay = highlighted_img[by:by + block_h, bx:bx + block_w]
                            highlighted_img[by:by + block_h, bx:bx + block_w] = cv2.addWeighted(overlay, 0.5,
                                                                                                np.full_like(overlay,
                                                                                                             [0, 0,
                                                                                                              255]),
                                                                                                0.5, 0)

                            break
                    if region_has_large_error:
                        break


                if region_has_large_error:

                    expanded_x = max(0, x - expand_pixels)
                    expanded_y = max(0, y - expand_pixels)
                    expanded_w = min(region_size + 2 * expand_pixels, w - expanded_x)
                    expanded_h = min(region_size + 2 * expand_pixels, h - expanded_y)


                    regions.append(Region(
                        current_frame_id,
                        expanded_x / w,
                        expanded_y / h,
                        expanded_w / w,
                        expanded_h / h,
                        1,
                        1
                    ))

        return highlighted_img, mask, regions





    def process_frames(self, current_img, current_frame_id, threshold=25, block_size=16,region_size=32):
        results = Results()



        frame_ids = sorted(self.memory_cache.keys())
        if len(frame_ids) < 3:
            return results



        prev_frame_id = frame_ids[-1]
        prev_prev_frame_id = frame_ids[-2]

        if prev_frame_id >= current_frame_id or prev_prev_frame_id >= current_frame_id:
            return results



        prev_prev_img = self.ensure_grayscale(self.memory_cache[prev_prev_frame_id]["image"])
        prev_img = self.ensure_grayscale(self.memory_cache[prev_frame_id]["image"])
        current_img = self.ensure_grayscale(current_img)




        if prev_img.shape != current_img.shape:
            prev_img = cv2.resize(prev_img, (current_img.shape[1], current_img.shape[0]))
        if prev_prev_img.shape != current_img.shape:
            prev_prev_img = cv2.resize(prev_prev_img, (current_img.shape[1], current_img.shape[0]))

        delta_t1 = prev_frame_id - prev_prev_frame_id
        delta_t2 = current_frame_id - prev_frame_id
        scale_factor = delta_t2 / delta_t1 if delta_t1 != 0 else 1




        flow = self.compute_optical_flow_farneback(prev_prev_img, prev_img)




        predicted_img = self.warp_image(prev_img, flow, scale_factor)



        if predicted_img.shape != current_img.shape:
            predicted_img = cv2.resize(predicted_img, (current_img.shape[1], current_img.shape[0]))



        residual = cv2.absdiff(current_img, predicted_img)





        highlighted_img, mask, regions = self.highlight_large_errors(
            current_img, residual, threshold, block_size,region_size, current_frame_id)



        for region in regions:
            region.resolution = current_img.shape[:2]
            results.add_single_result(region)




        return results


    def find_zero_motion_blocks_gpu(self, motion_vectors, block_size=16, zero_threshold=0.5):
        """
        GPU-optimized version of finding zero-motion blocks.
        """
        h, w, _ = motion_vectors.shape
        h_blocks = h // block_size
        w_blocks = w // block_size

        gpu_motion_vectors = cv2.cuda_GpuMat()
        gpu_motion_vectors.upload(motion_vectors)

        gpu_block_vectors = gpu_motion_vectors.reshape(h_blocks, block_size, w_blocks, block_size, 2)


        gpu_block_means = cv2.cuda.reduce(gpu_block_vectors, axis=(1, 3), op=cv2.REDUCE_AVG)

        gpu_magnitude = cv2.cuda.magnitude(gpu_block_means[..., 0], gpu_block_means[..., 1])

        gpu_zero_motion_mask = gpu_magnitude < zero_threshold

        zero_motion_mask = gpu_zero_motion_mask.download()


        zero_motion_blocks = np.argwhere(zero_motion_mask)

        zero_motion_blocks = [(x * block_size, y * block_size) for y, x in zero_motion_blocks]

        return zero_motion_blocks

    def find_zero_motion_blocks_optimized(self,motion_vectors, block_size=16, zero_threshold=0.5):
        h, w, _ = motion_vectors.shape
        h_blocks = h // block_size
        w_blocks = w // block_size

        block_vectors = motion_vectors[:h_blocks * block_size, :w_blocks * block_size, :].reshape(h_blocks, block_size,
                                                                                                  w_blocks, block_size,
                                                                                                  2)


        block_means = np.mean(block_vectors, axis=(1, 3))

        magnitude = np.sqrt(block_means[..., 0] ** 2 + block_means[..., 1] ** 2)


        zero_motion_mask = magnitude < zero_threshold
        zero_motion_blocks = np.argwhere(zero_motion_mask)


        zero_motion_blocks = [(x * block_size, y * block_size) for y, x in zero_motion_blocks]

        return zero_motion_blocks
    def find_background_blocks_ransac(self,motion_vectors, block_size=16, threshold=1.0, max_iters=1000):
        h, w, _ = motion_vectors.shape
        blocks = []
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                block = motion_vectors[y:y + block_size, x:x + block_size, :]
                blocks.append(((x, y), block))

        points = []
        for (x, y), block in blocks:
            u = np.mean(block[..., 0])
            v = np.mean(block[..., 1])
            points.append([x + block_size / 2, y + block_size / 2, u, v])
        points = np.array(points)

        src_points = points[:, :2]
        dst_points = src_points + points[:, 2:]

        model, inliers = cv2.findHomography(src_points, dst_points, cv2.RANSAC, threshold)

        background_blocks = []
        for i, inlier in enumerate(inliers):
            if inlier:
                background_blocks.append(blocks[i][0])

        return background_blocks
    def extract_motion_vectors(self,image1, image2):
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        motion_vectors = np.dstack((flow[..., 0], flow[..., 1]))
        return motion_vectors

    def convert_results_to_df(self,results):

        data = []
        for region in results.regions:
            xmin = region.x
            ymin = region.y
            xmax = region.x + region.w
            ymax = region.y + region.h
            data.append({
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'conf': region.conf,
                'label': region.label
            })

        df = pd.DataFrame(data)
        return df

    def normalize_bbox_coordinates(self, df, original_width, original_height, new_width, new_height):

        if df.empty:
            return df


        df['xmin'] = df['xmin'] / original_width * new_width
        df['ymin'] = df['ymin'] / original_height * new_height
        df['xmax'] = df['xmax'] / original_width * new_width
        df['ymax'] = df['ymax'] / original_height * new_height

        return df

    def find_background_blocks_ransac_gpu(self, motion_vectors):
        """
        Use RANSAC to detect background blocks from motion vectors on GPU.
        """

        gpu_motion_vectors = cv2.cuda_GpuMat()
        gpu_motion_vectors.upload(motion_vectors)


        height, width, _ = gpu_motion_vectors.shape
        motion_vectors_reshaped = gpu_motion_vectors.reshape((-1, 2))




        ransac_thresh = 1.0
        max_iters = 1000

        inliers = []
        for _ in range(max_iters):

            sample_idx = np.random.choice(len(motion_vectors_reshaped), 2, replace=False)
            sample_points = motion_vectors_reshaped[sample_idx]


            translation = np.mean(sample_points, axis=0)

            residuals = np.linalg.norm(motion_vectors_reshaped - translation, axis=1)


            inlier_mask = residuals < ransac_thresh


            inliers = motion_vectors_reshaped[inlier_mask]


        background_blocks = self._get_block_positions_from_motion_vectors(inliers, height, width)

        return background_blocks

    def _get_block_positions_from_motion_vectors(self, inliers, height, width, block_size=16):
        """
        Converts the inlier motion vectors into corresponding block positions in the image.
        """
        background_blocks = []
        for mv in inliers:

            x, y = mv
            block_x = int(x * width)
            block_y = int(y * height)


            background_blocks.append((block_x, block_y))

        return background_blocks



    def find_zero_motion_blocks_gpu(self, motion_vectors, block_size=16, zero_threshold=0.5):
        """
        GPU-optimized version of finding zero-motion blocks.
        """
        h, w, _ = motion_vectors.shape
        h_blocks = h // block_size
        w_blocks = w // block_size


        gpu_motion_vectors = cv2.cuda_GpuMat()
        gpu_motion_vectors.upload(motion_vectors)


        gpu_block_vectors = gpu_motion_vectors.reshape(h_blocks, block_size, w_blocks, block_size, 2)


        gpu_block_means = cv2.cuda.reduce(gpu_block_vectors, axis=(1, 3), op=cv2.REDUCE_AVG)


        gpu_magnitude = cv2.cuda.magnitude(gpu_block_means[..., 0], gpu_block_means[..., 1])


        gpu_zero_motion_mask = gpu_magnitude < zero_threshold


        zero_motion_mask = gpu_zero_motion_mask.download()


        zero_motion_blocks = np.argwhere(zero_motion_mask)


        zero_motion_blocks = [(x * block_size, y * block_size) for y, x in zero_motion_blocks]

        return zero_motion_blocks

    def process_image_triplet(self, image1_path, image2_path, image3_path, results_df, output_path=None):



        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        image3 = cv2.imread(image3_path)

        if image1 is None or image2 is None or image3 is None:

            return






        original_height, original_width, _ = image3.shape
        new_height, new_width, _ = image2.shape




        results_df = self.convert_results_to_df(results_df)

        results_df = self.normalize_bbox_coordinates(results_df, original_width, original_height, new_width, new_height)




        motion_vectors = self.extract_motion_vectors(image1, image2)


        h, w, _ = motion_vectors.shape
        all_blocks = [(x, y) for y in range(0, h, 16) for x in range(0, w, 16)]

        background_blocks = self.find_background_blocks_ransac(motion_vectors)

        zero_motion_blocks = self.find_zero_motion_blocks_optimized(motion_vectors, block_size=16, zero_threshold=0.5)

        block_size = 16
        bbox_blocks = set()
        for index, row in results_df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            for y in range(y1, y2, block_size):
                for x in range(x1, x2, block_size):
                    bbox_blocks.add((x, y))

        marked_blocks = set(background_blocks + zero_motion_blocks + list(bbox_blocks))
        unmarked_blocks = [block for block in all_blocks if block not in marked_blocks]
        return all_blocks, unmarked_blocks, bbox_blocks

    def calculate_large_blocks(self, all_blocks_list, unmarked_blocks_list, bbox_blocks_list, large_block_width, large_block_height):

        large_blocks_to_fill = {}


        def mark_large_block(x, y):
            large_block_x = (x // large_block_width) * large_block_width
            large_block_y = (y // large_block_height) * large_block_height
            large_blocks_to_fill[(large_block_x, large_block_y)] = True

        for (x, y) in bbox_blocks_list + unmarked_blocks_list:
            mark_large_block(x, y)

        return large_blocks_to_fill

    def normalize_coordinates(self,x, y, w, h, original_width, original_height):

            x_norm = x / original_width
            y_norm = y / original_height
            w_norm = w / original_width
            h_norm = h / original_height
            return x_norm, y_norm, w_norm, h_norm

    def base_req_regions_res(self, high_images_path, start_fid, end_fid, results_df, large_block_width=64,
                             large_block_height=64, n=5,padding=1):


        if len(results_df.regions) == 0:
            base_req_regions_res = [Region(0, 0, 0, 1, 1, 1.0, 2, 1)]
            return base_req_regions_res, []

        selected_fids = np.linspace(start_fid, end_fid - 1, n, dtype=int)


        all_blocks_list = []
        unmarked_blocks_list = []
        bbox_blocks_list = []

        def process_single_triplet(idx, fid):

            image1_path = os.path.join(high_images_path, f"{fid:08d}.jpg")
            image2_path = os.path.join(high_images_path, f"{fid + 1:08d}.jpg")
            image3_path = os.path.join(high_images_path, f"{fid + 2:08d}.jpg")


            if not os.path.exists(image1_path) or not os.path.exists(image2_path) or not os.path.exists(image3_path):


                return [], [], [], None, None


            img = cv2.imread(image1_path)
            if img is None:
                return [], [], [], None, None

            original_height, original_width = img.shape[:2]


            all_blocks, unmarked_blocks, bbox_blocks = self.process_image_triplet(image1_path, image2_path, image3_path,
                                                                                  results_df)


            return all_blocks, unmarked_blocks, bbox_blocks, original_width, original_height


        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_single_triplet, range(len(selected_fids)), selected_fids))

        original_width = original_height = None

        for all_blocks, unmarked_blocks, bbox_blocks, img_width, img_height in results:

            if img_width is None or img_height is None:
                continue


            if original_width is None or original_height is None:
                original_width, original_height = img_width, img_height


            all_blocks_list.extend(all_blocks)
            unmarked_blocks_list.extend(unmarked_blocks)
            bbox_blocks_list.extend(bbox_blocks)


        large_blocks_to_fill = self.calculate_large_blocks(all_blocks_list, unmarked_blocks_list, bbox_blocks_list,
                                                           large_block_width, large_block_height)

        base_req_regions_res = []
        for (large_block_x, large_block_y) in large_blocks_to_fill:

            x_norm, y_norm, w_norm, h_norm = self.normalize_coordinates(
                large_block_x - padding, large_block_y - padding,
                large_block_width + 2 * padding, large_block_height + 2 * padding,
                original_width=original_width, original_height=original_height
            )


            base_req_regions_res.append(
                Region(selected_fids[0], x_norm, y_norm, w_norm, h_norm, 1.0, 2, 1)
            )

        return base_req_regions_res, large_blocks_to_fill


