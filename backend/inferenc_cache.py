import os
import time
import pickle
from collections import deque
import threading
import cv2
class InferenceCache:
    def __init__(self, time_window, cache_dir, conf_threshold, update_interval=0.1):
        self.time_window = time_window
        self.cache_dir = cache_dir
        self.conf_threshold = conf_threshold
        self.memory_cache = {}
        self.update_interval = update_interval
        self._stop_event = threading.Event()
        self.cache_lock = threading.Lock()
        self.cache_update_thread = threading.Thread(target=self.update_cache_periodically)
        self.cache_update_thread.daemon = True
        self.cache_update_thread.start()
    def update_cache_periodically(self):
        while not self._stop_event.is_set():

            self.update_memory_cache()
            time.sleep(self.update_interval)


    def update_memory_cache(self):
        while not self._stop_event.is_set():
            for root, _, files in os.walk(self.cache_dir):
                for file in files:
                    if file == "cache.pkl":
                        cache = self._load_cache(os.path.join(root, file))
                        with self.cache_lock:
                            for frame_id, timestamp, results, image_path in cache:
                                self.memory_cache[image_path] = (frame_id, results)
            time.sleep(self.update_interval)

    def _current_time(self):
        return time.time()

    def _get_cache_path(self, video_name):
        return os.path.join(self.cache_dir, video_name, "cache.pkl")

    def _get_image_path(self, video_name, frame_id):
        return os.path.join(self.cache_dir, video_name, f"{frame_id}.jpg")

    def _get_lock_path(self, video_name):
        return os.path.join(self.cache_dir, video_name, "cache.lock")

    def _load_cache(self, path):
        if not os.path.exists(path):
            return deque()
        try:
            with open(path, 'rb') as f:
                cache = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            return deque()
        for frame_id, timestamp, results, image_path in cache:
            for region in results:
                if region.feature is not None and 'keypoints' in region.feature:
                    region.feature['keypoints'] = self.dict_to_keypoints(region.feature['keypoints'])
        return cache
    def stop(self):
        self._stop_event.set()
        self.cache_update_thread.join()
    def _save_cache(self, path, cache):
        for frame_id, timestamp, results, image_path in cache:
            for region in results:
                if region.feature is not None and 'keypoints' in region.feature:
                    region.feature['keypoints'] = self.keypoints_to_dict(region.feature['keypoints'])
        with open(path, 'wb') as f:
            pickle.dump(cache, f)
        for frame_id, timestamp, results, image_path in cache:
            for region in results:
                if region.feature is not None and 'keypoints' in region.feature:
                    region.feature['keypoints'] = self.dict_to_keypoints(region.feature['keypoints'])

    def keypoints_to_dict(self, keypoints):
        return [(kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in keypoints]

    def dict_to_keypoints(self, kp_dict):
        keypoints = []
        for pt, size, angle, response, octave, class_id in kp_dict:
            keypoints.append(
                cv2.KeyPoint(pt[0], pt[1], size, angle, response, octave, class_id))
        return keypoints
    def add_results(self, video_name, frame_id, results, frame_image):
        video_cache_dir = os.path.join(self.cache_dir, video_name)
        os.makedirs(video_cache_dir, exist_ok=True)
        cache_path = self._get_cache_path(video_name)
        image_path = self._get_image_path(video_name, frame_id)

        with self.cache_lock:


            cache = self._load_cache(cache_path)


            for region in results:
                if region.feature is None:
                    cropped_cached_image = self._crop_image(frame_image, region)
                    feature = self.extract_features(cropped_cached_image)
                    region.feature = feature


            cache.append((frame_id, self._current_time(), results, image_path))


            self._cleanup()
            self._save_cache(cache_path, cache)

            self.memory_cache[image_path] = (frame_id, results)

    def get_best_result(self, bbox, query_image):
        best_result = None
        best_image_path = None


        with self.cache_lock:
            for image_path, (frame_id, results) in self.memory_cache.items():
                current_frame = self._get_max_frame(self.memory_cache)
                min_frame = current_frame - self.time_window


                if frame_id < min_frame:
                    continue

                cropped_query_image = self._crop_image(query_image, bbox)

                for region in results:

                    if region.conf >= self.conf_threshold:
                        if self.match_features(self.extract_features(cropped_query_image), region.feature):
                            if best_result is None or region.conf > best_result.conf:
                                best_result = region
                                best_image_path = image_path
                                bbox.conf = best_result.conf
                                bbox.label = best_result.label
                                bbox.origin = "low-cache-res"
                                bbox.feature = region.feature
                                return bbox, best_image_path

        return best_result, best_image_path

    def _cleanup(self):
        if not self.memory_cache:
            return


        current_frame = self._get_max_frame(self.memory_cache)


        min_frame = current_frame - self.time_window


        keys_to_remove = []


        for image_path, (frame_id, results) in self.memory_cache.items():

            if frame_id < min_frame:
                keys_to_remove.append(image_path)


        for image_path in keys_to_remove:

            del self.memory_cache[image_path]

            if os.path.exists(image_path):
                os.remove(image_path)

    def _get_max_frame(self, cache):
        if not cache:
            return -1

        valid_frame_ids = []
        invalid_frame_ids = []

        for video_path, frame_data_list in cache.items():

            frame_id = frame_data_list[0]
            if isinstance(frame_id, int):
                valid_frame_ids.append(frame_id)
            else:
                invalid_frame_ids.append(frame_id)


        return max(valid_frame_ids, default=-1)

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
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        return len(good_matches) > 10

    def match_features(self,query_features, cached_features, ratio_threshold=0.75, match_threshold=10):
        query_keypoints, query_descriptors = query_features['keypoints'], query_features['descriptors']
        cached_keypoints, cached_descriptors = cached_features['keypoints'], cached_features['descriptors']

        if query_descriptors is None or cached_descriptors is None:
            return False


        bf = cv2.BFMatcher()


        matches = bf.knnMatch(query_descriptors, cached_descriptors, k=2)


        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)


        return len(good_matches) > match_threshold
    def extract_features(self,image):

        sift = cv2.SIFT_create()


        keypoints, descriptors = sift.detectAndCompute(image, None)

        return {'keypoints': keypoints, 'descriptors': descriptors}
