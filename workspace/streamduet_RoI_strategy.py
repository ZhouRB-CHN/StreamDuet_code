from abc import ABC, abstractmethod
import logging
import os
import json
import shutil
import cv2
import numpy as np
import time
from backend.server import Server
from frontend.client_factory import ClientFactory
from sd_utils import Results, Region, compute_regions_size, merge_boxes_in_results, cleanup, extract_images_from_video, read_results_dict
from streamduet_utils import list_frames,get_images_length,get_image_extension
from workspace.base_instance_strategy import InstanceStrategy
from backend.roi_cache_server import RoICacheServer
class StreamDuetRoIStrategy(InstanceStrategy):
    def run(self, args):
        self.logger.warning(f"Running StreamDuet in mode on {args.video_name}")
        if args.mode == "emulation":
            self.server = RoICacheServer(self.config)
            self.client = ClientFactory.get_roi_client(self.config, args.client_id, server=self.server)
            results, bw , lt = self.analyze_video_emulate(
            args.video_name, args.high_images_path,
            args.enforce_iframes, args.low_results_path, args.debug_mode)
        else:
            self.client = ClientFactory.get_roi_client(self.config, args.client_id, hname=args.hname)
            results, bw , lt = self.analyze_video(args.video_name, args.high_images_path, args.enforce_iframes)

        return results, bw, lt

    def get_current_bandwidth(self):
        if hasattr(self, 'network_monitor') and hasattr(self.network_monitor, 'get_bandwidth'):
            return self.network_monitor.get_bandwidth()
        else:

            self.logger.warning("Network monitor module not found. Using default bandwidth.")
            return 0

    def get_block_size(self, start_fid, end_fid):

        total_size = 0
        for fid in range(start_fid, end_fid):
            frame_path = os.path.join(self.config.high_images_path, f"{fid:08d}.jpg")
            total_size += os.path.getsize(frame_path)
        return total_size
    def analyze_video_emulate(self, video_name, high_images_path,
                              enforce_iframes, low_results_path=None, debug_mode=False):
        final_results = Results()
        low_phase_results = Results()
        high_phase_results = Results()
        number_of_frames = get_images_length(high_images_path)

        low_results_dict = None
        if low_results_path:
            low_results_dict = read_results_dict(low_results_path)

        total_size = [0, 0]
        total_regions_count = 0

        for i in range(0, number_of_frames, self.config.batch_size):
            start_fid = i
            end_fid = min(number_of_frames, i + self.config.batch_size)
            self.logger.info(f"Processing batch from {start_fid} to {end_fid}")

            n = 5
            selected_fids = np.linspace(start_fid, end_fid - 1, n, dtype=int)

            base_req_regions_res = Results()
            for fid in range(start_fid, end_fid):
                current_frame_image = cv2.imread(os.path.join(high_images_path, f"{fid:08d}.jpg"))
                frame_base_req_regions, frame_final_results = self.client.roi_cache.process_frame(fid,
                                                                                                  current_frame_image)
                final_results.combine_results(frame_final_results, self.config.intersection_threshold)


            low_images_path = f"{video_name}-base-phase-cropped"



            base_req_regions_res = Results()
            base_roi_regions, _ = self.client.roi_cache.base_req_regions_res(
                high_images_path, start_fid, end_fid, final_results, large_block_width=32,
                large_block_height=32, n=5)
            current_bandwidth = 0
            if hasattr(self, 'network_monitor'):

                current_bandwidth = self.get_current_bandwidth()
                block_size = self.get_block_size(start_fid, end_fid)
                self.config.low_qp,self.config.high_qp, self.config.low_resolution = self.adjust_encoding_parameters(current_bandwidth, block_size)

            for reg in base_roi_regions:
                x, y, w, h = reg.x, reg.y, reg.w, reg.h
                for fid in range(start_fid, end_fid):
                    base_req_regions_res.append(Region(fid, x, y, w, h, 1.0, 2, self.config.low_resolution))




            encoded_batch_video_size, batch_pixel_size = compute_regions_size(
                base_req_regions_res, f"{video_name}-base-phase", high_images_path,
                self.config.low_resolution, self.config.low_qp, enforce_iframes, True)


            self.logger.info(f"Sent {encoded_batch_video_size / 1024} in base phase")
            total_size[0] += encoded_batch_video_size

            r1, req_regions = self.server.simulate_low_query(
                start_fid, end_fid, low_images_path, low_results_dict, video_name, False,
                self.config.rpn_enlarge_ratio)
            total_regions_count += len(req_regions)

            low_phase_results.combine_results(r1, self.config.intersection_threshold)
            final_results.combine_results(r1, self.config.intersection_threshold)

            if len(req_regions) > 0:
                regions_size, _ = compute_regions_size(
                    req_regions, video_name, high_images_path,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                self.logger.info(
                    f"Sent {len(req_regions)} regions which have {regions_size / 1024}KB in second phase using {self.config.high_qp}")
                total_size[1] += regions_size


                inference_sharing_start = time.time()

                r2 = self.server.simulate_high_query(video_name, low_images_path, req_regions)
                self.logger.info(f"Got {len(r2)} results in second phase of batch")

                high_phase_results.combine_results(r2, self.config.intersection_threshold)
                final_results.combine_results(r2, self.config.intersection_threshold)


                lt['inference_sharing'] = time.time() - inference_sharing_start


            self.client.roi_cache.update_cache(start_fid, end_fid, final_results, high_images_path)
            cleanup(video_name, debug_mode, start_fid, end_fid)


        final_results.fill_gaps(number_of_frames)
        final_results.write(f"{video_name}")

        rdict = read_results_dict(f"{video_name}")
        final_results = merge_boxes_in_results(rdict, 0.3, 0.3)

        final_results.fill_gaps(number_of_frames)
        final_results.write(f"{video_name}")



        lt={}
        return final_results, total_size, lt

    def adjust_encoding_parameters(self, current_bandwidth, block_size):


        if current_bandwidth < 1e6:
            low_qp = 35
            high_qp = 30
            resolution = (640, 360)
        elif current_bandwidth < 5e6:
            low_qp = 28
            high_qp = 24
            resolution = (1280, 720)
        else:
            low_qp = 20
            high_qp = 18
            resolution = (1920, 1080)


        if block_size > 1e6:  #
            high_qp += 2

        return low_qp, high_qp, resolution

    def analyze_video(self, vid_name, raw_images, config, enforce_iframes):
        final_results = Results()
        all_required_regions = Results()
        low_phase_size = 0
        high_phase_size = 0

        nframes =list_frames(raw_images)

        self.client.init_server(nframes)

        for i in range(0, nframes, self.config.batch_size):
            start_frame = i
            end_frame = min(nframes, i + self.config.batch_size)
            self.logger.info(f"Processing frames {start_frame} to {end_frame}")

            # First iteration
            req_regions = Results()
            for fid in range(start_frame, end_frame):
                req_regions.append(Region(fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
            batch_video_size, _ =compute_regions_size(
                req_regions, f"{vid_name}-base-phase", raw_images,
                self.config.low_resolution, self.config.low_qp,
                enforce_iframes, True)
            low_phase_size += batch_video_size
            self.logger.info(f"{batch_video_size / 1024}KB sent in base phase."
                             f"Using QP {self.config.low_qp} and "
                             f"Resolution {self.config.low_resolution}.")
            results, rpn_regions = self.get_first_phase_results(vid_name)
            final_results.combine_results(
                results, self.config.intersection_threshold)
            all_required_regions.combine_results(
                rpn_regions, self.config.intersection_threshold)

            # Second Iteration
            if len(rpn_regions) > 0:
                batch_video_size, _ = compute_regions_size(
                    rpn_regions, vid_name, raw_images,
                    self.config.high_resolution, self.config.high_qp,
                    enforce_iframes, True)
                high_phase_size += batch_video_size
                self.logger.info(f"{batch_video_size / 1024}KB sent in second "
                                 f"phase. Using QP {self.config.high_qp} and "
                                 f"Resolution {self.config.high_resolution}.")
                results = self.get_second_phase_results(vid_name)
                final_results.combine_results(
                    results, self.config.intersection_threshold)


            cleanup(vid_name, False, start_frame, end_frame)

        self.logger.info(f"Merging results")
        final_results = merge_boxes_in_results(
            final_results.regions_dict, 0.3, 0.3)
        self.logger.info(f"Writing results for {vid_name}")
        final_results.fill_gaps(nframes)

        final_results.combine_results(
            all_required_regions, self.config.intersection_threshold)

        final_results.write(f"{vid_name}")

        return final_results, (low_phase_size, high_phase_size)