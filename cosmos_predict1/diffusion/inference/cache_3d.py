# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from einops import rearrange
from tqdm import tqdm

from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import (
    forward_warp,
    reliable_depth_mask_range_batch,
    unproject_points,
)
from cosmos_predict1.diffusion.inference.camera_utils import align_depth


from pytorch3d.renderer import (
    PerspectiveCameras, 
    PointsRasterizer, 
    PointsRasterizationSettings, 
    PointsRenderer, 
    AlphaCompositor,
    NormWeightedCompositor
)
from pytorch3d.structures import Pointclouds


def render_pointcloud_image(
    view_cameras_R_w2cs_pytorch3d: torch.Tensor, # [1, 3, 3]
    view_cameras_T_w2cs_pytorch3d: torch.Tensor, # [1, 3]
    view_cameras_focal_length: torch.Tensor, # [1, 2]
    view_cameras_principal_points: torch.Tensor, # [1, 2]
    points: torch.Tensor, # [N, 3]
    colors_with_alpha: torch.Tensor, # [N, 4]
    image_width: int = 1024,
    image_height: int = 704,
    device="cuda"
):
    radius_in_pixels = 1.5
    points_per_pixel = 32
    radius_ndc = 2.0 * radius_in_pixels / float(max(image_width, image_height))

    background_color_rgb = (0.0, 0.0, 0.0)

    compositor_type = "normal"
    if compositor_type == "alpha":
        compositor = AlphaCompositor(background_color=torch.tensor(background_color_rgb, device=device))
    else:
        compositor = NormWeightedCompositor(background_color=torch.tensor(background_color_rgb, device=device))



    # Distinct focal lengths in pixels; principal point in pixels; image size in pixels
    cameras = PerspectiveCameras(
        # focal_length=torch.tensor((fx, fy), device=device).unsqueeze(0),
        focal_length=view_cameras_focal_length,
        # principal_point=torch.tensor((cx, cy), device=device).unsqueeze(0),
        principal_point=view_cameras_principal_points,
        image_size=torch.tensor((image_height, image_width), device=device).expand(view_cameras_focal_length.shape[0], -1),  # enables screen-space parameterization
        # R=camera_R_w2c.unsqueeze(0), T=camera_T_w2c.unsqueeze(0), 
        R = view_cameras_R_w2cs_pytorch3d,
        T = view_cameras_T_w2cs_pytorch3d,
        # R=R, T=T,
        in_ndc=False,
        device=device
    )


    rasterizer = PointsRasterizer(
        cameras=cameras,
        raster_settings=PointsRasterizationSettings(
            image_size=(image_height, image_width),
            radius=radius_ndc,
            points_per_pixel = points_per_pixel,
            bin_size=64
        )
    )

    # print(rasterizer)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

    point_cloud = Pointclouds(
        # points=[points.clone() for _ in range(constrain_indices)],
        # features=[colors_with_alpha.clone() for _ in range(constrain_indices)],
        points = [points],
        features=[colors_with_alpha]
    )

    # images = renderer(point_cloud)
    images = renderer(point_cloud).clamp(0.0, 1.0)  # (N,H,W,3) float
    images_uint8 = (images * 255.0 + 0.5).to(torch.uint8)

    # return images
    # 


    # Fragments for coverage / masks / depth
    fragments = rasterizer(point_cloud)  # idx,zbuf,dists: (N,H,W,K)
    nearest_point_index = fragments.idx[..., 0]     # (N,H,W), -1 where empty
    cover_mask = nearest_point_index >= 0           # True where at least one point splatted
    holes_mask = ~cover_mask

    # Optional soft coverage confidence in [0,1]
    dists_squared = fragments.dists                  # (N,H,W,K)
    soft_coverage = (1.0 - dists_squared / (radius_ndc * radius_ndc)).clamp(min=0.0, max=1.0).max(dim=-1).values
    soft_coverage = torch.where(cover_mask, soft_coverage, torch.zeros_like(soft_coverage))

    # Optional first-hit view-space depth (NaN where holes)
    depth_view_z = fragments.zbuf[..., 0]
    depth_view_z = torch.where(cover_mask, depth_view_z, torch.full_like(depth_view_z, float("nan")))

    return {
        "rgb_composited_uint8": images_uint8,  # (N,H,W,3)
        "cover_mask": cover_mask,              # (N,H,W) bool
        "holes_mask": holes_mask,              # (N,H,W) bool
        "soft_coverage": soft_coverage,        # (N,H,W) float32
        "depth_view_z": depth_view_z,          # (N,H,W) float32 (NaN where invalid)
    }


class Cache3D_Base:
    def __init__(
        self,
        input_image,
        input_depth,
        input_w2c,
        input_intrinsics,
        input_mask=None,
        input_format=None,
        input_points=None,
        weight_dtype=torch.float32,
        is_depth=True,
        device="cuda",
        filter_points_threshold=1.0,
        foreground_masking=False,
    ):
        """
        input_image: Tensor with varying dimensions.
        input_format: List of dimension labels corresponding to input_image's dimensions.
                      E.g., ['B', 'C', 'H', 'W'], ['B', 'F', 'C', 'H', 'W'], etc.

        Description of dimensions:
        B = batch size. Usually 1
        F = frame count. For video inputs, this is the number of frames in time.
        N = number of views or buffers. For example, in a multi-view setup, this could be the number of cameras.
        V = not sure yet.
        C = number of channels in the image (e.g., RGB has 3 channels).
        H = height of the image.
        W = width of the image.

        """
        self.weight_dtype = weight_dtype
        self.is_depth = is_depth
        self.device = device
        self.filter_points_threshold = filter_points_threshold
        self.foreground_masking = foreground_masking
        if input_format is None:
            assert input_image.dim() == 4
            input_format = ["B", "C", "H", "W"]

        # Map dimension names to their indices in input_image
        format_to_indices = {dim: idx for idx, dim in enumerate(input_format)}
        input_shape = input_image.shape
        if input_mask is not None:
            input_image = torch.cat([input_image, input_mask], dim=format_to_indices.get("C"))

        # B (batch size), F (frame count), N dimensions: no aggregation during warping.
        # Only broadcasting over F to match the target w2c.
        # V: aggregate via concatenation or duster
        B = input_shape[format_to_indices.get("B", 0)] if "B" in format_to_indices else 1  # batch
        F = input_shape[format_to_indices.get("F", 0)] if "F" in format_to_indices else 1  # frame
        N = input_shape[format_to_indices.get("N", 0)] if "N" in format_to_indices else 1  # buffer
        V = input_shape[format_to_indices.get("V", 0)] if "V" in format_to_indices else 1  # view
        H = input_shape[format_to_indices.get("H", 0)] if "H" in format_to_indices else None
        W = input_shape[format_to_indices.get("W", 0)] if "W" in format_to_indices else None

        # Desired dimension order
        desired_dims = ["B", "F", "N", "V", "C", "H", "W"]

        # Build permute order based on input_format
        permute_order = []
        for dim in desired_dims:
            idx = format_to_indices.get(dim)
            if idx is not None:
                permute_order.append(idx)
            else:
                # Placeholder for dimensions to be added later
                permute_order.append(None)

        # Remove None values for permute operation
        permute_indices = [idx for idx in permute_order if idx is not None]
        input_image = input_image.permute(*permute_indices)

        # Insert dimensions of size 1 where necessary
        for i, idx in enumerate(permute_order):
            if idx is None:
                input_image = input_image.unsqueeze(i)

        # Now input_image has the shape B x F x N x V x C x H x W
        if input_mask is not None:
            self.input_image, self.input_mask = input_image[:, :, :, :, :3], input_image[:, :, :, :, 3:]
            self.input_mask = self.input_mask.to("cpu")
        else:
            self.input_mask = None
            self.input_image = input_image
        self.input_image = self.input_image.to(weight_dtype).to("cpu")

        if input_points is not None:
            self.input_points = input_points.reshape(B, F, N, V, H, W, 3).to("cpu")
            self.input_depth = None
        else:
            input_depth = torch.nan_to_num(input_depth, nan=100)
            input_depth = torch.clamp(input_depth, min=0, max=100)
            if weight_dtype == torch.float16:
                input_depth = torch.clamp(input_depth, max=70)
            self.input_points = (
                self._compute_input_points(
                    input_depth.reshape(-1, 1, H, W),
                    input_w2c.reshape(-1, 4, 4),
                    input_intrinsics.reshape(-1, 3, 3),
                )
                .to(weight_dtype)
                .reshape(B, F, N, V, H, W, 3)
                .to("cpu")
            )
            self.input_depth = input_depth

        if self.filter_points_threshold < 1.0 and input_depth is not None:
            input_depth = input_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(input_depth, ratio_thresh=self.filter_points_threshold).reshape(B, F, N, V, 1, H, W)
            if self.input_mask is None:
                self.input_mask = depth_mask.to("cpu")
            else:
                self.input_mask = self.input_mask * depth_mask.to(self.input_mask.device)
        self.boundary_mask = None
        if foreground_masking:
            input_depth = input_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(input_depth)
            self.boundary_mask = (~depth_mask).reshape(B, F, N, V, 1, H, W).to("cpu")

    def _compute_input_points(self, input_depth, input_w2c, input_intrinsics):
        input_points = unproject_points(
            input_depth,
            input_w2c,
            input_intrinsics,
            is_depth=self.is_depth,
        )
        return input_points

    def update_cache(self):
        raise NotImplementedError

    def input_frame_count(self) -> int:
        return self.input_image.shape[1]

    # def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
    #     bs, F_target, _, _ = target_w2cs.shape

    #     B, F, N, V, C, H, W = self.input_image.shape
    #     assert bs == B


    #     # (B*F*N), 4, 4
    #     target_w2cs = target_w2cs.reshape(B, F_target, 1, 4, 4).expand(B, F_target, N, 4, 4).reshape(-1, 4, 4)

    #     # (B*F*N), 3, 3
    #     target_intrinsics = (
    #         target_intrinsics.reshape(B, F_target, 1, 3, 3).expand(B, F_target, N, 3, 3).reshape(-1, 3, 3)
    #     )

    #     first_images = rearrange(self.input_image[:, start_frame_idx:start_frame_idx+F_target].expand(B, F_target, N, V, C, H, W), "B F N V C H W-> (B F N) V C H W").to(self.device)
    #     first_points = rearrange(
    #         self.input_points[:, start_frame_idx:start_frame_idx+F_target].expand(B, F_target, N, V, H, W, 3), "B F N V H W C-> (B F N) V H W C"
    #     ).to(self.device)
    #     first_masks = rearrange(
    #         self.input_mask[:, start_frame_idx:start_frame_idx+F_target].expand(B, F_target, N, V, 1, H, W), "B F N V C H W-> (B F N) V C H W"
    #     ).to(self.device) if self.input_mask is not None else None
    #     boundary_masks = rearrange(
    #         self.boundary_mask.expand(B, F_target, N, V, 1, H, W), "B F N V C H W-> (B F N) V C H W"
    #     ) if self.boundary_mask is not None else None

    #     if first_images.shape[1] == 1:
    #         # warp_chunk_size = 2 # number of images to warp at once. In this case 2. These 2 images are independent from each other!!!
    #         warp_chunk_size = 32 # number of images to warp at once. In this case 2. These 2 images are independent from each other!!!
    #         # warp_chunk_size = 4
    #         # warp_chunk_size = 16
    #         rendered_warp_images = []
    #         rendered_warp_masks = []
    #         rendered_warp_depth = []
    #         rendered_warped_flows = []
            

    #         first_images = first_images.squeeze(1)
    #         first_points = first_points.squeeze(1)
    #         first_masks = first_masks.squeeze(1) if first_masks is not None else None

    #         # Render for all target w2cs/intrinsics 
    #         for i in tqdm(range(0, first_images.shape[0], warp_chunk_size), desc="Rendering cache"):

    #             #  warp_chunk_size 
    #             # Number of images to warp at once. In this case 2. 
    #             # These 2 images are independent from each other!!!

    #             (
    #                 rendered_warp_images_chunk, # [warp_chunk_size=2, 3, H, W]
    #                 rendered_warp_masks_chunk, # [warp_chunk_size=2, 1, H, W]
    #                 rendered_warp_depth_chunk, # [warp_chunk_size=2, H, W] if render_depth else None
    #                 rendered_warped_flows_chunk, # [warp_chunk_size=2, 2, H, W]
    #             ) = forward_warp(
    #                 first_images[i : i + warp_chunk_size],
    #                 mask1=first_masks[i : i + warp_chunk_size] if first_masks is not None else None,
    #                 depth1=None,
    #                 transformation1=None,
    #                 transformation2=target_w2cs[i : i + warp_chunk_size],
    #                 intrinsic1=target_intrinsics[i : i + warp_chunk_size],
    #                 intrinsic2=target_intrinsics[i : i + warp_chunk_size],
    #                 render_depth=render_depth,
    #                 world_points1=first_points[i : i + warp_chunk_size],
    #                 foreground_masking=self.foreground_masking,
    #                 boundary_mask=boundary_masks[i : i + warp_chunk_size, 0, 0] if boundary_masks is not None else None
    #             )
    #             rendered_warp_images.append(rendered_warp_images_chunk)
    #             rendered_warp_masks.append(rendered_warp_masks_chunk)
    #             rendered_warp_depth.append(rendered_warp_depth_chunk)
    #             rendered_warped_flows.append(rendered_warped_flows_chunk)

    #             # delete rendered_warp_images_chunk, rendered_warp_masks_chunk, rendered_warp_depth_chunk, rendered_warped_flows_chunk
    #             del rendered_warp_images_chunk, rendered_warp_masks_chunk, rendered_warp_depth_chunk
    #             del rendered_warped_flows_chunk
    #         rendered_warp_images = torch.cat(rendered_warp_images, dim=0)
    #         rendered_warp_masks = torch.cat(rendered_warp_masks, dim=0)
    #         if render_depth:
    #             rendered_warp_depth = torch.cat(rendered_warp_depth, dim=0)
    #         rendered_warped_flows = torch.cat(rendered_warped_flows, dim=0)

    #     else:
    #         raise NotImplementedError

    #     pixels = rearrange(rendered_warp_images, "(b f n) c h w -> b f n c h w", b=bs, f=F_target, n=N)
    #     masks = rearrange(rendered_warp_masks, "(b f n) c h w -> b f n c h w", b=bs, f=F_target, n=N)
    #     if render_depth:
    #         pixels = rearrange(rendered_warp_depth, "(b f n) h w -> b f n h w", b=bs, f=F_target, n=N)
    #     return pixels, masks


    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        bs, F_target, _, _ = target_w2cs.shape

        B, F, N, V, C, H, W = self.input_image.shape
        assert bs == B


        opencv_to_pytorch3d = torch.diag(torch.as_tensor([-1, -1, 1, 1.0])).to("cuda")
        view_camera_intrinsics = target_intrinsics
        view_cameras_w2cs_pytorch3d = opencv_to_pytorch3d @ target_w2cs


        view_cameras_R_w2cs_pytorch3d = view_cameras_w2cs_pytorch3d[0, :, :3, :3]
        view_cameras_T_w2cs_pytorch3d = view_cameras_w2cs_pytorch3d[0, :, :3, 3]

        view_cameras_fx = view_camera_intrinsics[0, :, 0,0]
        view_cameras_fy = view_camera_intrinsics[0, :, 1,1]
        view_cameras_cx = view_camera_intrinsics[0, :, 0,2]
        view_cameras_cy = view_camera_intrinsics[0, :, 1,2]


        view_cameras_focal_length = torch.stack([view_cameras_fx, view_cameras_fy], dim=1)
        view_cameras_principal_points = torch.stack([view_cameras_cx, view_cameras_cy], dim=1)

        # view_cameras_focal_length = view_cameras_focal_length[:F]
        # view_cameras_principal_points = view_cameras_principal_points[:F]


        # 
    
        point_tensor = self.input_points
        color_tensor = self.input_image
        point_tensor = point_tensor.contiguous()
        color_tensor = color_tensor.contiguous()

        batch_size, frame_count, buffer_count, view_count, height, width, _ = point_tensor.shape
        total_pixels = batch_size * frame_count * buffer_count * view_count * height * width

        point_coordinates = point_tensor.view(total_pixels, 3)  # (N_total, 3)
        color_channels_first = color_tensor.permute(0, 1, 2, 3, 5, 6, 4)  # move C from 5th pos to last
        # rgb_colors = (color_channels_first * 255.0).clamp(0.0, 255.0).to(torch.uint8).view(total_pixels, 3)
        rgb_colors = color_channels_first * 0.5 + 0.5

        rgb_colors = rgb_colors.reshape(total_pixels, 3)  # (N_total, 3)

        rgba_colors = torch.cat([rgb_colors, torch.ones((total_pixels, 1), device=rgb_colors.device)], dim=1)

        point_coordinates = point_coordinates.to(self.device)
        rgba_colors = rgba_colors.to(self.device)

        pixels_list = []
        masks_list = []

        for i in tqdm(range(F_target), desc="Render images"):        
            result_dict = render_pointcloud_image(
                view_cameras_R_w2cs_pytorch3d=view_cameras_R_w2cs_pytorch3d[i:i+1],
                view_cameras_T_w2cs_pytorch3d=view_cameras_T_w2cs_pytorch3d[i:i+1],
                view_cameras_focal_length=view_cameras_focal_length[i:i+1],
                view_cameras_principal_points=view_cameras_principal_points[i:i+1],
                image_width=W,
                image_height=H,
                points=point_coordinates,
                colors_with_alpha=rgba_colors
            )
            # import pdb; pdb.set_trace()


            pixels_0_1 = result_dict["rgb_composited_uint8"][0, ..., :3] / 255.0
            pixels_minus_1_plus_1 = pixels_0_1 * 2.0 - 1.0

            mask_single = result_dict["cover_mask"][0]

            pixels_list.append(pixels_minus_1_plus_1)
            masks_list.append(mask_single)

        # import pdb; pdb.set_trace()
        # stack
        pixels = torch.stack(pixels_list, dim=0)
        masks = torch.stack(masks_list, dim=0)

        # b f n c h w
        pixels = pixels.permute(0, 3, 1, 2).unsqueeze(0).unsqueeze(2)
        masks = masks.unsqueeze(-1).permute(0, 3, 1, 2).unsqueeze(0).unsqueeze(2)

        return pixels, masks


class Cache3D_Buffer(Cache3D_Base):
    def __init__(self, frame_buffer_max=0, noise_aug_strength=0, generator=None, **kwargs):
        super().__init__(**kwargs)
        self.frame_buffer_max = frame_buffer_max
        self.noise_aug_strength = noise_aug_strength
        self.generator = generator

    def update_cache(self, new_image, new_depth, new_w2c, new_mask=None, new_intrinsics=None, depth_alignment=True, alignment_method="non_rigid"):  # 3D cache
        new_image = new_image.to(self.weight_dtype).to(self.device)
        new_depth = new_depth.to(self.weight_dtype).to(self.device)
        new_w2c = new_w2c.to(self.weight_dtype).to(self.device)
        if new_intrinsics is not None:
            new_intrinsics = new_intrinsics.to(self.weight_dtype).to(self.device)

        new_depth = torch.nan_to_num(new_depth, nan=1e4)
        new_depth = torch.clamp(new_depth, min=0, max=1e4)

        if depth_alignment:
            target_depth, target_mask = self.render_cache(
                new_w2c.unsqueeze(1), new_intrinsics.unsqueeze(1), render_depth=True
            )
            target_depth, target_mask = target_depth[:, :, 0], target_mask[:, :, 0]
            if alignment_method == "rigid":
                new_depth = (
                    align_depth(
                        new_depth.squeeze(),
                        target_depth.squeeze(),
                        target_mask.bool().squeeze(),
                    )
                    .reshape_as(new_depth)
                    .detach()
                )
            elif alignment_method == "non_rigid":
                with torch.enable_grad():
                    new_depth = (
                        align_depth(
                            new_depth.squeeze(),
                            target_depth.squeeze(),
                            target_mask.bool().squeeze(),
                            k=new_intrinsics.squeeze(),
                            c2w=torch.inverse(new_w2c.squeeze()),
                            alignment_method="non_rigid",
                            num_iters=100,
                            lambda_arap=0.1,
                            smoothing_kernel_size=3,
                        )
                        .reshape_as(new_depth)
                        .detach()
                    )
            else:
                raise NotImplementedError
        new_points = unproject_points(new_depth, new_w2c, new_intrinsics, is_depth=self.is_depth).cpu()
        new_image = new_image.cpu()

        if self.filter_points_threshold < 1.0:
            print("Applying depth filtering with threshold:", self.filter_points_threshold)
            B, F, N, V, C, H, W = self.input_image.shape
            new_depth = new_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(new_depth, ratio_thresh=self.filter_points_threshold).reshape(B, 1, H, W)
            if new_mask is None:
                new_mask = depth_mask.to("cpu")
            else:
                new_mask = new_mask * depth_mask.to(new_mask.device)
        if new_mask is not None:
            new_mask = new_mask.cpu()
        if self.frame_buffer_max > 1:  # newest frame first
            if self.input_image.shape[2] < self.frame_buffer_max:
                self.input_image = torch.cat([new_image[:, None, None, None], self.input_image], 2)
                self.input_points = torch.cat([new_points[:, None, None, None], self.input_points], 2)
                if self.input_mask is not None:
                    self.input_mask = torch.cat([new_mask[:, None, None, None], self.input_mask], 2)
            else:
                self.input_image[:, :, 0] = new_image[:, None, None]
                self.input_points[:, :, 0] = new_points[:, None, None]
                if self.input_mask is not None:
                    self.input_mask[:, :, 0] = new_mask[:, None, None]
        else:
            self.input_image = new_image[:, None, None, None]
            self.input_points = new_points[:, None, None, None]


    def render_cache(
        self,
        target_w2cs, # [B, F, 4, 4] F=121
        target_intrinsics, # [B, F, 3, 3]
        render_depth: bool = False,
        start_frame_idx: int = 0,  # For consistency with Cache4D
    ):
        assert start_frame_idx == 0, "start_frame_idx must be 0 for Cache3D_Buffer"

        output_device = target_w2cs.device
        target_w2cs = target_w2cs.to(self.weight_dtype).to(self.device)
        target_intrinsics = target_intrinsics.to(self.weight_dtype).to(self.device)
        pixels, masks = super().render_cache(
            target_w2cs, target_intrinsics, render_depth
        )
        if not render_depth:
            noise = torch.randn(pixels.shape, generator=self.generator, device=pixels.device, dtype=pixels.dtype)
            per_buffer_noise = (
                torch.arange(start=pixels.shape[2] - 1, end=-1, step=-1, device=pixels.device)
                * self.noise_aug_strength
            )
            pixels = pixels + noise * per_buffer_noise.reshape(1, 1, -1, 1, 1, 1)  # B, F, N, C, H, W
        return pixels.to(output_device), masks.to(output_device)


class Cache4D(Cache3D_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_cache(self, **kwargs):
        raise NotImplementedError

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        rendered_warp_images, rendered_warp_masks = super().render_cache(target_w2cs, target_intrinsics, render_depth, start_frame_idx)
        return rendered_warp_images, rendered_warp_masks
