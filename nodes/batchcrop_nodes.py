from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode
import math

def pil2tensor(image):
    if isinstance(image, list):
        return torch.cat([pil2tensor(img) for img in image], dim=0)
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image[i]))
        return out
    return [Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))]


def bbox_check(bbox, target_size=None):
    if not target_size:
        return bbox

    return (
        bbox[0],
        bbox[1],
        min(target_size[0] - bbox[0], bbox[2]),
        min(target_size[1] - bbox[1], bbox[3]),
    )


def bbox_to_region(bbox, target_size=None):
    bbox = bbox_check(bbox, target_size)
    return (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])


class ADBatchCropFromMaskAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "masks": ("MASK",),
                "crop_size_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "bbox_smooth_alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "preserve_aspect_ratio": ("BOOLEAN", {"default": False}),
                "max_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "max_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "MASK",
        "IMAGE",
        "MASK",
        "BBOX",
        "BBOX",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "original_images",
        "cropped_images",
        "cropped_masks",
        "combined_crop_image",
        "combined_crop_masks",
        "bboxes",
        "combined_bounding_box",
        "bbox_width",
        "bbox_height",
    )
    FUNCTION = "crop"
    CATEGORY = "KJNodes/masking"

    def smooth_bbox_size(self, prev_bbox_size, curr_bbox_size, alpha):
        if prev_bbox_size <= 0:
            return round(curr_bbox_size)
        if alpha <= 0:
            return round(prev_bbox_size)
        if alpha >= 1:
            return round(curr_bbox_size)
        return round(alpha * curr_bbox_size + (1 - alpha) * prev_bbox_size)

    def smooth_center(self, prev_center, curr_center, alpha=0.5):
        if prev_center is None:
            return curr_center
        if alpha <= 0:
            return prev_center
        if alpha >= 1:
            return curr_center
        return (
            round(alpha * curr_center[0] + (1 - alpha) * prev_center[0]),
            round(alpha * curr_center[1] + (1 - alpha) * prev_center[1]),
        )

    def crop(self, masks, original_images, crop_size_mult, bbox_smooth_alpha, preserve_aspect_ratio, max_width, max_height):
        bounding_boxes = []
        combined_bounding_box = []
        cropped_images = []
        cropped_masks = []
        combined_cropped_images = []
        combined_cropped_masks = []

        image_h = int(original_images[0].shape[0])
        image_w = int(original_images[0].shape[1])

        def calculate_bbox(mask):
            non_zero_indices = np.nonzero(np.array(mask))

            if len(non_zero_indices[1]) == 0 or len(non_zero_indices[0]) == 0:
                return {
                    "has_pixels": False,
                    "center_x": image_w / 2.0,
                    "center_y": image_h / 2.0,
                    "width": 0,
                    "height": 0,
                    "size": 0,
                }

            min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
            min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
            width = max(1, int((max_x - min_x) + 1))
            height = max(1, int((max_y - min_y) + 1))
            return {
                "has_pixels": True,
                "center_x": float((min_x + max_x) / 2.0),
                "center_y": float((min_y + max_y) / 2.0),
                "width": width,
                "height": height,
                "size": max(width, height),
            }

        def build_bbox(center_x, center_y, box_w, box_h, img_w, img_h):
            width = int(max(1, min(round(box_w), img_w)))
            height = int(max(1, min(round(box_h), img_h)))
            half_w = width / 2.0
            half_h = height / 2.0

            min_x = int(round(center_x - half_w))
            min_y = int(round(center_y - half_h))

            min_x = max(0, min(min_x, img_w - width))
            min_y = max(0, min(min_y, img_h - height))

            max_x = min_x + width
            max_y = min_y + height
            return min_x, min_y, max_x, max_y

        def clamp_dims_with_aspect(width, height, limit_w, limit_h):
            width = max(1, int(round(width)))
            height = max(1, int(round(height)))
            scale = 1.0
            if limit_w > 0:
                scale = min(scale, limit_w / width)
            if limit_h > 0:
                scale = min(scale, limit_h / height)
            if scale < 1.0:
                width = max(1, int(round(width * scale)))
                height = max(1, int(round(height * scale)))
            return width, height

        def expand_to_aspect_cover(width, height, aspect):
            width = max(1, int(round(width)))
            height = max(1, int(round(height)))
            if aspect <= 0:
                return width, height

            curr_aspect = width / height
            if curr_aspect < aspect:
                width = max(1, int(math.ceil(height * aspect)))
            else:
                height = max(1, int(math.ceil(width / aspect)))
            return width, height

        mask_infos = [calculate_bbox(tensor2pil(mask)[0]) for mask in masks]
        non_empty_infos = [info for info in mask_infos if info["has_pixels"]]

        if preserve_aspect_ratio:
            source_aspect = image_w / image_h
            if non_empty_infos:
                curr_max_bbox_width = max(info["width"] for info in non_empty_infos)
                curr_max_bbox_height = max(info["height"] for info in non_empty_infos)
            else:
                curr_max_bbox_width = image_w
                curr_max_bbox_height = image_h
            curr_max_bbox_width, curr_max_bbox_height = expand_to_aspect_cover(
                curr_max_bbox_width, curr_max_bbox_height, source_aspect
            )

            prev_w = getattr(self, "_prev_max_bbox_width", curr_max_bbox_width)
            prev_h = getattr(self, "_prev_max_bbox_height", curr_max_bbox_height)
            max_bbox_width = self.smooth_bbox_size(prev_w, curr_max_bbox_width, bbox_smooth_alpha)
            max_bbox_height = self.smooth_bbox_size(prev_h, curr_max_bbox_height, bbox_smooth_alpha)
            max_bbox_width, max_bbox_height = expand_to_aspect_cover(
                max_bbox_width, max_bbox_height, source_aspect
            )
            self._prev_max_bbox_width = max_bbox_width
            self._prev_max_bbox_height = max_bbox_height
            target_width = max(1, round(max_bbox_width * crop_size_mult))
            target_height = max(1, round(max_bbox_height * crop_size_mult))
            target_width, target_height = expand_to_aspect_cover(
                target_width, target_height, source_aspect
            )
            target_width, target_height = clamp_dims_with_aspect(target_width, target_height, image_w, image_h)

            output_width, output_height = target_width, target_height
            if max_width > 0 or max_height > 0:
                output_width, output_height = clamp_dims_with_aspect(output_width, output_height, max_width, max_height)
        else:
            if non_empty_infos:
                curr_max_bbox_size = max(info["size"] for info in non_empty_infos)
            else:
                curr_max_bbox_size = min(image_h, image_w)

            prev_size = getattr(self, "_prev_max_bbox_size", curr_max_bbox_size)
            max_bbox_size = self.smooth_bbox_size(prev_size, curr_max_bbox_size, bbox_smooth_alpha)
            self._prev_max_bbox_size = max_bbox_size
            max_bbox_size = max(1, round(max_bbox_size * crop_size_mult))
            max_bbox_size = math.ceil(max_bbox_size / 16) * 16

            min_image_dim = min(image_h, image_w)
            if max_bbox_size > min_image_dim:
                if min_image_dim >= 16:
                    max_bbox_size = (min_image_dim // 16) * 16
                else:
                    max_bbox_size = min_image_dim

            target_width = max_bbox_size
            target_height = max_bbox_size
            output_width = target_width
            output_height = target_height

        combined_mask = torch.max(masks, dim=0)[0]
        combined_info = calculate_bbox(tensor2pil(combined_mask)[0])

        if combined_info["has_pixels"]:
            combined_center_x = combined_info["center_x"]
            combined_center_y = combined_info["center_y"]
            combined_width = max(1, round(combined_info["width"] * crop_size_mult))
            combined_height = max(1, round(combined_info["height"] * crop_size_mult))
        else:
            combined_center_x = image_w / 2.0
            combined_center_y = image_h / 2.0
            combined_width = target_width
            combined_height = target_height

        if preserve_aspect_ratio:
            combined_width, combined_height = expand_to_aspect_cover(
                combined_width, combined_height, source_aspect
            )
            combined_width, combined_height = clamp_dims_with_aspect(combined_width, combined_height, image_w, image_h)
        else:
            combined_size = max(combined_width, combined_height)
            combined_size = max(1, int(round(combined_size)))
            combined_size = min(combined_size, min(image_w, image_h))
            combined_width = combined_height = combined_size

        new_min_x, new_min_y, new_max_x, new_max_y = build_bbox(
            combined_center_x,
            combined_center_y,
            combined_width,
            combined_height,
            image_w,
            image_h,
        )
        combined_bounding_box.append((new_min_x, new_min_y, new_max_x - new_min_x, new_max_y - new_min_y))

        self.prev_center = None

        for i, (mask, img, info) in enumerate(zip(masks, original_images, mask_infos)):
            if info["has_pixels"]:
                curr_center = (round(info["center_x"]), round(info["center_y"]))
            elif self.prev_center is not None:
                curr_center = self.prev_center
            else:
                curr_center = (image_w // 2, image_h // 2)

            if i > 0:
                center = self.smooth_center(self.prev_center, curr_center, bbox_smooth_alpha)
            else:
                center = curr_center
            self.prev_center = center

            min_x, min_y, max_x, max_y = build_bbox(
                center[0],
                center[1],
                target_width,
                target_height,
                int(img.shape[1]),
                int(img.shape[0]),
            )
            bounding_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

            cropped_img = img[min_y:max_y, min_x:max_x, :]
            cropped_mask = mask[min_y:max_y, min_x:max_x]

            if cropped_img.shape[0] != output_height or cropped_img.shape[1] != output_width:
                image_resize = Resize((output_height, output_width))
                mask_resize = Resize((output_height, output_width), interpolation=InterpolationMode.NEAREST)
                cropped_img = image_resize(cropped_img.permute(2, 0, 1)).permute(1, 2, 0)
                cropped_mask = mask_resize(cropped_mask.unsqueeze(0)).squeeze(0)

            cropped_images.append(cropped_img)
            cropped_masks.append(cropped_mask)

            combined_cropped_img = original_images[i][new_min_y:new_max_y, new_min_x:new_max_x, :]
            combined_cropped_images.append(combined_cropped_img)

            combined_cropped_mask = masks[i][new_min_y:new_max_y, new_min_x:new_max_x]
            combined_cropped_masks.append(combined_cropped_mask)

        cropped_out = torch.stack(cropped_images, dim=0)
        combined_crop_out = torch.stack(combined_cropped_images, dim=0)
        cropped_masks_out = torch.stack(cropped_masks, dim=0)
        combined_crop_mask_out = torch.stack(combined_cropped_masks, dim=0)

        return (
            original_images,
            cropped_out,
            cropped_masks_out,
            combined_crop_out,
            combined_crop_mask_out,
            bounding_boxes,
            combined_bounding_box,
            output_width,
            output_height,
        )


class ADBatchUncropAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "cropped_images": ("IMAGE",),
                "cropped_masks": ("MASK",),
                "combined_crop_mask": ("MASK",),
                "bboxes": ("BBOX",),
                "border_blending": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_rescale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "use_combined_mask": ("BOOLEAN", {"default": False}),
                "use_square_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "combined_bounding_box": ("BBOX", {"default": None}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "uncrop"
    CATEGORY = "KJNodes/masking"

    def uncrop(
        self,
        original_images,
        cropped_images,
        cropped_masks,
        combined_crop_mask,
        bboxes,
        border_blending,
        crop_rescale,
        use_combined_mask,
        use_square_mask,
        combined_bounding_box=None,
    ):
        def inset_border(image, border_width=20, border_color=(0)):
            width, height = image.size
            bordered_image = Image.new(image.mode, (width, height), border_color)
            bordered_image.paste(image, (0, 0))
            draw = ImageDraw.Draw(bordered_image)
            draw.rectangle((0, 0, width - 1, height - 1), outline=border_color, width=border_width)
            return bordered_image

        def scale_region_around_center(region, scale, image_size):
            x0, y0, x1, y1 = region
            if scale == 1.0:
                return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))

            width = x1 - x0
            height = y1 - y0
            center_x = x0 + (width / 2.0)
            center_y = y0 + (height / 2.0)

            scaled_w = max(1, int(round(width * scale)))
            scaled_h = max(1, int(round(height * scale)))

            new_x0 = int(round(center_x - (scaled_w / 2.0)))
            new_y0 = int(round(center_y - (scaled_h / 2.0)))
            new_x1 = new_x0 + scaled_w
            new_y1 = new_y0 + scaled_h

            img_w, img_h = image_size

            if new_x0 < 0:
                new_x1 -= new_x0
                new_x0 = 0
            if new_y0 < 0:
                new_y1 -= new_y0
                new_y0 = 0
            if new_x1 > img_w:
                shift = new_x1 - img_w
                new_x0 -= shift
                new_x1 = img_w
            if new_y1 > img_h:
                shift = new_y1 - img_h
                new_y0 -= shift
                new_y1 = img_h

            new_x0 = max(0, new_x0)
            new_y0 = max(0, new_y0)
            new_x1 = min(img_w, new_x1)
            new_y1 = min(img_h, new_y1)

            if new_x1 <= new_x0:
                new_x1 = min(img_w, new_x0 + 1)
            if new_y1 <= new_y0:
                new_y1 = min(img_h, new_y0 + 1)

            return (int(new_x0), int(new_y0), int(new_x1), int(new_y1))

        if len(original_images) != len(cropped_images):
            raise ValueError(
                f"The number of original_images ({len(original_images)}) and cropped_images ({len(cropped_images)}) should be the same"
            )

        if len(bboxes) > len(original_images):
            print(
                f"Warning: Dropping excess bounding boxes. Expected {len(original_images)}, but got {len(bboxes)}"
            )
            bboxes = bboxes[: len(original_images)]
        elif len(bboxes) < len(original_images):
            raise ValueError("There should be at least as many bboxes as there are original and cropped images")

        crop_imgs = tensor2pil(cropped_images)
        input_images = tensor2pil(original_images)
        out_images = []

        for i in range(len(input_images)):
            img = input_images[i]
            crop = crop_imgs[i]
            bbox = bboxes[i]

            if use_combined_mask:
                if combined_bounding_box is None or len(combined_bounding_box) == 0:
                    raise ValueError("combined_bounding_box is required when use_combined_mask is True")
                bb_x, bb_y, bb_width, bb_height = combined_bounding_box[0]
                paste_region = bbox_to_region((bb_x, bb_y, bb_width, bb_height), img.size)
                mask = combined_crop_mask[i]
            else:
                bb_x, bb_y, bb_width, bb_height = bbox
                paste_region = bbox_to_region((bb_x, bb_y, bb_width, bb_height), img.size)
                mask = cropped_masks[i]

            paste_region = scale_region_around_center(paste_region, crop_rescale, img.size)

            crop_width = max(1, round(paste_region[2] - paste_region[0]))
            crop_height = max(1, round(paste_region[3] - paste_region[1]))
            crop = crop.resize((crop_width, crop_height))
            crop_img = crop.convert("RGB")

            if border_blending > 1.0:
                border_blending = 1.0
            elif border_blending < 0.0:
                border_blending = 0.0

            blend_ratio = (max(crop_img.size) / 2) * float(border_blending)
            blend = img.convert("RGBA")

            if use_square_mask:
                mask = Image.new("L", img.size, 0)
                mask_block = Image.new("L", (crop_width, crop_height), 255)
                mask_block = inset_border(mask_block, round(blend_ratio / 2), (0))
                mask.paste(mask_block, paste_region)
            else:
                original_mask = tensor2pil(mask)[0]
                original_mask = original_mask.resize((crop_width, crop_height))
                mask = Image.new("L", img.size, 0)
                mask.paste(original_mask, paste_region)

            mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

            blend.paste(crop_img, paste_region)
            blend.putalpha(mask)

            img = Image.alpha_composite(img.convert("RGBA"), blend)
            out_images.append(img.convert("RGB"))

        return (pil2tensor(out_images),)
