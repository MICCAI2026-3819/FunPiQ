import math

import torch
from PIL import Image, ImageDraw


def random_fov(
    image: Image.Image,
    p: float,
    ratio_range: tuple[float, float],
    pixel_limits: int,
) -> Image.Image:
    if torch.rand(1).item() > p:
        return image

    width, height = image.size
    original_radius = width / 2
    min_ratio, max_ratio = ratio_range

    assert 0 < min_ratio <= max_ratio <= 1, "ratio_range should be in (0, 1] and min_ratio <= max_ratio"
    assert pixel_limits > 0, "pixel_limits should be positive"
    assert abs(height - width) < 1, "image should be square"

    if min_ratio * width < pixel_limits:
        min_ratio = pixel_limits / width

    ratio = min_ratio + (max_ratio - min_ratio) * torch.rand(1).item()
    new_radius = original_radius * ratio
    sample_radius = original_radius - new_radius
    center_x, center_y = width / 2, height / 2
    distance = sample_radius * math.sqrt(torch.rand(1).item())
    theta = 2 * math.pi * torch.rand(1).item()
    new_center_x = center_x + distance * math.cos(theta)
    new_center_y = center_y + distance * math.sin(theta)

    left = max(0, int(new_center_x - new_radius))
    right = min(width, int(new_center_x + new_radius))
    top = max(0, int(new_center_y - new_radius))
    bottom = min(height, int(new_center_y + new_radius))
    cropped = image.crop((left, top, right, bottom))

    crop_width, crop_height = cropped.size
    mask = Image.new("L", (crop_width, crop_height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, crop_width, crop_height), fill=255)
    return Image.composite(
        cropped,
        Image.new("RGB", (crop_width, crop_height), (0, 0, 0)),
        mask,
    )


class RandomFOV:
    def __init__(
        self,
        p: float = 0.5,
        ratio_range: tuple[float, float] = (0.2, 1.0),
        pixel_limits: int = 64,
        schedule: str | None = None,
        start_epoch: int = 0,
        end_epoch: int = 100,
        start_p: float = 1.0,
        end_p: float = 0.0,
    ) -> None:
        self.p = p
        self.ratio_range = ratio_range
        self.pixel_limits = pixel_limits
        self.schedule = schedule
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_p = start_p
        self.end_p = end_p

    def __call__(self, img: Image.Image) -> Image.Image:
        return random_fov(
            img,
            p=self.p,
            ratio_range=self.ratio_range,
            pixel_limits=self.pixel_limits,
        )

    def on_epoch_start(self, epoch: int) -> None:
        if self.schedule is None:
            return

        if self.schedule.lower() != "linear":
            raise ValueError(f"Unknown schedule: {self.schedule}")

        if epoch < self.start_epoch:
            self.p = self.start_p
        elif epoch > self.end_epoch:
            self.p = self.end_p
        else:
            ratio = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            self.p = self.start_p + (self.end_p - self.start_p) * ratio

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(p={self.p}, "
            f"ratio_range={self.ratio_range}, "
            f"pixel_limits={self.pixel_limits})"
        )
