"""
Preprocessing Pipeline
Complete pipeline for preparing handwriting images for the CNN model.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2


class PreprocessingPipeline:
    """
    Full preprocessing pipeline for handwriting images.

    Steps:
    1. Grayscale conversion
    2. Binarization (Otsu's method)
    3. Noise removal (morphological operations)
    4. Skew correction
    5. Normalization (resize & scale)
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        binarize_threshold: Optional[int] = None,
        denoise_kernel_size: int = 3,
        skip_skew: bool = False,
    ):
        self.target_size = target_size
        self.binarize_threshold = binarize_threshold
        self.denoise_kernel_size = denoise_kernel_size
        self.skip_skew = skip_skew

    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Apply binarization using Otsu's method or fixed threshold."""
        if self.binarize_threshold is None:
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(
                image, self.binarize_threshold, 255, cv2.THRESH_BINARY
            )
        return binary

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using morphological operations."""
        kernel = np.ones((self.denoise_kernel_size, self.denoise_kernel_size), np.uint8)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        return closed

    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct skew angle."""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 5:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 15:
            return image  # Skip extreme angles
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Resize and normalize pixel values."""
        resized = cv2.resize(image, self.target_size)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def process(self, image: np.ndarray) -> np.ndarray:
        """Run full preprocessing pipeline."""
        gray = self.grayscale(image)
        binary = self.binarize(gray)
        denoised = self.denoise(binary)
        deskewed = denoised if self.skip_skew else self.correct_skew(denoised)
        normalized = self.normalize(deskewed)
        return normalized

    def process_file(self, input_path: str, output_path: str) -> bool:
        """Process a single image file."""
        try:
            image = cv2.imread(input_path)
            if image is None:
                print(f"Could not read image: {input_path}")
                return False
            processed = self.process(image)
            cv2.imwrite(output_path, (processed * 255).astype(np.uint8))
            return True
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return False

    def process_directory(self, input_dir: str, output_dir: str, emotion: str) -> int:
        """Process all images in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir) / emotion
        output_path.mkdir(parents=True, exist_ok=True)

        extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        processed_count = 0

        for img_file in input_path.iterdir():
            if img_file.suffix.lower() in extensions:
                out_file = output_path / f"{img_file.stem}.png"
                if self.process_file(str(img_file), str(out_file)):
                    processed_count += 1

        return processed_count


if __name__ == "__main__":
    from ..utils.config import config

    pipeline = PreprocessingPipeline(
        target_size=config.preprocessing.target_size,
        binarize_threshold=config.preprocessing.binarize_threshold,
        denoise_kernel_size=config.preprocessing.denoise_kernel_size,
    )

    for emotion in ["HAPPY", "SAD"]:
        input_dir = Path(config.data.labeled_dir) / emotion
        if input_dir.exists():
            count = pipeline.process_directory(
                str(input_dir), config.data.processed_dir, emotion
            )
            print(f"{emotion}: {count} images processed")
        else:
            print(f"Skipping {emotion}: {input_dir} not found")
