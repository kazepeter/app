import cv2
import numpy as np


def load_image(file_bytes: bytes, max_width: int = 1000) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    # chuyen đổi từ bytes sang mảng 1D numpy để cv2 có thể hiểu được
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # giải mã dữ liệu ảnh từ mảng 1D sang mảng 3D (H, W, 3)
    if img is None:
        raise ValueError("Unable to read the image. Please try a different JPG/PNG file.")
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    return img


def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def magnitude_to_uint8(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Độ lớn gradient sqrt(gx² + gy²), chuẩn hoá về [0, 255]."""
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def step1_denoise(img: np.ndarray, method: str = "Median", ksize: int = 5) -> np.ndarray:
    ksize = ksize if ksize % 2 == 1 else ksize + 1  
    if method == "Median":
        return cv2.medianBlur(img, ksize)
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


SHARPEN_KERNEL = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)


def step2_sharpen(img: np.ndarray) -> np.ndarray:
    return cv2.filter2D(img, ddepth=-1, kernel=SHARPEN_KERNEL)

def step3a_sobel(img: np.ndarray, thresh: int = 60):
    """Trả về (độ lớn gradient, ảnh biên nhị phân)."""
    gray = to_gray(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = magnitude_to_uint8(gx, gy)
    _, edges = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)
    return mag, edges

PREWITT_X = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)
PREWITT_Y = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]], dtype=np.float32)


def step3b_prewitt(img: np.ndarray, thresh: int = 60):
    """Trả về (độ lớn gradient, ảnh biên nhị phân)."""
    gray = to_gray(img).astype(np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, PREWITT_X)
    gy = cv2.filter2D(gray, cv2.CV_32F, PREWITT_Y)
    mag = magnitude_to_uint8(gx, gy)
    _, edges = cv2.threshold(mag, thresh, 255, cv2.THRESH_BINARY)
    return mag, edges

def step3c_canny(img: np.ndarray, low: int = 80, high: int = 160) -> np.ndarray:
    return cv2.Canny(to_gray(img), low, high)


def run_pipeline(img, method="Median", ksize=5,
                 sobel_t=60, prewitt_t=60, canny_low=80, canny_high=160) -> dict:
    denoised = step1_denoise(img, method, ksize)
    sharpened = step2_sharpen(denoised)
    sobel_mag, sobel_edges = step3a_sobel(sharpened, sobel_t)
    prewitt_mag, prewitt_edges = step3b_prewitt(sharpened, prewitt_t)
    canny_edges = step3c_canny(sharpened, canny_low, canny_high)
    return {
        "original": img, "denoised": denoised, "sharpened": sharpened,
        "sobel_mag": sobel_mag, "sobel": sobel_edges,
        "prewitt_mag": prewitt_mag, "prewitt": prewitt_edges,
        "canny": canny_edges,
    }
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    with open("sample.jpg", "rb") as f:        
        img = load_image(f.read())
    out = run_pipeline(img)

    titles = ["Gốc", "Khử nhiễu", "Sắc nét", "Sobel", "Prewitt", "Canny"]
    keys = ["original", "denoised", "sharpened", "sobel", "prewitt", "canny"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, t, k in zip(axes.ravel(), titles, keys):
        im = out[k]
        ax.imshow(bgr2rgb(im), cmap="gray" if im.ndim == 2 else None)
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
