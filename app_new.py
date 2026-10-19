import streamlit as st

from processing import bgr2rgb, load_image, run_pipeline

st.set_page_config(page_title="Image Processing with OpenCV", layout="wide")
st.title("🖼️ Image Processing Application")

# ---------- Sidebar: tải ảnh + tham số ----------
with st.sidebar:
    st.header("1. Upload Image")
    uploaded = st.file_uploader("Select an image (JPG/PNG)", type=["jpg", "jpeg", "png", "bmp"])

    st.header("2. Parameters")
    method = st.radio("Denoising Filter", ["Median", "Gaussian"],
                      help="Median is effective for salt-and-pepper noise; Gaussian smooths uniformly.")
    ksize = st.slider("Kernel Size", 3, 15, 5, step=2)
    st.divider()
    sobel_t = st.slider("Sobel Threshold", 0, 255, 60)
    prewitt_t = st.slider("Prewitt Threshold", 0, 255, 60)
    canny_low = st.slider("Canny - Low Threshold", 0, 255, 80)
    canny_high = st.slider("Canny - High Threshold", 0, 255, 160)

if uploaded is None:
    st.info("👈 Please upload an image on the left sidebar to get started.")
    st.stop()

# ---------- Gọi backend ----------
try:
    img = load_image(uploaded.getvalue())
except ValueError as e:
    st.error(str(e))
    st.stop()

r = run_pipeline(img, method, ksize, sobel_t, prewitt_t, canny_low, canny_high)

# ---------- Các khối hiển thị ----------
def show_original():
    st.subheader("Original Image")
    st.image(bgr2rgb(r["original"]), caption="Original color image uploaded",
             use_container_width=True)


def show_denoise():
    st.subheader("Denoising/Smoothing")
    st.image(bgr2rgb(r["denoised"]),
             caption=f"Image after {method} filtering",
             use_container_width=True)


def show_sharpen():
    st.subheader("Sharpening")
    st.image(bgr2rgb(r["sharpened"]),
             caption="Image after sharpening",
             use_container_width=True)


def show_edges():
    st.subheader("Edge Detection")
    c1, c2, c3 = st.columns(3)
    c1.image(r["sobel"], use_container_width=True,
             caption=f"Sobel")
    c2.image(r["prewitt"], use_container_width=True,
             caption=f"Prewitt")
    c3.image(r["canny"], use_container_width=True,
             caption=f"Canny")
    with st.expander("View gradient magnitude before thresholding"):
        g1, g2 = st.columns(2)
        g1.image(r["sobel_mag"], caption="Sobel Gradient Magnitude", use_container_width=True)
        g2.image(r["prewitt_mag"], caption="Prewitt Gradient Magnitude", use_container_width=True)


# ---------- Chọn bước xem ----------
view = st.radio(
    "Select a step to view:",
    ["All Steps", "Original Image", "Denoising/Smoothing", "Sharpening", "Edge Detection"],
    horizontal=True,
)

if view == "All Steps":
    c1, c2, c3 = st.columns(3)
    with c1:
        show_original()
    with c2:
        show_denoise()
    with c3:
        show_sharpen()
    st.divider()
    show_edges()
elif view == "Original Image":
    show_original()
elif view == "Denoising/Smoothing":
    show_denoise()
elif view == "Sharpening":
    show_sharpen()
else:
    show_edges()
