"""
app_demo.py - Giao diện Streamlit Demo:
1. Chế độ chính: Tự động phát hiện và định vị khuôn mặt trên ảnh đầy đủ (Full Image Face Detection).
2. Chế độ bổ trợ: Minh họa phân loại ảnh con 24×24 pixel trên tập huấn luyện (Sub-window Classification).

Dự án: Phát hiện khuôn mặt người trong ảnh (Tuần 6)
Thư mục: FaceDetection_Week4

Cách chạy:
    python -m streamlit run app_demo.py
    hoặc:
    streamlit run app_demo.py
"""

from pathlib import Path
import io
import json
import random
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import joblib

from haar_extractor import extract_haar_features, predict_single_image
from face_detector import FullImageFaceDetector, draw_bounding_boxes

# Đường dẫn thư mục tương đối
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "demo"
RAW_FACES_DIR = BASE_DIR / "data" / "raw" / "faces" / "lfw_cache" / "lfw_home" / "lfw_funneled"
RAW_BG_DIR = BASE_DIR / "data" / "raw" / "backgrounds" / "caltech101" / "BACKGROUND_Google"

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Face Detection System - Tuần 6",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_project_data():
    """Nạp dữ liệu từ thư mục data/processed/ cho chế độ minh họa ảnh con 24x24."""
    images_path = DATA_DIR / "images_24x24.npy"
    labels_path = DATA_DIR / "labels.npy"
    metadata_path = DATA_DIR / "metadata.csv"
    splits_path = DATA_DIR / "split_indices.npz"

    missing = [str(p) for p in [images_path, labels_path, metadata_path, splits_path] if not p.exists()]
    if missing:
        return None, f"Thiếu các tệp dữ liệu bắt buộc: {', '.join(missing)}"

    images = np.load(images_path)
    labels = np.load(labels_path)
    metadata = pd.read_csv(metadata_path)
    splits = np.load(splits_path)

    train_indices = splits["train"]
    if len(train_indices) != 2511:
        return None, f"Số lượng mẫu tập train không đúng ({len(train_indices)} != 2511)"

    train_meta = metadata.iloc[train_indices].copy().reset_index(drop=False)
    train_meta.rename(columns={"index": "global_row_index"}, inplace=True)

    return {
        "images": images,
        "labels": labels,
        "train_meta": train_meta,
        "train_indices": train_indices
    }, None


@st.cache_data
def load_demo_metadata():
    """Nạp tệp thông tin mô hình demo_metadata.json."""
    meta_path = MODELS_DIR / "demo_metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_pipeline_model(filename: str):
    """Nạp mô hình Pipeline từ models/demo/<filename>."""
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        return None
    return joblib.load(model_path)


# ==============================================================================
# CHẾ ĐỘ 1: TỰ ĐỘNG PHÁT HIỆN VÀ ĐỊNH VỊ KHUÔN MẶT TRÊN ẢNH ĐẦY ĐỦ (MAIN MODE)
# ==============================================================================
def render_full_image_detection_mode(models_info: dict, demo_meta: dict):
    st.subheader("🎯 Phát hiện khuôn mặt tự động trên ảnh đầy đủ")
    st.markdown(
        "Hệ thống tự động quét đa tỷ lệ (Kim tự tháp ảnh), trích xuất $258$ đặc trưng Haar-like từ các cửa sổ con, "
        "dự đoán bằng Pipeline đã huấn luyện và áp dụng **Non-Maximum Suppression (NMS)** để định vị khuôn mặt người."
    )

    # Khung lưu ý khoa học và phạm vi
    st.info(
        "ℹ️ **Nguyên lý hoạt động và Phạm vi:**\n"
        "- **Phương pháp cài đặt:** Bộ phân loại 258 Haar-like + Quét cửa sổ trượt đa tỷ lệ (Sliding Window Pyramid) + Triệt tiêu cực đại cục bộ (NMS).\n"
        "- **Lưu ý đánh giá:** Các chỉ số kiểm định chéo và test trong đồ án đánh giá năng lực **phân loại trên từng cửa sổ ảnh 24×24**. "
        "Chất lượng định vị trên ảnh đầy đủ (tọa độ khung bao) là bài toán mở rộng và cần được kiểm chứng riêng bằng tập ảnh có nhãn hộp chuẩn (Ground-truth Bounding Boxes).\n"
        "- **Bảo mật dữ liệu:** Ảnh tải lên chỉ dùng để thực hiện suy luận trong phiên làm việc hiện tại, tuyệt đối không được thêm vào dữ liệu huấn luyện."
    )

    # 1. Chọn mô hình suy luận
    st.sidebar.header("⚙️ Cấu hình phát hiện")
    ml_models = [
        "AdaBoost",
        "SVM RBF",
        "Random Forest",
        "Logistic Regression",
        "k-NN",
        "Decision Tree"
    ]
    selected_model_name = st.sidebar.selectbox(
        "Chọn mô hình phát hiện:",
        options=ml_models,
        index=0,  # Mặc định AdaBoost
        help="AdaBoost (100 Decision Stumps) là thuật toán trọng tâm của đề tài."
    )

    pipe_info = models_info.get(selected_model_name, {})
    pipeline_obj = load_pipeline_model(pipe_info.get("filename", ""))

    if pipeline_obj is None:
        st.error(f"❌ Không tìm thấy tệp mô hình `{pipe_info.get('filename')}` trong `models/demo/`!")
        return

    # 2. Tham số quét cửa sổ và NMS
    with st.sidebar.expander("🛠️ Tham số quét cửa sổ & NMS", expanded=False):
        stride = st.slider("Bước trượt pixel (Stride):", min_value=2, max_value=12, value=6, step=2,
                           help="Bước trượt càng nhỏ thì quét càng kỹ nhưng tăng số lượng cửa sổ.")
        scale_factor = st.slider("Hệ số thu nhỏ (Scale factor):", min_value=1.1, max_value=1.4, value=1.2, step=0.05,
                                 help="Tỷ lệ co nhỏ ảnh qua các bậc kim tự tháp.")
        min_face = st.slider("Kích thước mặt nhỏ nhất (px):", min_value=24, max_value=120, value=36, step=6,
                             help="Bỏ qua các cửa sổ nhỏ hơn kích thước này.")
        max_face = st.slider("Kích thước mặt lớn nhất (px):", min_value=100, max_value=600, value=300, step=20,
                             help="Bỏ qua các cửa sổ lớn hơn kích thước này.")
        iou_thresh = st.slider("Ngưỡng IoU của NMS:", min_value=0.1, max_value=0.6, value=0.3, step=0.05,
                               help="Khung có IoU lớn hơn ngưỡng này với khung điểm cao hơn sẽ bị triệt tiêu.")

        # Lựa chọn chế độ ra quyết định
        decision_mode = st.radio(
            "Cơ chế lọc cửa sổ ứng viên:",
            options=["Mặc định: Pipeline.predict()", "Tùy chỉnh ngưỡng điểm số (Thử nghiệm Demo)"],
            index=0,
            help="Chế độ mặc định dùng đúng Pipeline.predict(). Chế độ thử nghiệm cho phép đặt ngưỡng score tùy biến cho demo."
        )

        use_custom_th = (decision_mode == "Tùy chỉnh ngưỡng điểm số (Thử nghiệm Demo)")
        custom_score_th = None
        if use_custom_th:
            if pipe_info.get("score_method") == "decision_function":
                # Thử nghiệm ngưỡng điểm quyết định
                default_th = float(pipe_info.get("selected_threshold", 0.0) or 0.0)
                custom_score_th = st.slider("Ngưỡng điểm quyết định (score >= th):",
                                            min_value=-2.0, max_value=2.0, value=default_th, step=0.05)
            else:
                # Xác suất
                default_th = float(pipe_info.get("selected_threshold", 0.5) or 0.5)
                custom_score_th = st.slider("Ngưỡng xác suất lớp face (prob >= th):",
                                            min_value=0.1, max_value=0.95, value=default_th, step=0.05)
            st.caption("⚠️ **Ghi chú:** Đây là cấu hình thử nghiệm riêng cho bộ phát hiện trên toàn ảnh, không thay thế cho giao thức đánh giá phân loại ảnh con tuần 6.")

    # 3. Lựa chọn nguồn ảnh: Tải ảnh từ máy tính HOẶC Chọn ảnh mẫu có sẵn của dự án
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Nguồn ảnh đầu vào")
    img_source_type = st.sidebar.radio(
        "Chọn nguồn ảnh:",
        options=["Tải ảnh từ máy tính (JPG/PNG)", "Sử dụng ảnh mẫu có sẵn của dự án"],
        index=0
    )

    loaded_pil_image = None
    image_title = ""

    if img_source_type == "Tải ảnh từ máy tính (JPG/PNG)":
        uploaded_file = st.sidebar.file_uploader(
            "Chọn tệp ảnh từ máy tính:",
            type=["jpg", "jpeg", "png"],
            help="Tải lên ảnh đầy đủ chứa một hoặc nhiều khuôn mặt, hoặc ảnh nền để kiểm tra báo động giả."
        )
        if uploaded_file is not None:
            try:
                loaded_pil_image = Image.open(uploaded_file)
                loaded_pil_image = ImageOps.exif_transpose(loaded_pil_image)
                image_title = uploaded_file.name
            except Exception as e:
                st.error(f"❌ Không mở được tệp ảnh đã tải: {e}")
                return
    else:
        # Danh sách ảnh mẫu thực tế từ dự án
        sample_options = {
            "Ảnh mặt 1 (LFW: Aaron Eckhart)": RAW_FACES_DIR / "Aaron_Eckhart" / "Aaron_Eckhart_0001.jpg",
            "Ảnh mặt 2 (LFW: Colin Powell)": RAW_FACES_DIR / "Colin_Powell" / "Colin_Powell_0001.jpg",
            "Ảnh mặt 3 (LFW: George W Bush)": RAW_FACES_DIR / "George_W_Bush" / "George_W_Bush_0001.jpg",
            "Ảnh nền 1 (Caltech: Không có mặt #0003)": RAW_BG_DIR / "image_0003.jpg",
            "Ảnh nền 2 (Caltech: Cảnh phức tạp #0001)": RAW_BG_DIR / "image_0001.jpg",
        }
        chosen_sample = st.sidebar.selectbox("Chọn ảnh mẫu dự án:", options=list(sample_options.keys()))
        sample_p = sample_options[chosen_sample]
        if sample_p.exists():
            loaded_pil_image = Image.open(sample_p)
            loaded_pil_image = ImageOps.exif_transpose(loaded_pil_image)
            image_title = sample_p.name
        else:
            st.sidebar.warning(f"Không tìm thấy ảnh mẫu: {sample_p}")

    # ==========================
    # HIỂN THỊ VÀ THỰC THI PHÁT HIỆN
    # ==========================
    if loaded_pil_image is None:
        st.info("👈 Vui lòng tải một ảnh từ máy tính hoặc chọn ảnh mẫu có sẵn ở thanh bên trái để bắt đầu phát hiện.")
        return

    col_view1, col_view2 = st.columns([1, 1], gap="medium")
    with col_view1:
        st.markdown(f"**Ảnh gốc:** `{image_title}` ({loaded_pil_image.width} × {loaded_pil_image.height} px)")
        st.image(loaded_pil_image, use_container_width=True)

    btn_detect = st.button("🔍 Bắt đầu phát hiện khuôn mặt", type="primary", use_container_width=True)

    if btn_detect:
        # Khởi tạo bộ phát hiện FullImageFaceDetector
        detector = FullImageFaceDetector(
            pipeline=pipeline_obj,
            scale_factor=scale_factor,
            stride=stride,
            min_face_size=min_face,
            max_face_size=max_face,
            iou_threshold=iou_thresh,
            score_threshold=custom_score_th,
            use_threshold=use_custom_th,
            max_image_dim=800,
            batch_size=512
        )

        prog_bar = st.progress(0.0)
        status_txt = st.empty()

        def on_progress(frac: float, msg: str):
            prog_bar.progress(frac)
            status_txt.text(msg)

        # Thực thi quét và suy luận
        result = detector.detect(loaded_pil_image, progress_callback=on_progress)
        prog_bar.empty()
        status_txt.empty()

        detections = result["detections"]
        num_faces = len(detections)

        st.markdown("---")
        # 1. Thống kê kết quả
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Khuôn mặt phát hiện", f"{num_faces} vùng")
        with m2:
            st.metric("Tổng cửa sổ đã quét", f"{result['total_windows_scanned']:,}")
        with m3:
            st.metric("Cửa sổ dương tính (trước NMS)", f"{result['positive_windows_before_nms']:,}")
        with m4:
            st.metric("Thời gian xử lý thực tế", f"{result['time_total_seconds']:.2f} s")

        # 2. Hiển thị ảnh kết quả
        if num_faces > 0:
            annotated_img = draw_bounding_boxes(loaded_pil_image, detections, line_width=3, color="#00FF00")
            
            with col_view2:
                st.markdown(f"**Ảnh kết quả phát hiện:** (Tìm thấy **{num_faces}** khuôn mặt)")
                st.image(annotated_img, use_container_width=True)

                # Nút tải ảnh kết quả
                buf = io.BytesIO()
                annotated_img.save(buf, format="JPEG", quality=95)
                st.download_button(
                    label="💾 Tải ảnh kết quả về máy",
                    data=buf.getvalue(),
                    file_name=f"detected_{image_title}",
                    mime="image/jpeg",
                    use_container_width=True
                )

            # 3. Bảng tọa độ và điểm số từng khung
            st.markdown("#### 📋 Chi tiết tọa độ các khung bao và điểm số:")
            rows = []
            for idx, det in enumerate(detections, start=1):
                s_val = det["score"]
                if result["score_type"] == "decision_function":
                    score_repr = f"{s_val:+.4f}"
                else:
                    score_repr = f"{s_val:.4f}"

                rows.append({
                    "Khung #": idx,
                    "Tọa độ x (px)": det["x"],
                    "Tọa độ y (px)": det["y"],
                    "Chiều rộng w (px)": det["w"],
                    "Chiều cao h (px)": det["h"],
                    "Loại điểm số": result["score_label"],
                    "Điểm số": score_repr
                })
            df_det = pd.DataFrame(rows)
            st.dataframe(df_det, use_container_width=True)

            # 4. Tính năng giải thích: Xem ảnh con 24x24 tương ứng với khung phát hiện
            st.markdown("#### 🔬 Phân tích giải thích: Kiểm tra ảnh con 24×24 tương ứng của từng khung")
            st.caption(
                "Mỗi khung bao trên ảnh lớn thực chất được trích xuất và chuẩn hóa về ma trận điểm ảnh 24×24, "
                "sau đó tính ảnh tích phân và trích xuất 258 đặc trưng Haar-like để đưa vào mô hình."
            )
            selected_box_idx = st.selectbox(
                "Chọn khung bao muốn kiểm tra:",
                options=list(range(1, num_faces + 1)),
                format_func=lambda i: f"Khung #{i} (Điểm số: {rows[i-1]['Điểm số']})"
            )
            chosen_det = detections[selected_box_idx - 1]
            patch_arr = chosen_det["patch_24x24"]

            col_p1, col_p2 = st.columns([1, 2])
            with col_p1:
                patch_pil = Image.fromarray(patch_arr, mode="L")
                patch_enlarged = patch_pil.resize((192, 192), resample=Image.NEAREST)
                st.image(patch_enlarged, caption=f"Ảnh con 24×24 phóng to 8x (Khung #{selected_box_idx})", width=192)
            with col_p2:
                st.markdown(f"- **Vị trí trên ảnh gốc:** `x={chosen_det['x']}, y={chosen_det['y']}, w={chosen_det['w']}, h={chosen_det['h']}`")
                st.markdown(f"- **{result['score_label']}:** `{chosen_det['score']:+.6f}`")
                
                # Trích xuất vector đặc trưng trực tiếp để chứng minh
                f_vec = extract_haar_features(patch_arr)[0]
                st.markdown(f"- **Vector đặc trưng:** `(1, 258)`, min=`{f_vec.min():.4f}`, max=`{f_vec.max():.4f}`, mean=`{f_vec.mean():.4f}`")
                top_indices = np.argsort(np.abs(f_vec))[-5:][::-1]
                st.markdown(f"- **Top 5 đặc trưng mạnh nhất:** `{', '.join([f'feat_{idx}: {f_vec[idx]:+.4f}' for idx in top_indices])}`")

        else:
            with col_view2:
                st.warning("⚠️ **Không phát hiện được khuôn mặt với cấu hình hiện tại.**")
                st.markdown(
                    "- **Giải thích nguyên nhân:** Tất cả các cửa sổ quét đều không thỏa mãn điều kiện dương tính của mô hình hoặc điểm số nằm dưới ngưỡng thiết lập.\n"
                    "- **Gợi ý thao tác:** Bạn có thể thử:\n"
                    "  + Giảm kích thước mặt nhỏ nhất (`min_face_size`).\n"
                    "  + Giảm bước trượt (`stride`) để quét dày hơn.\n"
                    "  + Sử dụng chế độ *Tùy chỉnh ngưỡng điểm số* để hạ thấp ngưỡng lọc nếu khuôn mặt có độ tương phản mờ."
                )

        st.caption(
            "📌 **Tính trung thực khoa học:** Do ảnh tải lên chưa có nhãn hoặc hộp bao chuẩn (Ground Truth), "
            "hệ thống tuyệt đối không tự tính điểm Accuracy hay khẳng định kết quả là đúng/sai tuyệt đối. "
            "Các điểm quyết định hiển thị nguyên bản theo thang đo khoảng cách lề của thuật toán, không chuyển thành phần trăm xác suất giả định."
        )


# ==============================================================================
# CHẾ ĐỘ 2: MINH HỌA PHÂN LOẠI ẢNH CON 24x24 TRÊN TẬP TRAIN (SUB-WINDOW MODE)
# ==============================================================================
def render_patch_classification_mode(data_bundle: dict, models_info: dict, demo_meta: dict):
    st.subheader("🔬 Minh họa phân loại ảnh con kích thước 24×24 pixel (Tập huấn luyện)")
    st.markdown(
        "Chế độ này phục vụ việc **giải thích cơ chế hoạt động của các bộ phân loại** trên từng ảnh con đơn lẻ, "
        "đối chiếu nhãn dự đoán với nhãn thực tế đã gán nhãn trong tập huấn luyện ($2.511$ mẫu)."
    )

    st.warning(
        "⚠️ **Khuyến cáo khoa học:** Mẫu được chọn thuộc tập huấn luyện (*train split*), "
        "dùng cho mục đích minh họa thao tác trích xuất và đối chiếu. "
        "Kết quả trên tập này không thay thế cho đánh giá khách quan trên tập validation/test đã khóa trong báo cáo."
    )

    images_all = data_bundle["images"]
    train_meta = data_bundle["train_meta"]
    train_indices = list(data_bundle["train_indices"])

    # Sidebar: Chọn mô hình và mẫu
    st.sidebar.header("⚙️ Cấu hình phân loại ảnh con")
    ml_model_names = [
        "AdaBoost",
        "SVM RBF",
        "Random Forest",
        "Logistic Regression",
        "k-NN",
        "Decision Tree"
    ]
    selected_model_name = st.sidebar.selectbox(
        "Chọn mô hình học máy:",
        options=ml_model_names,
        index=0
    )

    show_baseline = st.sidebar.checkbox(
        "Bổ sung mốc cơ sở (DummyClassifier)",
        value=False,
        help="DummyClassifier luôn dự đoán lớp đa số (Face = 1), dùng làm mốc so sánh cơ sở."
    )

    apply_threshold = st.sidebar.checkbox(
        "Áp dụng ngưỡng quyết định đã chọn trên Validation",
        value=True,
        help="Áp dụng quy tắc ngưỡng đã được khóa tại Giao thức Tuần 6."
    )

    curr_info = models_info.get(selected_model_name, {})
    th_val = curr_info.get("selected_threshold") if apply_threshold else None

    with st.sidebar.expander(f"Chi tiết mô hình: {selected_model_name}", expanded=False):
        st.write(f"- **Mô tả:** {curr_info.get('description', '')}")
        st.write(f"- **Tệp lưu trữ:** `{curr_info.get('filename', '')}`")
        st.write(f"- **Loại điểm số:** {curr_info.get('score_label', '')}")
        st.write(f"- **Quy tắc quyết định:** `{curr_info.get('decision_rule', '')}`")

    st.sidebar.markdown("---")
    st.sidebar.header("🖼️ Chọn mẫu từ tập train")

    if "selected_global_idx" not in st.session_state:
        st.session_state["selected_global_idx"] = int(train_indices[0])

    col_rand1, col_rand2 = st.sidebar.columns([1, 1])
    with col_rand1:
        if st.button("🎲 Chọn ngẫu nhiên", use_container_width=True):
            st.session_state["selected_global_idx"] = int(random.choice(train_indices))
    with col_rand2:
        class_filter = st.selectbox("Lọc lớp:", options=["Tất cả", "Chỉ Face (1)", "Chỉ Non-face (0)"], index=0)

    if class_filter == "Chỉ Face (1)":
        filtered_meta = train_meta[train_meta["label"] == 1]
    elif class_filter == "Chỉ Non-face (0)":
        filtered_meta = train_meta[train_meta["label"] == 0]
    else:
        filtered_meta = train_meta

    filtered_global_indices = filtered_meta["global_row_index"].tolist()
    curr_g_idx = st.session_state["selected_global_idx"]
    if curr_g_idx not in filtered_global_indices:
        curr_g_idx = filtered_global_indices[0]
        st.session_state["selected_global_idx"] = curr_g_idx

    chosen_idx = st.sidebar.selectbox(
        "Chọn chỉ số dòng toàn cục:",
        options=filtered_global_indices,
        index=filtered_global_indices.index(curr_g_idx)
    )
    st.session_state["selected_global_idx"] = chosen_idx

    active_g_idx = st.session_state["selected_global_idx"]
    active_row = train_meta[train_meta["global_row_index"] == active_g_idx].iloc[0]
    active_image = images_all[active_g_idx]
    true_label = int(active_row["label"])
    true_label_str = "Khuôn mặt (Face = 1)" if true_label == 1 else "Không phải khuôn mặt (Non-face = 0)"

    # Khu vực chính
    col_img, col_pred = st.columns([1, 1.4], gap="large")

    with col_img:
        st.markdown(f"**Ảnh con 24×24 phóng to 8x** (Dòng toàn cục: `{active_g_idx}`)")
        pil_img = Image.fromarray(active_image, mode="L")
        pil_img_enlarged = pil_img.resize((192, 192), resample=Image.NEAREST)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.image(pil_img_enlarged, width=192)
        with c2:
            st.markdown(f"**Tập dữ liệu:** `Train`")
            st.markdown(f"**Nhãn thực tế:**")
            if true_label == 1:
                st.success(f"**{true_label_str}**")
            else:
                st.info(f"**{true_label_str}**")

        st.markdown(
            f"""
            | Siêu dữ liệu | Giá trị |
            | :--- | :--- |
            | `source_id` | `{active_row['source_id']}` |
            | `group` | `{active_row['group']}` |
            | Mã SHA-256 | `{active_row['sha256'][:16]}...` |
            """
        )

    with col_pred:
        st.markdown(f"**Mô hình đang kiểm tra:** `{selected_model_name}`")
        active_pipeline = load_pipeline_model(curr_info["filename"])

        col_b1, col_b2 = st.columns([1, 1.2])
        btn_predict_patch = col_b1.button("🔍 Dự đoán ảnh con này", type="primary", use_container_width=True)
        btn_compare_patch = col_b2.button("📊 So sánh sáu mô hình", use_container_width=True)

        if btn_predict_patch:
            th_to_use = curr_info.get("selected_threshold") if apply_threshold else None
            res = predict_single_image(active_pipeline, active_image, threshold=th_to_use)

            y_pred = res["y_pred_selected"]
            is_cor = (y_pred == true_label)

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Nhãn dự đoán", f"{y_pred} ({'Face' if y_pred == 1 else 'Non-face'})")
            with m_col2:
                st.metric("Nhãn thực tế", f"{true_label} ({'Face' if true_label == 1 else 'Non-face'})")
            with m_col3:
                if is_cor:
                    st.success("✅ **ĐÚNG**")
                else:
                    st.error("❌ **SAI**")

            s_val = res["score"]
            if s_val is not None:
                if res["score_type"] == "decision_function":
                    st.markdown(f"- **{res['score_label']}:** `{s_val:+.6f}`")
                else:
                    st.markdown(f"- **{res['score_label']}:** `{s_val:.6f}` ({s_val*100:.2f}%)")

            # Hiển thị đặc trưng Haar
            f_vec = res["features"][0]
            with st.expander("🔬 Xem chi tiết 258 đặc trưng Haar-like"):
                st.write(f"- Vector shape `(1, 258)`, min=`{f_vec.min():.6f}`, max=`{f_vec.max():.6f}`, mean=`{f_vec.mean():.6f}`")
                top_idx = np.argsort(np.abs(f_vec))[-5:][::-1]
                st.table(pd.DataFrame([{"Đặc trưng #": int(i), "Giá trị": f"{f_vec[i]:.6f}"} for i in top_idx]))

    if btn_compare_patch:
        st.markdown("---")
        st.subheader("📊 Bảng so sánh 6 mô hình trên mẫu ảnh con này:")
        models_to_test = list(ml_model_names)
        if show_baseline:
            models_to_test.append("DummyClassifier")

        comp_rows = []
        for m_name in models_to_test:
            m_info = models_info.get(m_name, {})
            p_obj = load_pipeline_model(m_info.get("filename"))
            if p_obj is None:
                continue

            th_app = m_info.get("selected_threshold") if apply_threshold else None
            p_res = predict_single_image(p_obj, active_image, threshold=th_app)
            pred_v = p_res["y_pred_selected"]
            is_c = (pred_v == true_label)
            sc_val = p_res["score"]

            score_str = f"{sc_val:+.4f}" if (sc_val is not None and p_res["score_type"] == "decision_function") else (
                f"{sc_val:.4f}" if sc_val is not None else "N/A"
            )

            comp_rows.append({
                "Mô hình": m_name,
                "Dự đoán": f"{pred_v} ({'Face' if pred_v == 1 else 'Non-face'})",
                "Thực tế": f"{true_label} ({'Face' if true_label == 1 else 'Non-face'})",
                "Kết quả": "✅ ĐÚNG" if is_c else "❌ SAI",
                "Loại điểm số": p_res["score_label"],
                "Điểm số": score_str,
                "Quy tắc áp dụng": m_info.get("decision_rule", "Pipeline.predict()")
            })

        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
        st.warning(
            "⚠️ **Lưu ý khoa học:** Không so sánh trực tiếp điểm số thô giữa các thuật toán có thang đo khác nhau. "
            "SVM dùng khoảng cách siêu phẳng, AdaBoost dùng tổng trọng số weak learners, Random Forest dùng tỷ lệ biểu quyết."
        )


# ==============================================================================
# HÀM CHÍNH (MAIN)
# ==============================================================================
def main():
    st.title("👤 Hệ thống Phát hiện và Định vị Khuôn mặt người trong ảnh")
    st.markdown("**Đề tài Nghiên cứu:** Học máy Tuần 6 — Tích hợp trích xuất Haar-like, Phân loại Pipeline và Quét đa tỷ lệ")

    # Nạp dữ liệu và metadata
    demo_meta = load_demo_metadata()
    if demo_meta is None:
        st.error(
            "❌ **Chưa tìm thấy mô hình trong thư mục `models/demo/`!**\n\n"
            "Vui lòng chạy lệnh sau từ Terminal để chuẩn bị mô hình:\n"
            "```bash\n"
            "python train_export_demo.py\n"
            "```"
        )
        return

    models_info = demo_meta.get("models", {})
    data_bundle, err = load_project_data()

    # Điều hướng chế độ hoạt động tại thanh bên
    st.sidebar.title("🎛️ Điều hướng Chế độ")
    app_mode = st.sidebar.radio(
        "Chọn chức năng làm việc:",
        options=[
            "🎯 Phát hiện khuôn mặt trong ảnh (Chế độ chính)",
            "🔬 Minh họa phân loại ảnh con (24×24)"
        ],
        index=0  # Mặc định mở chế độ phát hiện khuôn mặt trên ảnh đầy đủ
    )
    st.sidebar.markdown("---")

    if app_mode == "🎯 Phát hiện khuôn mặt trong ảnh (Chế độ chính)":
        render_full_image_detection_mode(models_info, demo_meta)
    else:
        if err is not None:
            st.error(f"❌ Không mở được chế độ phân loại ảnh con: {err}")
        else:
            render_patch_classification_mode(data_bundle, models_info, demo_meta)


if __name__ == "__main__":
    main()
