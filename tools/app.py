#!/usr/bin/env python3
"""
DETR 交通场景检测演示 Web UI
提供三个主要功能模块：
1. 预训练模型推理
2. COCO GT 可视化
3. 预测与 GT 对比展示
"""

import sys
from pathlib import Path
import json
from typing import Optional, Tuple, List
import io

import streamlit as st
from PIL import Image
import yaml

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# 导入自定义模块
try:
    from infer_pretrained import PretrainedDETRInference
    from viz_coco_gt import COCOGroundTruthVisualizer
except ImportError as e:
    st.error(f"模块导入失败: {e}")
    st.stop()


# ==================== 配置与常量 ====================

DEFAULT_CONFIG = {
    "model_name": "facebook/detr-resnet-50",
    "confidence_threshold": 0.8,
    "classes_yaml": "../configs/classes.yaml",
    "coco_datasets": {
        "BDD100K (train)": "../data/traffic_coco/bdd100k_det/annotations/instances_train.json",
        "BDD100K (val)": "../data/traffic_coco/bdd100k_det/annotations/instances_val.json",
        "CCTSDB (train)": "../data/traffic_coco/cctsdb_det/annotations/instances_train.json",
        "CCTSDB (test)": "../data/traffic_coco/cctsdb_det/annotations/instances_test.json",
        "TT100K (train)": "../data/traffic_coco/tt100k_det/annotations/instances_train.json"
    },
    "image_roots": {
        "BDD100K (train)": "../data/traffic_coco/bdd100k_det/images/train",
        "BDD100K (val)": "../data/traffic_coco/bdd100k_det/images/val",
        "CCTSDB (train)": "../data/traffic_coco/cctsdb_det/images/train",
        "CCTSDB (test)": "../data/traffic_coco/cctsdb_det/images/test",
        "TT100K (train)": "../data/traffic_coco/tt100k_det/images/train"
    }
}


# ==================== 辅助函数 ====================

@st.cache_resource
def load_inferencer(model_name: str, threshold: float, keep_labels: Optional[List[str]] = None, device: int = -1):
    """加载推理模型（带缓存）"""
    try:
        inferencer = PretrainedDETRInference(
            model_name=model_name,
            device=device,
            confidence_threshold=threshold,
            keep_labels=keep_labels,
            classes_yaml=DEFAULT_CONFIG["classes_yaml"]
        )
        return inferencer
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None


@st.cache_resource
def load_gt_visualizer(dataset_name: str):
    """加载 GT 可视化器（带缓存）"""
    try:
        coco_json = DEFAULT_CONFIG["coco_datasets"].get(dataset_name)
        image_root = DEFAULT_CONFIG["image_roots"].get(dataset_name)
        
        if not coco_json or not Path(coco_json).exists():
            st.warning(f"数据集 {dataset_name} 的 COCO JSON 文件不存在: {coco_json}")
            return None
        
        if not image_root or not Path(image_root).exists():
            st.warning(f"数据集 {dataset_name} 的图片根目录不存在: {image_root}")
            return None
        
        visualizer = COCOGroundTruthVisualizer(
            coco_json=coco_json,
            image_root=image_root,
            classes_yaml=DEFAULT_CONFIG["classes_yaml"]
        )
        return visualizer
    except Exception as e:
        st.error(f"GT 可视化器加载失败: {e}")
        return None


def pil_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """将 PIL Image 转换为字节流"""
    buf = io.BytesIO()
    image.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()


# ==================== 页面：预训练模型推理 ====================

def page_inference():
    """Tab 1: 预训练模型推理（支持上传和目录选图）"""
    st.header("🚗 预训练 DETR 模型推理")
    st.markdown("上传图片或从数据集中选择图片进行目标检测")
    
    # 侧边栏：模型配置
    with st.sidebar:
        st.subheader("模型配置")
        model_name = st.selectbox(
            "选择模型",
            ["facebook/detr-resnet-50", "facebook/detr-resnet-101"],
            index=0
        )
        threshold = st.slider("置信度阈值", 0.1, 0.9, 0.8, 0.05)
        
        # keep_labels 过滤
        keep_labels_input = st.text_input(
            "类别过滤 (可选)",
            placeholder="如: car, truck, bus",
            help="逗号分隔，留空表示不过滤"
        )
        keep_labels = None
        if keep_labels_input.strip():
            keep_labels = [label.strip() for label in keep_labels_input.split(",")]
        
        device = st.radio("设备", ["CPU", "GPU"], index=0)
        device_id = -1 if device == "CPU" else 0
    
    # 选择输入方式
    input_mode = st.radio("输入方式", ["上传图片", "从数据集选择"], horizontal=True)
    
    # 加载推理器（注意缓存键包含keep_labels）
    cache_key = f"{model_name}_{threshold}_{str(keep_labels)}_{device_id}"
    inferencer = load_inferencer(model_name, threshold, keep_labels, device_id)
    if inferencer is None:
        st.stop()
    
    if input_mode == "上传图片":
        # 文件上传模式
        uploaded_file = st.file_uploader(
            "上传图片",
            type=["jpg", "jpeg", "png", "bmp"],
            help="支持 JPG, PNG, BMP 格式"
        )
        
        if uploaded_file is not None:
            # 显示原图
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("原图")
                st.image(image, use_container_width=True)
            
            # 推理按钮
            if st.button("🔍 开始检测", type="primary"):
                with st.spinner("正在推理..."):
                    try:
                        # 保存临时文件
                        temp_path = "/tmp/uploaded_image.jpg"
                        image.save(temp_path)
                        
                        # 执行推理
                        detections, _ = inferencer.infer_single(temp_path)
                        
                        # 可视化
                        result_image = inferencer.visualize(image.copy(), detections)
                        
                        with col2:
                            st.subheader("检测结果")
                            st.image(result_image, use_container_width=True)
                        
                        # 显示检测详情
                        st.subheader(f"检测到 {len(detections)} 个目标")
                        
                        if detections:
                            # 创建表格
                            import pandas as pd
                            df = pd.DataFrame([
                                {
                                    "类别": det["label"],
                                    "置信度": f"{det['score']:.3f}",
                                    "边界框": f"[{det['box'][0]:.1f}, {det['box'][1]:.1f}, {det['box'][2]:.1f}, {det['box'][3]:.1f}]"
                                }
                                for det in detections
                            ])
                            st.dataframe(df, use_container_width=True)
                            
                            # 下载按钮
                            result_bytes = pil_to_bytes(result_image, format="JPEG")
                            st.download_button(
                                label="📥 下载结果图片",
                                data=result_bytes,
                                file_name="detection_result.jpg",
                                mime="image/jpeg"
                            )
                        
                    except Exception as e:
                        st.error(f"推理失败: {e}")
    
    else:
        # 从数据集选择模式
        dataset_name = st.selectbox(
            "选择数据集",
            list(DEFAULT_CONFIG["coco_datasets"].keys()),
            index=0,
            key="inf_dataset"
        )
        
        # 加载数据集
        visualizer = load_gt_visualizer(dataset_name)
        if visualizer is None:
            st.warning("数据集不可用")
            st.stop()
        
        # 图片选择器
        image_ids = list(visualizer.images.keys())
        selected_idx = st.slider(
            "选择图片",
            0,
            len(image_ids) - 1,
            0,
            key="inf_slider"
        )
        
        selected_image_id = image_ids[selected_idx]
        img_info = visualizer.images[selected_image_id]
        img_path = Path(visualizer.image_root) / img_info['file_name']
        
        st.info(f"**图片:** {img_info['file_name']} | **Image ID:** {selected_image_id}")
        
        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("原图")
                st.image(image, use_container_width=True)
            
            # 推理按钮
            if st.button("🔍 开始检测", type="primary", key="dataset_infer"):
                with st.spinner("正在推理..."):
                    try:
                        # 执行推理
                        detections, _ = inferencer.infer_single(str(img_path))
                        
                        # 可视化
                        result_image = inferencer.visualize(image.copy(), detections)
                        
                        with col2:
                            st.subheader("检测结果")
                            st.image(result_image, use_container_width=True)
                        
                        # 显示检测详情
                        st.subheader(f"检测到 {len(detections)} 个目标")
                        
                        if detections:
                            import pandas as pd
                            df = pd.DataFrame([
                                {
                                    "类别": det["label"],
                                    "置信度": f"{det['score']:.3f}",
                                    "边界框": f"[{det['box'][0]:.1f}, {det['box'][1]:.1f}, {det['box'][2]:.1f}, {det['box'][3]:.1f}]"
                                }
                                for det in detections
                            ])
                            st.dataframe(df, use_container_width=True)
                            
                            result_bytes = pil_to_bytes(result_image, format="JPEG")
                            st.download_button(
                                label="📥 下载结果图片",
                                data=result_bytes,
                                file_name=f"{img_info['file_name']}_pred.jpg",
                                mime="image/jpeg"
                            )
                        
                    except Exception as e:
                        st.error(f"推理失败: {e}")
        else:
            st.error(f"图片文件不存在: {img_path}")


# ==================== 页面：GT 可视化 ====================

def page_gt_visualization():
    """Tab 2: COCO GT 可视化"""
    st.header("📊 Ground Truth 可视化")
    st.markdown("浏览和可视化 COCO 格式数据集的标注信息")
    
    # 侧边栏：数据集选择
    with st.sidebar:
        st.subheader("数据集选择")
        dataset_name = st.selectbox(
            "选择数据集",
            list(DEFAULT_CONFIG["coco_datasets"].keys()),
            index=0
        )
        
        show_area = st.checkbox("显示面积", value=False)
        font_size = st.slider("字体大小", 10, 40, 20, 2)
        line_width = st.slider("边框粗细", 1, 8, 3, 1)
    
    # 加载可视化器
    visualizer = load_gt_visualizer(dataset_name)
    if visualizer is None:
        st.warning(f"数据集 {dataset_name} 不可用，请检查数据路径")
        st.stop()
    
    # 显示统计信息
    stats = visualizer.get_statistics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总图片数", stats["total_images"])
    with col2:
        st.metric("总标注数", stats["total_annotations"])
    with col3:
        st.metric("类别数", len(stats["categories"]))
    
    # 类别分布
    with st.expander("查看类别分布"):
        import pandas as pd
        cat_df = pd.DataFrame([
            {"类别": name, "数量": count, "占比": f"{count/stats['total_annotations']*100:.2f}%"}
            for name, count in sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(cat_df, use_container_width=True)
    
    # 图片选择器
    st.subheader("选择图片")
    image_ids = list(visualizer.images.keys())
    
    col_a, col_b = st.columns([3, 1])
    with col_a:
        selected_idx = st.slider(
            "图片索引",
            0,
            len(image_ids) - 1,
            0,
            help=f"共 {len(image_ids)} 张图片"
        )
    
    selected_image_id = image_ids[selected_idx]
    img_info = visualizer.images[selected_image_id]
    
    with col_b:
        st.info(f"**Image ID:** {selected_image_id}\n\n**文件名:** {img_info['file_name']}")
    
    # 类别过滤
    available_categories = list(stats["categories"].keys())
    category_filter = st.multiselect(
        "类别过滤（留空显示全部）",
        available_categories,
        default=None
    )
    
    # 可视化按钮
    if st.button("🎨 生成可视化", type="primary"):
        with st.spinner("正在绘制..."):
            try:
                result_image = visualizer.visualize_single(
                    image_id=selected_image_id,
                    output_path=None,
                    show_labels=True,
                    show_area=show_area,
                    font_size=font_size,
                    line_width=line_width,
                    category_filter=category_filter if category_filter else None
                )
                
                if result_image:
                    st.image(result_image, use_container_width=True, caption=img_info['file_name'])
                    
                    # 显示该图片的标注信息
                    annotations = visualizer.annotations_by_image.get(selected_image_id, [])
                    if category_filter:
                        annotations = [
                            ann for ann in annotations
                            if visualizer.categories[ann["category_id"]] in category_filter
                        ]
                    
                    st.info(f"该图片共有 {len(annotations)} 个标注")
                    
                    # 下载按钮
                    result_bytes = pil_to_bytes(result_image, format="JPEG")
                    st.download_button(
                        label="📥 下载 GT 可视化图片",
                        data=result_bytes,
                        file_name=f"{Path(img_info['file_name']).stem}_gt.jpg",
                        mime="image/jpeg"
                    )
                
            except Exception as e:
                st.error(f"可视化失败: {e}")


# ==================== 页面：预测与 GT 对比 ====================

def page_comparison():
    """Tab 3: 预测与 GT 对比"""
    st.header("🔬 预测结果与 GT 对比")
    st.markdown("将预训练模型的预测结果与数据集 Ground Truth 进行并排对比")
    
    # 侧边栏配置
    with st.sidebar:
        st.subheader("模型配置")
        model_name = st.selectbox(
            "模型",
            ["facebook/detr-resnet-50", "facebook/detr-resnet-101"],
            index=0,
            key="comp_model"
        )
        threshold = st.slider("置信度阈值", 0.1, 0.9, 0.8, 0.05, key="comp_threshold")
        device = st.radio("设备", ["CPU", "GPU"], index=0, key="comp_device")
        device_id = -1 if device == "CPU" else 0
        
        st.divider()
        
        st.subheader("数据集配置")
        dataset_name = st.selectbox(
            "数据集",
            list(DEFAULT_CONFIG["coco_datasets"].keys()),
            index=0,
            key="comp_dataset"
        )
    
    # 加载模型和数据
    inferencer = load_inferencer(model_name, threshold, None, device_id)
    visualizer = load_gt_visualizer(dataset_name)
    
    if inferencer is None or visualizer is None:
        st.warning("模型或数据集加载失败")
        st.stop()
    
    # 图片选择
    image_ids = list(visualizer.images.keys())
    selected_idx = st.slider(
        "选择图片",
        0,
        len(image_ids) - 1,
        0,
        key="comp_slider"
    )
    
    selected_image_id = image_ids[selected_idx]
    img_info = visualizer.images[selected_image_id]
    img_path = Path(visualizer.image_root) / img_info['file_name']
    
    st.info(f"**图片:** {img_info['file_name']} | **Image ID:** {selected_image_id}")
    
    # 对比按钮
    if st.button("🔄 生成对比", type="primary"):
        with st.spinner("正在生成对比结果..."):
            try:
                # 加载原图
                original_image = Image.open(img_path).convert("RGB")
                
                # 生成 GT 可视化
                gt_image = visualizer.visualize_single(
                    image_id=selected_image_id,
                    output_path=None,
                    show_labels=True
                )
                
                # 生成预测可视化
                detections, _ = inferencer.infer_single(str(img_path))
                pred_image = inferencer.visualize(original_image.copy(), detections)
                
                # 并排显示
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Ground Truth")
                    st.image(gt_image, use_container_width=True)
                    gt_anns = visualizer.annotations_by_image.get(selected_image_id, [])
                    st.caption(f"标注数量: {len(gt_anns)}")
                
                with col2:
                    st.subheader("模型预测")
                    st.image(pred_image, use_container_width=True)
                    st.caption(f"检测数量: {len(detections)}")
                
                # 详细对比表
                st.subheader("详细对比")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**GT 标注列表**")
                    if gt_anns:
                        import pandas as pd
                        gt_df = pd.DataFrame([
                            {
                                "类别": visualizer.categories[ann["category_id"]],
                                "面积": int(ann.get("area", 0))
                            }
                            for ann in gt_anns
                        ])
                        st.dataframe(gt_df, use_container_width=True)
                
                with col_b:
                    st.markdown("**模型预测列表**")
                    if detections:
                        import pandas as pd
                        pred_df = pd.DataFrame([
                            {
                                "类别": det["label"],
                                "置信度": f"{det['score']:.3f}"
                            }
                            for det in detections
                        ])
                        st.dataframe(pred_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"对比生成失败: {e}")


# ==================== 页面：批量导出与日志 ====================

def page_batch_export():
    """Tab 4: 批量推理与导出"""
    st.header("📦 批量推理与导出")
    st.markdown("批量处理数据集图片，导出推理结果和日志文件")
    
    # 侧边栏配置
    with st.sidebar:
        st.subheader("模型配置")
        model_name = st.selectbox(
            "模型",
            ["facebook/detr-resnet-50", "facebook/detr-resnet-101"],
            index=0,
            key="batch_model"
        )
        threshold = st.slider("置信度阈值", 0.1, 0.9, 0.8, 0.05, key="batch_threshold")
        
        keep_labels_input = st.text_input(
            "类别过滤 (可选)",
            placeholder="如: car, truck, bus",
            key="batch_keep_labels"
        )
        keep_labels = None
        if keep_labels_input.strip():
            keep_labels = [label.strip() for label in keep_labels_input.split(",")]
        
        device = st.radio("设备", ["CPU", "GPU"], index=0, key="batch_device")
        device_id = -1 if device == "CPU" else 0
    
    # 数据集选择
    dataset_name = st.selectbox(
        "选择数据集",
        list(DEFAULT_CONFIG["coco_datasets"].keys()),
        index=0,
        key="batch_dataset"
    )
    
    # 批量设置
    col1, col2 = st.columns(2)
    with col1:
        max_images = st.number_input(
            "最大处理数量",
            min_value=1,
            max_value=10000,
            value=100,
            help="限制批量处理的图片数量"
        )
    
    with col2:
        output_name = st.text_input(
            "输出目录名称",
            value="batch_export",
            help="将在 outputs/demo_pred/ 下创建子目录"
        )
    
    # 导出选项
    save_json = st.checkbox("保存 JSON 结果", value=True)
    save_log = st.checkbox("保存日志文件", value=True)
    
    # 批量处理按钮
    if st.button("🚀 开始批量处理", type="primary"):
        # 加载数据集
        visualizer = load_gt_visualizer(dataset_name)
        if visualizer is None:
            st.error("数据集加载失败")
            st.stop()
        
        # 加载推理器
        inferencer = load_inferencer(model_name, threshold, keep_labels, device_id)
        if inferencer is None:
            st.error("模型加载失败")
            st.stop()
        
        # 准备输出目录
        output_dir = Path(f"../outputs/demo_pred/{output_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取图片列表
        image_ids = list(visualizer.images.keys())[:max_images]
        
        # 进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        all_results = {}
        log_lines = []
        success_count = 0
        
        for idx, img_id in enumerate(image_ids):
            # 更新进度
            progress = (idx + 1) / len(image_ids)
            progress_bar.progress(progress)
            
            img_info = visualizer.images[img_id]
            status_text.text(f"处理中 [{idx+1}/{len(image_ids)}]: {img_info['file_name']}")
            
            try:
                img_path = Path(visualizer.image_root) / img_info['file_name']
                if not img_path.exists():
                    log_lines.append(f"[SKIP] {img_info['file_name']}: 文件不存在")
                    continue
                
                # 推理
                detections, image = inferencer.infer_single(str(img_path))
                
                # 保存可视化
                output_path = output_dir / f"{Path(img_info['file_name']).stem}_pred.jpg"
                inferencer.visualize(image, detections, str(output_path))
                
                # 记录结果
                all_results[img_info['file_name']] = {
                    "image_id": img_id,
                    "num_detections": len(detections),
                    "detections": detections
                }
                
                log_lines.append(f"[OK] {img_info['file_name']}: {len(detections)} 个目标")
                success_count += 1
                
            except Exception as e:
                log_lines.append(f"[ERROR] {img_info['file_name']}: {str(e)}")
                continue
        
        # 保存 JSON
        if save_json:
            json_path = output_dir / "detections.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            st.success(f"✅ JSON 结果已保存: {json_path}")
        
        # 保存日志
        if save_log:
            log_dir = Path("../outputs/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"batch_{output_name}.log"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_lines))
            st.success(f"✅ 日志已保存: {log_file}")
        
        # 完成提示
        status_text.text(f"✅ 批量处理完成！成功处理 {success_count}/{len(image_ids)} 张图片")
        
        # 显示统计
        with results_container:
            st.subheader("处理统计")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("总计", len(image_ids))
            with col_b:
                st.metric("成功", success_count)
            with col_c:
                st.metric("失败", len(image_ids) - success_count)
            
            # 显示最近日志
            with st.expander("查看日志 (最后20条)"):
                for line in log_lines[-20:]:
                    st.text(line)


# ==================== 主应用 ====================

def main():
    st.set_page_config(
        page_title="DETR 交通场景检测演示",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 标题
    st.title("🚦 DETR 交通场景目标检测演示系统")
    st.markdown("---")
    
    # Tab 导航
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 预训练模型推理",
        "📊 Ground Truth 可视化",
        "🔬 预测与 GT 对比",
        "📦 批量导出与日志"
    ])
    
    with tab1:
        page_inference()
    
    with tab2:
        page_gt_visualization()
    
    with tab3:
        page_comparison()
    
    with tab4:
        page_batch_export()
    
    # 页脚
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        DETR Traffic Analysis Demo | Powered by Hugging Face Transformers & Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
