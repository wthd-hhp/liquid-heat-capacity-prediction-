# ===============================
# 📘 Streamlit + Chemprop GNN 热容预测 Web 应用
# ===============================
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
import pandas as pd
import numpy as np
import re
import os
import tempfile
import torch
import gc
from chemprop.train import make_predictions
from chemprop.args import PredictArgs

# ---------------- 页面样式 ----------------
st.markdown("""
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 50%;
        background-color: #f9f9f9;
        padding: 20px;
    }
    .rounded-container h2 {
        text-align: center;
        background-color: #e0e0e0;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- 页面标题 ----------------
st.markdown("""
<div class='rounded-container'>
    <h2>🔬 Predict Heat Capacity (Cp) of Organic Molecules</h2>
    <blockquote>
        Enter a valid <b>SMILES</b> string, and this app will predict the 
        <b>heat capacity (Cp)</b> using a trained <b>Chemprop GNN model</b>.
    </blockquote>
</div>
""", unsafe_allow_html=True)

# ---------------- 用户输入 ----------------
smiles = st.text_input("🧪 Enter SMILES string:", placeholder="e.g., C1=CC=CC=C1O")
submit_button = st.button("🚀 Predict")

# ---------------- Chemprop 模型预测函数 ----------------
def chemprop_predict(smiles_list):
    """使用 Chemprop 模型进行热容预测"""
    try:
        model_dir = "./chemprop_model"
        if not os.path.exists(model_dir):
            raise FileNotFoundError("❌ Chemprop model folder not found. Please upload './chemprop_model/'.")

        # 创建临时输入输出文件
        temp_input = os.path.join(tempfile.gettempdir(), "chemprop_input.csv")
        temp_output = os.path.join(tempfile.gettempdir(), "chemprop_output.csv")
        pd.DataFrame({"smiles": smiles_list}).to_csv(temp_input, index=False)

        # 构建 Chemprop 参数
        args = PredictArgs().parse_args([
            "--test_path", temp_input,
            "--checkpoint_dir", model_dir,
            "--preds_path", temp_output,
        ])

        # 自动检测 GPU / CPU
        if torch.cuda.is_available():
            st.info("🚀 GPU detected — using CUDA for prediction.")
            args.no_cuda = False
        else:
            st.warning("⚙️ No GPU detected — switching to CPU mode.")
            args.no_cuda = True

        # 🔧 安全映射：即便模型在 GPU 训练，也能在 CPU 环境加载
        torch.load = lambda *a, **kw: torch.load(*a, map_location="cpu", **kw)

        with st.spinner("🔬 Running Chemprop prediction..."):
            make_predictions(args=args)

        preds = pd.read_csv(temp_output).iloc[:, -1].tolist()

        # 清理临时文件
        if os.path.exists(temp_input): os.remove(temp_input)
        if os.path.exists(temp_output): os.remove(temp_output)

        return preds

    except Exception as e:
        raise RuntimeError(f"Chemprop prediction failed: {str(e)}")

# ---------------- 分子绘图 ----------------
def mol_to_image(mol, size=(300, 300)):
    d2d = MolDraw2DSVG(size[0], size[1])
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    svg = re.sub(r"<rect[^>]*>", "", svg, flags=re.DOTALL)
    return svg

# ---------------- 主逻辑 ----------------
if submit_button:
    if not smiles:
        st.error("❗Please enter a valid SMILES string.")
    else:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                st.error("❌ Invalid SMILES format.")
                st.stop()

            # 绘制结构图
            mol = Chem.AddHs(mol)
            AllChem.Compute2DCoords(mol)
            svg = mol_to_image(mol)
            st.markdown(f'<div style="text-align:center;">{svg}</div>', unsafe_allow_html=True)

            # 分子量
            mol_weight = Descriptors.MolWt(mol)
            st.markdown(f"**Molecular Weight:** {mol_weight:.2f} g/mol")

            # 🔮 Chemprop GNN 预测
            preds = chemprop_predict([smiles])
            st.success(f"✅ **Predicted Heat Capacity (Cp): {preds[0]:.2f} J/(mol·K)**")

            gc.collect()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
