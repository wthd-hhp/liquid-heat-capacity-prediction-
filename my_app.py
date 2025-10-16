import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
import pandas as pd
import numpy as np
import os
import re
import tempfile
import torch
from chemprop.train import make_predictions
from chemprop.args import PredictArgs


# ---------------- 页面样式 ----------------
st.markdown("""
<style>
.stApp {
    border: 2px solid #808080;
    border-radius: 20px;
    margin: 50px auto;
    max-width: 45%;
    background-color: #f9f9f9f9;
    padding: 20px;
}
.rounded-container h2 {
    text-align: center;
    background-color: #e0e0e0e0;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- 页面标题 ----------------
st.markdown("""
<div class='rounded-container'>
    <h2>Predict Heat Capacity (Cp) of Organic Molecules</h2>
    <blockquote>
        This app predicts the heat capacity (Cp) of organic molecules using a pretrained Chemprop Graph Neural Network (GNN) model.<br>
        Please enter a valid SMILES string to begin.
    </blockquote>
</div>
""", unsafe_allow_html=True)

# ---------------- 用户输入 ----------------
smiles = st.text_input("Enter SMILES:", placeholder="e.g., C1=CC=CC=C1O")
submit_button = st.button("Submit and Predict")

# ---------------- Chemprop 模型预测函数 ----------------
def chemprop_predict(smiles_list):
    """使用 Chemprop GNN 模型预测热容"""
    try:
        model_dir = "./chemprop_model"  # 你的模型文件夹路径
        if not os.path.exists(model_dir):
            raise FileNotFoundError("❌ Chemprop model folder not found. Please upload './chemprop_model/'")

        # 临时输入输出文件
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        pd.DataFrame({"smiles": smiles_list}).to_csv(temp_input.name, index=False)
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")

        # 构建参数
        args = PredictArgs().parse_args([
            "--test_path", temp_input.name,
            "--checkpoint_dir", model_dir,
            "--preds_path", temp_output.name,
        ])

        # 无 GPU 则禁用 CUDA
        if not torch.cuda.is_available():
            args.no_cuda = True

        # 执行预测
        with st.spinner("Running Chemprop (GNN) prediction..."):
            make_predictions(args=args)

        preds = pd.read_csv(temp_output.name).iloc[:, -1].tolist()

        os.remove(temp_input.name)
        os.remove(temp_output.name)
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
        st.error("Please enter a valid SMILES string.")
    else:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                st.error("Invalid SMILES format.")
                st.stop()

            # 分子绘制
            mol = Chem.AddHs(mol)
            AllChem.Compute2DCoords(mol)
            svg = mol_to_image(mol)
            st.markdown(f'<div style="text-align:center;">{svg}</div>', unsafe_allow_html=True)

            # 分子量信息
            mol_weight = Descriptors.MolWt(mol)
            st.markdown(f"**Molecular Weight:** {mol_weight:.2f} g/mol")

            # 调用 Chemprop 预测
            try:
                preds = chemprop_predict([smiles])
                st.success(f"**Predicted Heat Capacity (Cp): {preds[0]:.2f} J/(mol·K)**")
            except Exception as chem_error:
                st.warning(str(chem_error))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
