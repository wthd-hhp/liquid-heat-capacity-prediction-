import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import re
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
from chemprop.train import make_predictions
from chemprop.args import PredictArgs

# ---------------- 页面样式 ----------------
st.markdown("""
<style>
.stApp {
    border: 2px solid #808080;
    border-radius: 20px;
    margin: 50px auto;
    max-width: 40%;
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
        This web app predicts the heat capacity (Cp) of organic molecules 
        using a trained <b>Chemprop</b> graph neural network (GNN) model.<br><br>
        Please enter a valid SMILES string below.
    </blockquote>
</div>
""", unsafe_allow_html=True)

# ---------------- 用户输入 ----------------
smiles = st.text_input("Enter the SMILES representation of the molecule:", placeholder="e.g., C1=CC=CC=C1O")
submit_button = st.button("Submit and Predict")

# ---------------- 分子绘图函数 ----------------
def mol_to_image(mol, size=(300, 300)):
    """绘制分子结构为SVG"""
    d2d = MolDraw2DSVG(size[0], size[1])
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    svg = re.sub(r"<rect[^>]*>", "", svg, flags=re.DOTALL)
    return svg

# ---------------- Chemprop 预测函数 ----------------
def chemprop_predict(smiles_list):
    """使用 Chemprop 模型进行热容预测"""
    try:
        model_dir = "./chemprop_model"
        if not os.path.exists(model_dir):
            raise FileNotFoundError("❌ Chemprop model folder not found in './chemprop_model/'.")

        # 临时输入输出文件
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        pd.DataFrame({"smiles": smiles_list}).to_csv(temp_input.name, index=False)

        # 构建 Chemprop 参数
        args = PredictArgs().parse_args([
            "--test_path", temp_input.name,
            "--checkpoint_dir", model_dir,
            "--preds_path", temp_output.name,
        ])

        if not torch.cuda.is_available():
            args.no_cuda = True

        make_predictions(args=args)

        preds = pd.read_csv(temp_output.name).iloc[:, -1].tolist()
        os.remove(temp_input.name)
        os.remove(temp_output.name)

        return preds

    except Exception as e:
        raise RuntimeError(f"Chemprop prediction failed: {str(e)}")

# ---------------- 主逻辑 ----------------
if submit_button:
    if not smiles:
        st.error("Please enter a valid SMILES string.")
    else:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                st.error("Invalid SMILES format.")
                st.stop()

            # 绘制分子结构
            mol = Chem.AddHs(mol)
            AllChem.Compute2DCoords(mol)
            svg = mol_to_image(mol)
            st.markdown(f'<div style="text-align:center;">{svg}</div>', unsafe_allow_html=True)

            # 分子量
            mw = Descriptors.MolWt(mol)
            st.markdown(f"**Molecular Weight:** {mw:.2f} g/mol")

            # 调用 Chemprop 模型预测
            with st.spinner("🔬 Running Chemprop prediction..."):
                preds = chemprop_predict([smiles])

            st.success(f"**Predicted Heat Capacity (Cp): {preds[0]:.2f} J/(mol·K)**")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
