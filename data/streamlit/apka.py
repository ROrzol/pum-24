import streamlit as st
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen

# Load the model
with open("C:\\Users\\HP\\Desktop\\pum\\pum-24\\data\\modele\\model3.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Featurizer class
class Featurizer:
    def featurize(self, smiles):
        df = pd.DataFrame(smiles, columns=['SMILES'])
        return self.generate_descriptors(df)

    def generate_descriptors(self, df):
        df['mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
        df['mol_wt'] = df['mol'].apply(rdMolDescriptors.CalcExactMolWt)
        df['logp'] = df['mol'].apply(Crippen.MolLogP)
        df['num_heavy_atoms'] = df['mol'].apply(rdMolDescriptors.CalcNumHeavyAtoms)
        df['num_HBD'] = df['mol'].apply(rdMolDescriptors.CalcNumHBD)
        df['num_HBA'] = df['mol'].apply(rdMolDescriptors.CalcNumHBA)
        df['aromatic_rings'] = df['mol'].apply(rdMolDescriptors.CalcNumAromaticRings)
        return df[['mol_wt', 'logp', 'num_heavy_atoms', 'num_HBD', 'num_HBA', 'aromatic_rings']].values.tolist()

featurizer = Featurizer()

# Streamlit app
st.title("Solubility Prediction Model")
st.write("Enter SMILES of your molecule to predict its solubility.")

smiles_input = st.text_input("Enter SMILES here")
if st.button("Predict solubility"):
    descriptors = featurizer.featurize([smiles_input])
    if descriptors:
        solubility = model.predict(descriptors)[0]
        st.write(f"Predicted Solubility: {solubility}")
    else:
        st.write("Invalid SMILES")
