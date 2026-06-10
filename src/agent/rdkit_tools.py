from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def _validate(smiles: str) -> tuple[str, Chem.Mol] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol)
    return (canonical, mol)


def compute_molecular_weight(smiles: str) -> dict:
    validated = _validate(smiles)
    if validated is None:
        return {"valid": False, "error": "Invalid SMILES string", "smiles": smiles}
    canonical, mol = validated
    mw = Descriptors.MolWt(mol)
    return {"valid": True, "smiles": canonical, "molecular_weight": round(mw, 2)}


def compute_logp(smiles: str) -> dict:
    validated = _validate(smiles)
    if validated is None:
        return {"valid": False, "error": "Invalid SMILES string", "smiles": smiles}
    canonical, mol = validated
    logp = Descriptors.MolLogP(mol)
    return {"valid": True, "smiles": canonical, "logp": round(logp, 4)}


def compute_descriptors(smiles: str) -> dict:
    validated = _validate(smiles)
    if validated is None:
        return {"valid": False, "error": "Invalid SMILES string", "smiles": smiles}
    canonical, mol = validated
    return {
        "valid": True,
        "smiles": canonical,
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 4),
        "hba": Lipinski.NumHAcceptors(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "psa": round(Descriptors.TPSA(mol), 2),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
    }
