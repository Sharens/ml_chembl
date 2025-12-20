SELECT
    act.activity_id,
    act.molregno,
    cs.canonical_smiles,
    cp.mw_freebase,       -- Masa cząsteczkowa
    cp.alogp,             -- LogP (lipofilowość)
    cp.hba,               -- Akceptory wiązań wodorowych
    cp.hbd,               -- Donory wiązań wodorowych
    cp.psa,               -- Polarna powierzchnia (Polar Surface Area)
    cp.rtb,               -- Wiązania rotowalne (elastyczność)
    cp.aromatic_rings,    -- Liczba pierścieni aromatycznych
    cp.qed_weighted,      -- Ilościowa ocena lekopodobności (Drug-likeness)
    act.standard_value,
    act.standard_units,
    act.standard_type,
    act.pchembl_value,    -- pIC50 (już zlogarytmowane)
    td.chembl_id AS target_chembl_id,
    td.pref_name AS target_name
FROM activities act
JOIN assays ass ON act.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
JOIN compound_structures cs ON act.molregno = cs.molregno
JOIN compound_properties cp ON act.molregno = cp.molregno
WHERE td.organism = 'Homo sapiens'
and td.chembl_id = 'CHEMBL203'
    AND act.pchembl_value IS NOT NULL    -- Musi istnieć wartość celu
    AND cs.canonical_smiles IS NOT NULL   -- Musimy mieć strukturę dla GNN
    AND (act.potential_duplicate IS NULL OR act.potential_duplicate = 0)
LIMIT 2000000 -- mozna zmienic w zaleznosci od zasobów
;


select * from target_dictionary trg where trg.chembl_id = 'CHEMBL203'