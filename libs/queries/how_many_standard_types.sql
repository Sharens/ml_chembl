SELECT AVG(standard_value) AS AVG_NULL from activities where standard_type = 'IC50'
                                             and standard_units IS NULL
                                             and standard_value IS NOT NULL
-- 406.41057260517897
SELECT AVG(standard_value) AS AVG_nM from activities where standard_type = 'IC50'
                                             and standard_units = 'nM'
                                             and standard_value IS NOT NULL

-- 1418802120433858050

SELECT standard_units, COUNT(*) as count
FROM activities
where standard_type = 'IC50'
GROUP BY standard_units order by count desc;

-- nM,3276327
-- <null>,196043 <- to jest poważny problem
-- ug.mL-1,78221
-- mg kg-1,253
-- ug,241
-- molar ratio,205
-- ppm,177
-- %,157
-- 10^2 uM,115
-- equiv,89
-- 10^-7mol/L,81
-- 10^-9mol/L,74
-- % conc,70
-- 10^-8mol/L,60
-- uM well-1,52
-- p.p.m.,51
-- 10^-5 mol/L,47
-- g/ha,40
-- mMequiv,38
-- 10^3nM,31
-- 10^-5 uM,29
-- uL,27
-- 10^-4microM,25
-- /uM,25
-- milliequivalent,22
-- ug/g,20
-- kJ m**-2,20
-- 10^-6 mol/L,18
-- 10'-5g/L,18
-- pmol/L,16
-- M ml-1,15
-- ug well-1,12
-- umol/Kg,11
-- ug/mol,11
-- mmol/Kg,10
-- uM tube-1,9
-- mol,9
-- min,9
-- 10^-2microM,9
-- 10^-10mol/L,9
-- 10'-4nM,9
-- ppm g dm**-3,7
-- nM/unit,7
-- 10^3 uM,7
-- ucm,6
-- nmol/mg,6
-- nA,6
-- microg/cm3,6
-- 10^2pM,6
-- 10'-4g/L,6
-- liposomes ml-1,5
-- umol/dm3,4
-- nM kg-1,4
-- mg.min/m3,4
-- 10^2 nM,4
-- 10^-1microM,4
-- 10'5mM,4
-- 10'-8mg/ml,4
-- 10'-10 uM,4
-- nmol/Kg,3
-- ml,3
-- mg Kg-1,3
-- 10^-6 uM,3
-- 10^-4nM,3
-- 10'5pM,3
-- 10'-3g/L,3
-- ucm/s,2
-- 10^4 uM,2
-- 10^-8nmol,2
-- 10^-6nmol,2
-- 10^-5 ug/ml,2
-- 10^-3mg/ml,2
-- 10'8nM,2
-- 10'6uM,2
-- 10'6pM,2
-- 10'3pM,2
-- ug/kg,1
-- ug cm**-2,1
-- uM.hr,1
-- uL/ml,1
-- nmol/min,1
-- ng/mg,1
-- nM/g,1
-- microA,1
-- mg/kg/day,1
-- mg/cm3,1
-- 10^5uM,1
-- 10^-6 ug/ml,1
-- 10^-5mg/ml,1
-- 10^-4mg/ml,1
-- 10^-4 ug/ml,1
-- 10^-3umol/L,1
-- 10'7nM,1
-- 10'5uM,1
-- 10'20 uM,1
-- 10'16 uM,1
-- 10'13nM,1
-- 10'-4umol/L,1
-- 10'-13 ug/ml,1
-- 10'-11uM,1
-- /uM/s,1
