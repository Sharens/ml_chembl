SELECT standard_type, COUNT(*) AS occurrences
FROM activities
GROUP BY standard_type
ORDER BY occurrences DESC;

-- Potency,4473542
-- IC50,3552865
-- GI50,2629061
-- Inhibition,1593108
-- Activity,1357448
-- Percent Effect,1328366
-- Ki,880730
-- k_off,826806
-- kon,826637
-- MIC,780708
-- EC50,601331
-- INHIBITION,339133
-- AC50,278273

-- czy mozna przelicac AC50 na IC50
-- EC50 / AC50: Często używane zamiennie z IC50 w testach funkcjonalnych.
-- Jeśli test dotyczył hamowania, $AC50 \approx IC50$.

-- EC50 / AC50: Często używane zamiennie z IC50 w testach funkcjonalnych.
-- Jeśli test dotyczył hamowania, $AC50 \approx IC50$.GI50 (Growth Inhibition):
-- Używane w testach cytotoksyczności (np. na liniach nowotworowych).
-- Można je traktować jako "komórkowe IC50".

--Przeliczanie Ki na IC50 (i odwrotnie)
-- To najczęstsza konwersja. Ki to stała dysocjacji kompleksu enzym-inhibitor (wartość bezwzględna),
-- a IC50 zależy od stężenia substratu użytego w teście.
-- Do przeliczenia stosuje się równanie Chenga-Prusoffa:$$IC50 = Ki \cdot \left(1 + \frac{[S]}{K_m}\right)$$