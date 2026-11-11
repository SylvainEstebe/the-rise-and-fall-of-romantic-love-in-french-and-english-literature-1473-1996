# Chiffres complémentaires pour les sections 3.3 et 4.2

Toutes les estimations portent sur **Romantic affiliation (ING_38)**. Contrastes = score attendu à la date finale moins celui à la date initiale ; médiane [intervalle crédible central à 95 %], à partir des 16 000 tirages finaux, à effet auteur nul. Aucun réajustement. Les modèles ci-dessous sont tous à moins de deux SE du meilleur ELPD de leur corpus. La spline df = 3 est un polynôme cubique global ; df = 4–6 sont les splines flexibles de la comparaison principale. Les intervalles propres à chaque modèle ne sont pas des intervalles intégrant l’incertitude de sélection.

## 1. Livres français

| Modèle | 1700→1800 | 1800→1900 | 1900→1996 |
| --- | --- | --- | --- |
| quadratic | -0.41 [-0.52, -0.31] | -0.94 [-1.14, -0.75] | -1.11 [-1.26, -0.91] |
| spline_3 | -0.76 [-0.96, -0.56] | -0.83 [-1.02, -0.64] | -0.21 [-0.73, 0.42] |
| spline_4 | -0.71 [-0.90, -0.50] | -1.11 [-1.40, -0.82] | 0.79 [-0.21, 1.80] |
| spline_5 | -0.80 [-1.11, -0.49] | -1.03 [-1.36, -0.70] | 0.62 [-0.61, 1.91] |
| spline_6 (affiché) | -0.73 [-1.06, -0.40] | -1.10 [-1.44, -0.75] | -0.14 [-1.18, 1.29] |

## 2. Date de la baisse la plus forte — livres français

| Modèle | Année t [95 %] | P(baisse maximale au XIXe) | P(premier intervalle) | P(dernier intervalle) | P(taux moyen 1800–1900 le plus négatif) |
| --- | --- | --- | --- | --- | --- |
| quadratic | 1930 [1912, 1954] | 0.0 % | 0.0 % | 0.0 % | 0.1 % |
| spline_3 | 1810 [1782, 1867] | 73.1 % | 0.0 % | 0.0 % | 70.3 % |
| spline_4 | 1823 [1807, 1838] | 99.2 % | 0.0 % | 0.0 % | 98.2 % |
| spline_5 | 1818 [1785, 1862] | 78.5 % | 1.0 % | 1.2 % | 79.8 % |
| spline_6 | 1824 [1794, 1995] | 54.2 % | 1.3 % | 32.5 % | 84.2 % |

Définition : minimiser E(t+1) − E(t), séparément dans chaque tirage, sur 1500–1996. L’année rapportée est le début t de cet intervalle annuel ; c’est une approximation annuelle de la pente instantanée la plus négative, et non le maximum ni le franchissement de −0,25 point. La dernière variation est 1995→1996. Les CSV donnent aussi les milieux des intervalles, les taux de baisse et une sensibilité sur des différences de dix ans.

**Interprétation :** « steepest in the nineteenth century » n’est pas robuste à toutes les spécifications plausibles. La spline df = 6 affichée place la baisse annuelle maximale au XIXe siècle dans 54,2 % des tirages, et dans le dernier intervalle dans 32,5 %. La probabilité est 99,2 % pour df = 4 et 78,5 % pour df = 5. La quadratique, également à moins de deux SE du meilleur modèle, la place au XXe siècle. Le classement des taux moyens par période répond à une autre question et tient compte de la durée de 96 ans pour 1900–1996.

**Après 1900 :** tous les intervalles des splines incluent zéro pour 1900→1996 dans les trois corpus. Cela ne démontre ni une absence de baisse ni une stabilisation. Les quadratiques prédisent encore une baisse dans les trois corpus (pour Babel anglais, l’intervalle est affiché à trois décimales pour conserver le signe de sa borne supérieure). Les différences de période entre analyses quadratiques, splines et modèles à deux composantes doivent rester explicites.

## 3a. Babel français

| Modèle | 1800→1900 | 1900→1996 |
| --- | --- | --- |
| quadratic | -0.41 [-0.62, -0.20] | -0.59 [-0.92, -0.24] |
| spline_3 | -0.42 [-0.68, -0.17] | -0.38 [-0.86, 0.13] |
| spline_4 | -0.40 [-0.76, -0.05] | -0.41 [-0.98, 0.22] |
| spline_5 | -0.46 [-1.00, 0.07] | -0.37 [-0.95, 0.25] |
| spline_6 (affiché) | -0.47 [-0.97, 0.01] | -0.30 [-0.88, 0.34] |

## 3b. Babel anglais

| Modèle | 1800→1900 | 1900→1996 |
| --- | --- | --- |
| quadratic | -0.33 [-0.54, -0.11] | -0.388 [-0.769, -0.005] |
| spline_3 | -0.55 [-0.90, -0.19] | -0.20 [-0.62, 0.23] |
| spline_4 (affiché) | -0.56 [-0.90, -0.20] | -0.37 [-0.83, 0.10] |
| spline_5 | -0.67 [-1.04, -0.29] | -0.29 [-0.78, 0.20] |
| spline_6 | -0.82 [-1.27, -0.35] | -0.27 [-0.77, 0.22] |

## 4. Babel : XVIIIe versus XIXe siècle

| Corpus | Période | N / N positifs | Proportion positive, % [95 %] | Moyenne parmi positifs [95 %] |
| --- | --- | --- | --- | --- |
| babel_fr | 1700–1799 | 296 / 230 | 77.7 [68.3, 86.6] | 3.96 [3.52, 4.36] |
| babel_fr | 1800–1899 | 690 / 492 | 71.3 [63.6, 80.1] | 3.37 [3.07, 3.63] |
| babel_uk | 1700–1799 | 26 / 22 | 84.6 [64.3, 100.0] | 4.14 [3.46, 4.77] |
| babel_uk | 1800–1899 | 156 / 120 | 76.9 [68.7, 87.3] | 3.79 [3.27, 4.34] |

| Corpus | Δ proportion, points de % | Δ moyenne parmi positifs |
| --- | --- | --- |
| babel_fr | -6.4 [-18.7, 6.7] | -0.59 [-1.10, -0.09] |
| babel_uk | -7.7 [-25.1, 14.7] | -0.34 [-1.23, 0.61] |

Estimations observées et intervalles bootstrap percentile à 95 % : 3 000 rééchantillonnages de groupes d’auteurs, seed 42, moyennes pondérées par le nombre d’observations. Un même tirage sert aux deux périodes. Le flux aléatoire de l’analyse antérieure est reproduit exactement, et ses intervalles des différences ont été retrouvés à 10⁻¹² près. Positif signifie score > 0. Babel anglais ne comprend que 26 observations au XVIIIe siècle, dont 22 positives.

La baisse de la moyenne conditionnelle est étayée en français (−0,59 [−1,10 ; −0,09]), mais pas clairement en anglais. Les changements de proportion sont incertains dans les deux langues. Les contributions symétriques de chaque composante sont également fournies dans le CSV ; elles sont descriptives et non causales.

## 5. Degrés de liberté affichés dans la Figure 1

| dataset | displayed_model | selection_scope |
| --- | --- | --- |
| french_books | spline_6 | highest ELPD among spline df 4, 5, 6 |
| english_books | spline_4 | highest ELPD among spline df 4, 5, 6 |
| french_theatre | spline_6 | highest ELPD among spline df 4, 5, 6 |
| english_theatre | spline_6 | highest ELPD among spline df 4, 5, 6 |

Choix par meilleur ELPD parmi df = 4, 5, 6 : livres français **6**, livres anglais **4**, théâtre français **6**, théâtre anglais **6**.
