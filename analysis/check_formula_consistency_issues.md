# Issues identifiés dans `check_formula_consistency`

Cette analyse se concentre sur les problèmes observés dans l'implémentation fournie de `check_formula_consistency`.

## 1. Incohérence de signature avec l'exemple d'utilisation
L'exemple d'appel du docstring utilise le paramètre `require_denominators_for_complex=True`, mais la signature de la fonction ne définit pas ce paramètre. Toute tentative de reproduire l'exemple provoquerait une erreur `TypeError` avant même d'exécuter la logique principale.

## 2. Expressions non valides avec des noms de colonnes complexes
Les formules d'exemple contiennent des noms de colonnes comportant des espaces, parenthèses et autres caractères spéciaux (ex. `SiteEUI(kBtu/sf)`). Dans `pandas.DataFrame.eval`, de tels noms doivent impérativement être entourés de backticks. Or la fonction transmet l'expression telle quelle à `df.eval` et n'applique pas de normalisation. Résultat : les formules d'exemple lèvent une erreur de syntaxe au lieu d'être évaluées.

## 3. Extraction partielle des colonnes utilisées dans la formule
Le code tente de récupérer les colonnes référencées dans la formule via `re.findall(r"`([^`]+)`", expr)`. Si l'utilisateur n'emploie pas de backticks (comme dans l'exemple fourni), cette extraction renvoie une liste vide. Cela vide les colonnes de contexte censées aider au diagnostic, même lorsque la formule est correctement évaluée. Le comportement est donc incohérent avec les exemples.

## 4. Calcul erroné de l'écart relatif quand la valeur attendue est nulle
Le calcul de l'écart relatif remplace les zéros par `NaN` (`computed[mask_valid].replace(0, np.nan)`). Ainsi, lorsqu'une ligne respecte parfaitement la formule mais que la valeur attendue vaut 0, l'écart relatif devient `NaN` et pollue la moyenne (`mean_rel_diff`). On s'attendrait plutôt à un écart relatif nul dans ce cas.

## 5. Libellé ambigu dans le rapport verbose
Le récapitulatif affiché en mode `verbose` indique « Lignes valides » pour compter les lignes utilisées dans la comparaison. Ce libellé laisse entendre qu'il s'agit du nombre de lignes conformes, alors qu'il s'agit du volume total de lignes analysées après filtrage (`mask_valid.sum()`). Renommer cette mention (par exemple en « Lignes comparées ») clarifierait qu'il s'agit d'un dénombre des lignes prises en compte, indépendamment de leur statut conforme ou non.
