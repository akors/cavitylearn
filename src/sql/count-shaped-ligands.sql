SELECT ligands, count(*) FROM fridge_cavities WHERE shaped = 1 AND cavitysearch_uuid IN ("568BC29C-9B76-4933-8448-6190A4E9E20D", "769EE068-BADF-49C5-AEA4-9B5CA1DED878" ) GROUP BY ligands ORDER BY count(*) ASC;

