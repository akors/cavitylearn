SELECT uuid FROM fridge_cavities 
WHERE shaped = 1
AND cavitysearch_uuid IN ("568BC29C-9B76-4933-8448-6190A4E9E20D", "769EE068-BADF-49C5-AEA4-9B5CA1DED878" )
AND ligands IN ("NA", "ACT", "MN", "CA", "PO4", "NAG", "HEM", "CL", "ZN", "EDO", "MG", "GOL", "SO4")
AND volume < 1400
