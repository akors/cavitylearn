SELECT uuid FROM fridge_cavities
WHERE status = 3
AND A_n != 0 AND C_n != 0 AND d_n != 0 AND e_n !=0
AND shaped = 1
AND cavitysearch_uuid IN ("568BC29C-9B76-4933-8448-6190A4E9E20D", "7B2361DA-D125-4E2E-877F-450804C5CFC6" )
AND ligands IN ("GOL", "HEM")
AND volume < 745 -- (0.375 * 30)^3 * 1/6 * pi
AND volume > 20
AND coverage_ligand_cavity > 50
