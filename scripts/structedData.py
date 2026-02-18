import pandas as pd
import os
def transformer() :
    for i in range(17,23) :
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(script_dir, "..", "data", "input", "insee_data", f"FD_MOBPRO_20{i}.csv")
        output_file = os.path.join(script_dir, "..", "data", "output", "insee_data", f"FD_MOBPRO_20{i}_result.csv")
        cols_to_use = ['REGION', 'REGLT', 'COMMUNE', 'DCLT', 'TRANS', 'IPONDI']
        if i == 16 :
            # Dictionnaire de correspondance pour les modes de transport
            dict_trans = {
                '1': 'Pas de transport (travail domicile)',
                '2': 'Marche à pied',
                '3': 'Deux-roues motorisé',
                '4': 'Voiture, camion, fourgonnette',
                '5': 'Transports en commun',
                'Z': 'Sans objet'
            }
        else :
            # Dictionnaire de correspondance pour les modes de transport
            dict_trans = {
                '1': 'Pas de transport (travail domicile)',
                '2': 'Marche à pied',
                '3': 'Vélo',
                '4': 'Deux-roues motorisé',
                '5': 'Voiture, camion, fourgonnette',
                '6': 'Transports en commun',
                'Z': 'Sans objet'
            }
        
        # 2. Chargement des données
        # On force COMMUNE et DCLT en 'str' pour ne pas perdre les zéros au début des codes postaux
        df = pd.read_csv(input_file, sep=';', usecols=cols_to_use, dtype={'COMMUNE': str, 'DCLT': str, 'TRANS': str})
        
        # 3. Filtre sur l'Île-de-France (REGION 11 et REGLT 11)
        df_idf = df[(df['REGION'] == 11) & (df['REGLT'] == 11)].copy()
        
        # 4. Remplacement des codes par les noms parlants
        df_idf['TRANS'] = df_idf['TRANS'].map(dict_trans)
        
        # 5. Calcul des flux pondérés
        # On groupe par communes et par mode de transport, puis on fait la somme des poids
        flux_pondere = df_idf.groupby(['COMMUNE', 'DCLT', 'TRANS'])['IPONDI'].sum().reset_index()
        
        # 6. Pivot pour mettre les modes de transport en colonnes
        df_final = flux_pondere.pivot_table(
            index=['COMMUNE', 'DCLT'], 
            columns='TRANS', 
            values='IPONDI', 
            aggfunc='sum'
        ).fillna(0)
        
        # 7. Ajout de la colonne Total
        df_final['TOTAL_FLUX'] = df_final.sum(axis=1)
        
        # 8. Exportation
        df_final.reset_index(inplace=True)
        df_final.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
        
        print(f"Fichier généré avec succès : {output_file}")
        
transformer()     
        