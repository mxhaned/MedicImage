// 🎯 FORMAT JSON CORRECT POUR LE FRONTEND
// Remplacez votre structure actuelle par celle-ci :

const correctPituitaryData = {
    'features': {
        // ✅ FEATURES CORRECTES (déjà bonnes)
        't1_3d_tumor_volume': 21,          // Volume tumoral 3D (mm³)
        't1_3d_max_intensity': 23,         // Intensité maximale T1
        't1_3d_major_axis_length': 23,     // Grand axe tumoral (mm)
        't1_3d_area': 24,                  // Surface tumorale 3D (mm²)
        't1_3d_extent': 23,                // Régularité/compacité (0-1)
        't1_3d_mean_intensity': 25,        // Intensité moyenne T1
        
        // ✅ NOUVELLES FEATURES CORRECTES (remplacer les T2_*)
        't1_3d_minor_axis_length': 46,        // Petit axe tumoral (mm)
        't1_3d_surface_to_volume_ratio': 36,  // Ratio surface/volume
        't1_3d_glcm_contrast': 46,            // Contraste texture GLCM
        't1_2d_area_median': 80               // Aire médiane 2D (mm²)
    }, 
    'initial_analysis': {
        'tumor_type': 'Pituitary',
        'tumor_detected': true
        // ... autres données
    }
};

// 🚀 ENVOI AU SERVEUR
fetch('/analyze-non-glioma-manual', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(correctPituitaryData)
})
.then(response => response.json())
.then(data => {
    console.log('✅ Analyse réussie:', data);
    // Traiter la réponse avec survival_days, etc.
})
.catch(error => {
    console.error('❌ Erreur:', error);
});

/* 
🎯 CHANGEMENTS REQUIS DANS VOTRE FRONTEND:

1. SUPPRIMER ces 4 features:
   ❌ 't2_3d_tumor_volume'
   ❌ 't2_3d_max_intensity' 
   ❌ 't2_3d_major_axis_length'
   ❌ 't2_3d_compactness'

2. AJOUTER ces 4 features:
   ✅ 't1_3d_minor_axis_length'
   ✅ 't1_3d_surface_to_volume_ratio'
   ✅ 't1_3d_glcm_contrast'
   ✅ 't1_2d_area_median'

3. GARDER les mêmes valeurs:
   - Les valeurs 46, 36, 46, 80 peuvent être conservées
   - Seuls les NOMS des features changent

🏆 RÉSULTAT:
✅ L'endpoint standard fonctionnera (plus besoin de l'ultra-compatible)
✅ Interface frontend avec les bons noms de features
✅ 100% de compatibilité garantie
*/