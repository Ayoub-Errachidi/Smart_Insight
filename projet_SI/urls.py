from django.urls import path
from projet_SI.views import views_importation, data_cleaning, data_informations, data_statistiques, data_visualisations,data_preprocessing

urlpatterns = [
    # Page d'accueil (redirection vers dashboard)
    path('', views_importation.dashboard_view, name='home'),
    
    # Dashboard principal
    path('dashboard/', views_importation.dashboard_view, name='dashboard'),

    # Affichage du dataset
    path('dataset/', views_importation.dataset_view, name='dataset'),

    # Informations générales sur le dataset
    path('informations_dataset/', data_informations.information_view, name='informations_dataset'),

    # Statistiques descriptives du dataset
    path('statistiques/', data_statistiques.statistique_view, name='statistiques'),


    # 📉 Choix du type de visualisation
    path('visualisations/', data_visualisations.visualisations_view, name='visualisations'),
    
    # 🖼️ Génération de graphiques dynamiques (plot_type : histogram, box, scatter ...)
    path('select_pairplot/', data_visualisations.select_pairplot_columns_view, name='select_pairplot_columns_view'),
    path('generate_plot/<str:plot_type>/', data_visualisations.plot_result_view, name='generate_plot'),




    # Nettoyage des données
    path('nettoyage/', data_cleaning.nettoyer_dataset_view, name='nettoyage'),

    # Supprimer les doublons
    path('supprimer_doublons/', data_cleaning.supprimer_doublons_view, name='supprimer_doublons'),

    # Nettoyage des outliers
    path('nettoyer_outliers/', data_cleaning.traiter_outliers_view, name='nettoyer_outliers'),
    
    # Supprimer des colonnes
    path("supprimer_colonnes/", data_cleaning.supprimer_colonnes_page, name="supprimer_colonnes_page"),
    path("supprimer_colonnes/action/", data_cleaning.supprimer_colonnes_action, name="supprimer_colonnes_action"),

    # Téléchargement du dataset nettoyé
    path('download_dataset/', data_cleaning.download_dataset_view, name='download_dataset'),

    # Téléchargement du dataset nettoyé
    path('choix_cible/', data_preprocessing.choix_cible_view, name='choix_cible'),

    # Transformation des colonnes
    path('transformation_colonnes/', data_preprocessing.transformer_colonnes_view, name='transformation_colonnes'),

    # Telecharger dataset transforme
    path('telecharger_dataset_transforme/', data_preprocessing.telecharger_dataset_transforme, name='telecharger_dataset_transforme'),

    # Entraînement du Modèle 
    #path("entrainer_modeles/", model_training.model_training_view, name="entrainer_modeles")
    

]