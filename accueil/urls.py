from django.urls import path
from .views import liste_taches, ajouter_tache, saluer

# Namespace de l'application pour les URLs réversibles
app_name = 'accueil'

urlpatterns = [
    path('', liste_taches, name='liste'),
    path('ajouter/', ajouter_tache, name='ajouter_tache'),
    path('saluer/', saluer, name='saluer'),
]