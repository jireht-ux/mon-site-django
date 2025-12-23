from django import forms

class NomForm(forms.Form):
    nom = forms.CharField(label="Votre nom", max_length=100)


from django.forms import ModelForm
from .models import Tache


class TacheForm(ModelForm):
    """Formulaire lié au modèle Tache — expose uniquement title et done.

    Les labels sont francisés selon la demande : 'titre' et 'termine'.
    """

    class Meta:
        model = Tache
        fields = ["title", "done"]
        labels = {
            "title": "titre",
            "done": "termine",
        }